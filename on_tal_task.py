import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import wandb
import os
import json
import math
import sys
import copy
import time
import random
import numpy as np
from util.utils import *
import util.misc as utils
import util.logger as loggers
import torchvision.transforms as transforms
from models import build_model
from collections import defaultdict
from dataset import THUMOS14Dataset
from criterion import build_criterion
import cv2
from eval import evaluation_detection
from pathlib import Path
import glob
from collections import OrderedDict


D1_FORBIDDEN_MODEL_INFO = {
    "duration",
    "true_duration",
    "video_time",
    "frame_to_time",
}
D1_RUNTIME_FORBIDDEN_MODEL_INFO = D1_FORBIDDEN_MODEL_INFO | {
    # This is a ground-truth-derived supervision flag.  D1 training may use it
    # for the inherited MATR memory teacher path, but inference/replay must use
    # the learned flag head and must not even expose the label at the boundary.
    "segment_flag",
}
D12_CHECKPOINT_SCHEMA = "eventmatr_d12_independent_birth_v1"
D13_CHECKPOINT_SCHEMA = "eventmatr_d13_factorial_mechanism_v1"
D14_CHECKPOINT_SCHEMA = "eventmatr_d14_decision_alignment_mechanism_v1"


def censor_d1_event_targets_for_model(event_targets, event_valid_mask):
    """Hide a GT endpoint until its first observable END prefix."""

    censored = event_targets.clone()
    valid = event_valid_mask.to(device=censored.device).bool()
    observed_end = censored[..., 7] > 0.5
    censored[..., 3] = censored[..., 3].masked_fill(
        valid & ~observed_end, float("nan")
    )
    return censored


def make_model_inputs(args, feature_inputs, infos, targets=None):
    """Build the causal model boundary while retaining evaluator-only metadata."""

    lifecycle = getattr(args, "event_lifecycle_version", "v1_dense")
    if lifecycle == "d1_censored":
        forbidden = (
            D1_FORBIDDEN_MODEL_INFO
            if bool(getattr(args, "training", False))
            else D1_RUNTIME_FORBIDDEN_MODEL_INFO
        )
        model_infos = {
            key: value
            for key, value in infos.items()
            if key not in forbidden
        }
    else:
        model_infos = infos
    payload = {"inputs": feature_inputs, "infos": model_infos}
    if (
        lifecycle == "d1_censored"
        and targets is not None
        and bool(getattr(args, "training", False))
    ):
        payload["event_targets"] = censor_d1_event_targets_for_model(
            targets["event_targets"], targets["event_valid_mask"]
        )
        payload["event_valid_mask"] = targets["event_valid_mask"]
    return payload


def write_model_predictions(args, model, infos, outputs, path, label_map):
    model_variant = getattr(args, 'model_variant', None)
    if model_variant is None:
        model_variant = 'eventmatr' if getattr(args, 'event_arm', None) else 'native_matr'
    if model_variant == 'native_matr':
        # Strict native evaluator path, independent of every BxO cell.
        make_txt(args, infos, outputs, path, label_map)
    else:
        make_eventmatr_txt(args, model, infos, path, label_map)

def on_tal(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval(args)


def d1_checkpoint_contract(model, args):
    lifecycle = getattr(args, 'event_lifecycle_version', 'v1_dense')
    raw_model = model.module if hasattr(model, 'module') else model
    event_memory = getattr(raw_model, 'event_memory', None)
    owner_state_count = getattr(event_memory, 'owner_state_count', None)
    if lifecycle == 'd1_censored':
        d13_variant = getattr(raw_model, 'event_d13_variant', 'd12_control')
        d14_variant = getattr(raw_model, 'event_d14_variant', 'none')
        if owner_state_count != 3:
            raise RuntimeError(
                'D1 checkpoint contract requires a three-state owner head'
            )
        birth_head = getattr(
            getattr(raw_model, 'event_transition_head', None),
            'birth',
            None,
        )
        if birth_head is None:
            raise RuntimeError(
                'D1 checkpoint contract requires an independent birth head'
            )
        contract = {
            'checkpoint_schema': (
                D14_CHECKPOINT_SCHEMA
                if d14_variant != 'none'
                else (
                    D12_CHECKPOINT_SCHEMA
                    if d13_variant == 'd12_control'
                    else D13_CHECKPOINT_SCHEMA
                )
            ),
            'event_lifecycle_version': lifecycle,
            'event_d1_lane': getattr(args, 'event_d1_lane', None),
            'owner_state_count': owner_state_count,
            'birth_head': 'independent_binary_hazard',
        }
        if d13_variant != 'd12_control':
            contract.update(
                {
                    'event_d13_variant': d13_variant,
                    'association_contract': getattr(
                        raw_model, 'event_association_contract', None
                    ),
                    'birth_risk_contract': getattr(
                        raw_model, 'event_birth_risk_contract', None
                    ),
                }
            )
        if d14_variant != 'none':
            contract.update(
                {
                    'event_d14_variant': d14_variant,
                    'birth_objective_contract': getattr(
                        raw_model, 'event_birth_objective_contract', None
                    ),
                }
            )
        return contract
    return {
        'checkpoint_schema': 'matr_v1_dense_v1',
        'event_lifecycle_version': lifecycle,
        'event_d1_lane': None,
        'owner_state_count': owner_state_count,
        'birth_head': None,
    }


def validate_d1_checkpoint_compatibility(checkpoint, model, args):
    lifecycle = getattr(args, 'event_lifecycle_version', 'v1_dense')
    checkpoint_lifecycle = checkpoint.get('event_lifecycle_version')
    if lifecycle != 'd1_censored':
        if (
            checkpoint_lifecycle is not None
            and checkpoint_lifecycle != lifecycle
        ):
            raise RuntimeError(
                'checkpoint lifecycle mismatch: '
                f'{checkpoint_lifecycle!r} != {lifecycle!r}'
            )
        return
    expected = d1_checkpoint_contract(model, args)
    mismatches = {
        field: {'expected': value, 'actual': checkpoint.get(field)}
        for field, value in expected.items()
        if checkpoint.get(field) != value
    }
    if mismatches:
        raise RuntimeError(
            'D1 checkpoint is incompatible; start fresh instead of mapping '
            'an older lifecycle architecture:\n'
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )


def d1_gradient_metrics(model, args):
    if getattr(args, 'event_lifecycle_version', 'v1_dense') != 'd1_censored':
        return {}
    raw_model = model.module if hasattr(model, 'module') else model
    parameters = {
        'event_transition_gradient_norm': (
            raw_model.event_transition_head.fuse[0].weight
        ),
        'event_birth_gradient_norm': raw_model.event_transition_head.birth.weight,
        'event_owner_gradient_norm': raw_model.event_owner_decoder.state.weight,
    }
    metrics = {}
    for name, parameter in parameters.items():
        gradient = parameter.grad
        value = 0.0 if gradient is None else float(gradient.norm().item())
        if not math.isfinite(value):
            raise RuntimeError(f'non-finite D1.2 gradient: {name}={value}')
        metrics[name] = value
    return metrics


def prepare_d11_effective_dose(args, optimizer, scheduler, loader_batches):
    enabled = bool(getattr(args, 'd11_effective_dose', False))
    if not enabled:
        return None
    if getattr(args, 'study_protocol', None) not in {
        'd11_mechanism',
        'd13_mechanism',
        'd14_mechanism',
    }:
        raise ValueError(
            'd11_effective_dose is restricted to one-epoch D1 mechanism protocols'
        )
    expected_schedule = {
        'epochs': 1,
        'min_lr': 1e-8,
        'max_lr': 1e-5,
        'lr_Tup': 3,
        'lr_Tcycle': 10,
        'lr_gamma': 0.9,
    }
    mismatches = {
        field: {'expected': expected, 'actual': getattr(args, field, None)}
        for field, expected in expected_schedule.items()
        if getattr(args, field, None) != expected
    }
    if mismatches:
        raise ValueError(
            'D1.2 effective-dose schedule drifted:\n'
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )
    if int(loader_batches) != 3270:
        raise ValueError(
            f'D1.2 effective dose requires 3270 loader batches, got {loader_batches}'
        )
    initial_lr = float(optimizer.param_groups[0]['lr'])
    if not math.isclose(initial_lr, args.min_lr, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(
            f'D1.2 optimizer did not start at min_lr: {initial_lr} != {args.min_lr}'
        )
    expected_training_lr = (
        args.min_lr + (args.max_lr - args.min_lr) / args.lr_Tup
    )
    scheduler.step(1)
    training_lr = float(optimizer.param_groups[0]['lr'])
    if not math.isclose(
        training_lr,
        expected_training_lr,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            'D1.2 first warmup learning rate mismatch: '
            f'{training_lr} != {expected_training_lr}'
        )
    return {
        'enabled': True,
        'initial_learning_rate': initial_lr,
        'training_learning_rate': training_lr,
        'expected_optimizer_steps': int(loader_batches),
        'scheduler_last_epoch_at_train_start': int(scheduler.last_epoch),
        'scheduler_t_cur_at_train_start': int(scheduler.T_cur),
    }


def training_state(model, criterion, optimizer, scheduler, epoch, args):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'criterion_dict': criterion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'study_protocol': getattr(args, 'study_protocol', 'upstream_native'),
        'model_variant': getattr(args, 'model_variant', 'native_matr'),
    }
    state.update(d1_checkpoint_contract(model, args))
    return state
        
def train(args):
    save_path = args.save_path
    study_protocol = getattr(args, 'study_protocol', 'upstream_native')
    matched_study = study_protocol == 'matched_study'
    d1_preexperiment = study_protocol == 'd1_preexperiment'
    d11_mechanism = study_protocol == 'd11_mechanism'
    d13_mechanism = study_protocol == 'd13_mechanism'
    d14_mechanism = study_protocol == 'd14_mechanism'
    d1_train_only = (
        d1_preexperiment or d11_mechanism or d13_mechanism or d14_mechanism
    )
    if matched_study and args.epochs != 100:
        raise ValueError('matched_study requires the preregistered terminal epoch 100')
    if d1_preexperiment and args.epochs not in {5, 10, 20}:
        raise ValueError('d1_preexperiment epochs must be one of 5, 10, or 20')
    if d11_mechanism and args.epochs != 1:
        raise ValueError('d11_mechanism requires exactly one epoch')
    if d11_mechanism and args.load_model:
        raise ValueError('d11_mechanism forbids checkpoint resume')
    if d13_mechanism and args.epochs != 1:
        raise ValueError('d13_mechanism requires exactly one epoch')
    if d13_mechanism and args.load_model:
        raise ValueError('d13_mechanism forbids checkpoint resume')
    if d14_mechanism and args.epochs != 1:
        raise ValueError('d14_mechanism requires exactly one epoch')
    if d14_mechanism and args.load_model:
        raise ValueError('d14_mechanism forbids checkpoint resume')

    train_dataset = THUMOS14Dataset(args, subset='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=args.batch, shuffle= False,
                                            num_workers=args.num_workers, pin_memory=True,drop_last=False)

    # The formal matched study trains on the complete official validation/train
    # features.  It must not construct or iterate a THUMOS test loader.  The
    # untouched upstream path remains available outside the formal protocol.
    if matched_study:
        test_dataset = None
        test_loader = None
    elif d1_train_only:
        test_dataset = None
        test_loader = None
    else:
        test_dataset = THUMOS14Dataset(args, subset='test')
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    model = build_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model).to(device)

    criterion = build_criterion(args, device)
    
    max_mAP = 0
    
    # get the number of model parameters
    parameters = 'Number of full model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    print_log(save_path, '--------------------Number of parameters--------------------')
    print_log(save_path, parameters)

    # optimizer and scheduler        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.min_lr, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.lr_Tcycle, T_mult=1, eta_max=args.max_lr,
                                                T_up=args.lr_Tup, gamma=args.lr_gamma)
    
    if args.load_model:
        checkpoint = torch.load(args.model_path)
        validate_d1_checkpoint_compatibility(checkpoint, model, args)
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
        model.load_state_dict(pretrained_dict)
        criterion.load_state_dict(checkpoint['criterion_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 1
    d11_schedule_audit = prepare_d11_effective_dose(
        args,
        optimizer,
        scheduler,
        len(train_loader),
    )
    
    if args.wandb:
        wandb.watch(model)
    
    # training
    for epoch in range(start_epoch, args.epochs + 1):
        print_log(save_path, '----- %s at epoch #%d' % ('Train', epoch))
        train_log = train_one_epoch(
            args,
            train_dataset,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            device,
            schedule_audit=d11_schedule_audit,
        )
        if d1_train_only:
            metrics_filename = (
                'pilot_epoch_metrics.jsonl'
                if d1_preexperiment
                else 'mechanism_epoch_metrics.jsonl'
            )
            metrics_path = os.path.join(save_path, metrics_filename)
            with open(metrics_path, 'a', encoding='utf-8') as metrics_file:
                metrics_file.write(
                    json.dumps(
                        {
                            'epoch': epoch,
                            'metrics': train_log,
                            'study_protocol': study_protocol,
                            'event_d13_variant': getattr(
                                args, 'event_d13_variant', 'd12_control'
                            ),
                            'event_d14_variant': getattr(
                                args, 'event_d14_variant', 'none'
                            ),
                            'strict_causal_paper_result_valid': False,
                            'test_access': False,
                        },
                        sort_keys=True,
                    )
                    + '\n'
                )
        
        if epoch % args.train_eval_step == 0:
            print_log(save_path, 'mAP: %.2f' % (train_log['mAP_train']))
            print_log(save_path, 'mAP@.3: %.2f' % (train_log['mAP_03_train']))
            print_log(save_path, 'mAP@0.4: %.2f' % (train_log['mAP_04_train']))
            print_log(save_path, 'mAP@.5: %.2f' % (train_log['mAP_05_train']))
            print_log(save_path, 'mAP@.6: %.2f' % (train_log['mAP_06_train']))
            print_log(save_path, 'mAP@.7: %.2f' % (train_log['mAP_07_train']))
        scheduler.step()
        
        if args.wandb:
            wandb.log(train_log)
        if matched_study:
            if epoch == args.epochs:
                result_path = os.path.join(
                    save_path, 'terminal_epoch{}.pth'.format(args.epochs)
                )
                torch.save(
                    training_state(model, criterion, optimizer, scheduler, epoch, args),
                    result_path,
                )
            continue
        if d1_train_only:
            if epoch == args.epochs:
                result_path = os.path.join(
                    save_path, 'terminal_epoch{}.pth'.format(args.epochs)
                )
                torch.save(
                    training_state(model, criterion, optimizer, scheduler, epoch, args),
                    result_path,
                )
            continue

        if epoch % args.test_freq == 0:
            print_log(save_path, '----- %s at epoch #%d' % ('Test', epoch))
            test_log = test_one_epoch(args, test_dataset, test_loader, model, criterion, optimizer, epoch, device)
               
            if epoch % args.test_eval_step == 0:
                print_log(save_path, 'mAP: %.2f' % (test_log['mAP_test']))
                print_log(save_path, 'mAP@.3: %.2f' % (test_log['mAP_03_test']))
                print_log(save_path, 'mAP@0.4: %.2f' % (test_log['mAP_04_test']))
                print_log(save_path, 'mAP@.5: %.2f' % (test_log['mAP_05_test']))
                print_log(save_path, 'mAP@.6: %.2f' % (test_log['mAP_06_test']))
                print_log(save_path, 'mAP@.7: %.2f' % (test_log['mAP_07_test']))
            if args.wandb:
                wandb.log(test_log)

        if max_mAP < test_log['mAP_test']:
            state = training_state(model, criterion, optimizer, scheduler, epoch, args)
            result_path = save_path + '/best_epoch%d.pth' % epoch
            
            # remove previous epoch model
            pth_files = [file for file in os.listdir(save_path) if '.pth' in file]
            
            for f in pth_files:
                f = os.path.join(save_path, f)
                os.remove(f)                        
            torch.save(state, result_path)
            max_mAP = test_log['mAP_test']
            
@torch.no_grad()
def eval(args):
    args.training = False
    save_path = args.save_path
    test_dataset = THUMOS14Dataset(args, subset='test')
                
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=args.batch, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True,drop_last=False) 
        
    model = build_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model).to(device)
    criterion = build_criterion(args, device)
    
    checkpoint = torch.load(args.model_path)
    validate_d1_checkpoint_compatibility(checkpoint, model, args)

    pretrained_dict = checkpoint['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
    model.load_state_dict(pretrained_dict) 

    epoch = checkpoint['epoch']

    model.eval()
    criterion.eval()
    
    memory_initialize(model, args)

    metric_logger = loggers.MetricLogger(mode="evaluation", delimiter="  ")
    header = 'Evaluation Inference: '
    
    print_freq = len(test_loader)
    
    # output path
    proposal_file = args.proposal_path.format({}, 'eval', str(epoch))
    proposal_txt_path = os.path.join(save_path, (proposal_file+'.txt'))
    Path(proposal_txt_path.format("pred")).write_text("", encoding="utf-8")
    att_cnt = 0
    for i, (inputs, targets, infos) in enumerate(metric_logger.log_every(test_loader, print_freq, header)):
        inputs, targets, infos = parrallel_collate_fn(inputs, targets, infos, args.p_videos)
        inputs = inputs.to(device) # batch x seq lens x feature size
        bs, seq_len, _ = inputs.shape
        targets = {k: v.to(device) for k, v in targets.items()}
        inputs = make_model_inputs(args, inputs, infos)
        
        # compute output
        outputs = model(inputs, device)
        _loss_dict = criterion(outputs, targets, infos, device)
        loss_dict = {k: v for k,v in _loss_dict.items()}
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        
        if args.make_output:
            write_model_predictions(
                args,
                model,
                infos,
                outputs,
                proposal_txt_path,
                test_dataset.label_name,
            )
    
    proposal_json_path = os.path.join(args.save_path, (proposal_file+'.json')).format('pred')
    proposal_pred_txt_path = proposal_txt_path.format('pred')
    
    result_dict = online_nms(args, proposal_pred_txt_path, test_dataset)
    output_dict={"version":"VERSION 1", "results": result_dict, "external_data": {}}
    
    outfile=open(proposal_json_path, "w")
    json.dump(output_dict,outfile, indent=2)
    outfile.close()
    
    tiou_thresholds=np.linspace(0.3,0.70,5)
    
    mAP = evaluation_detection(args, proposal_json_path, subset='test', 
                                tiou_thresholds=tiou_thresholds, verbose=True)
    
    metric_logger.update(mAP=mAP.mean())
    metric_logger.update(mAP_03=mAP[0])
    metric_logger.update(mAP_04=mAP[1])
    metric_logger.update(mAP_05=mAP[2])
    metric_logger.update(mAP_06=mAP[3])
    metric_logger.update(mAP_07=mAP[4])
    
    print("Averaged stats:", metric_logger)
    
    eval_log = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    print_log(save_path, 'mAP: %.2f' % (eval_log['mAP']))
    print_log(save_path, 'mAP@.3: %.2f' % (eval_log['mAP_03']))
    print_log(save_path, 'mAP@0.4: %.2f' % (eval_log['mAP_04']))
    print_log(save_path, 'mAP@.5: %.2f' % (eval_log['mAP_05']))
    print_log(save_path, 'mAP@.6: %.2f' % (eval_log['mAP_06']))
    print_log(save_path, 'mAP@.7: %.2f' % (eval_log['mAP_07']))

def train_one_epoch(
    args,
    train_dataset,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    device,
    schedule_audit=None,
):
    model.train()
    criterion.train()
    
    memory_initialize(model, args)
    
    args.training = True
    # logger
    metric_logger = loggers.MetricLogger(mode="train", delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=2, fmt='{value:.6f}'))
    # 최대 epoch 크기에 맞게 space padding
    space_fmt =str(len(str(args.epochs)))
    header = 'Epoch [{start_epoch: >{fill}}/{end_epoch}]'.format(start_epoch=epoch, end_epoch=args.epochs,
                                                                 fill=space_fmt)
    print_freq = len(train_loader)
    
    # output path
    proposal_file = args.proposal_path.format({},'train', str(epoch))
    proposal_txt_path = os.path.join(args.save_path, (proposal_file+'.txt'))
    Path(proposal_txt_path.format("pred")).write_text("", encoding="utf-8")
    optimizer_step_count = 0
    learning_rates = []
    mechanism_census_names = (
        'event_birth_positive_count',
        'event_birth_selected_negative_count',
        'event_birth_negative_candidate_count',
        'event_birth_positive_batch_count',
        'event_birth_zero_positive_batch_count',
        'event_birth_interval_fallback_count',
        'event_birth_prebirth_exposure_count',
        'event_birth_interval_exposure_count',
        'event_birth_selected_risk_logit_count',
        'event_birth_postinterval_ignored_exposure_count',
        'event_birth_prebirth_group_count',
        'event_birth_zero_prebirth_group_count',
        'event_birth_normalized_survival_event_count',
        'event_birth_decision_aligned_positive_bag_count',
        'event_birth_decision_aligned_negative_bag_count',
    )
    mechanism_epoch_census = {name: 0 for name in mechanism_census_names}

    for i, (inputs, targets, infos) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        inputs, targets, infos = parrallel_collate_fn(inputs, targets, infos, args.p_videos)
        inputs = inputs.to(device) # batch x seq lens x feature size
        bs, seq_len, _ = inputs.shape
        targets = {k: v.to(device) for k, v in targets.items()}
            
        inputs = make_model_inputs(args, inputs, infos, targets)

        outputs = model(inputs, device)
        _loss_dict = criterion(outputs, targets, infos, device)
        loss_weight = criterion.weight_dict
        
        loss = sum(_loss_dict[k] * loss_weight[k] for k in _loss_dict.keys() if k in loss_weight)

        loss_dict = {k: v for k,v in _loss_dict.items()}
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        if getattr(args, 'study_protocol', None) in {
            'd13_mechanism',
            'd14_mechanism',
        }:
            stage_name = (
                'D1.4'
                if getattr(args, 'study_protocol', None) == 'd14_mechanism'
                else 'D1.3'
            )
            for name in mechanism_census_names:
                if name not in loss_dict_reduced:
                    raise RuntimeError(
                        f'{stage_name} mechanism census metric is missing: {name}'
                    )
                value = float(loss_dict_reduced[name].detach().cpu().item())
                if not math.isfinite(value) or not value.is_integer():
                    raise RuntimeError(
                        f'{stage_name} mechanism census metric is not an integer: '
                        f'{name}={value}'
                    )
                mechanism_epoch_census[name] += int(value)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * loss_weight[k] for k, v in loss_dict_reduced.items() if k in loss_weight}
        losses_reduced_scaled =sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        
        if args.wandb:
            wandb.log(loss_dict_reduced_scaled)
            
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        # compute gradient and optimization step
        optimizer.zero_grad()
        loss.backward()
        gradient_metrics = d1_gradient_metrics(model, args)
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        optimizer_step_count += 1

        metric_logger.update(
            loss=loss_value,
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
            **gradient_metrics,
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if args.make_output:
            write_model_predictions(
                args,
                model,
                infos,
                outputs,
                proposal_txt_path,
                train_dataset.label_name,
            )
    
    if epoch % args.train_eval_step == 0:
        proposal_json_path = os.path.join(args.save_path, (proposal_file+'.json')).format('pred')
        proposal_pred_txt_path = proposal_txt_path.format('pred')

        result_dict = online_nms(args, proposal_pred_txt_path, train_dataset)
        output_dict={"version":"VERSION 1", "results": result_dict, "external_data": {}}
        
        outfile=open(proposal_json_path, "w")
        json.dump(output_dict,outfile, indent=2)
        outfile.close()
        
        tiou_thresholds=np.linspace(0.3,0.70,5)
        mAP = evaluation_detection(args, proposal_json_path, subset='train', 
                                    tiou_thresholds=tiou_thresholds, verbose=True)        
    else:        
        mAP = np.array([0,0,0,0,0], dtype=np.float)

    metric_logger.synchronize_between_processes()  

    metric_logger.update(mAP_train=mAP.mean())
    metric_logger.update(mAP_03_train=mAP[0])
    metric_logger.update(mAP_04_train=mAP[1])
    metric_logger.update(mAP_05_train=mAP[2])
    metric_logger.update(mAP_06_train=mAP[3])
    metric_logger.update(mAP_07_train=mAP[4])
    
    print("Averaged stats:", metric_logger)
    
    result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if getattr(args, 'study_protocol', None) in {
        'd13_mechanism',
        'd14_mechanism',
    }:
        census_prefix = (
            'd14'
            if getattr(args, 'study_protocol', None) == 'd14_mechanism'
            else 'd13'
        )
        result.update(
            {
                '{}_epoch_physical_batch_count'.format(
                    census_prefix
                ): float(len(train_loader)),
                **{
                    '{}_epoch_{}_total'.format(
                        census_prefix, name[len('event_'):]
                    ): float(
                        value
                    )
                    for name, value in mechanism_epoch_census.items()
                },
            }
        )
    if schedule_audit is not None:
        expected_steps = int(schedule_audit['expected_optimizer_steps'])
        expected_lr = float(schedule_audit['training_learning_rate'])
        if optimizer_step_count != expected_steps:
            raise RuntimeError(
                'D1.2 effective-dose optimizer steps did not close: '
                f'{optimizer_step_count} != {expected_steps}'
            )
        if (
            not learning_rates
            or any(
                not math.isclose(
                    learning_rate,
                    expected_lr,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                for learning_rate in learning_rates
            )
        ):
            raise RuntimeError('D1.2 effective-dose learning rate drifted within epoch')
        result.update(
            {
                'd11_effective_dose_enabled': 1.0,
                'd11_optimizer_step_count': float(optimizer_step_count),
                'd11_initial_learning_rate': float(
                    schedule_audit['initial_learning_rate']
                ),
                'd11_learning_rate_first': learning_rates[0],
                'd11_learning_rate_minimum': min(learning_rates),
                'd11_learning_rate_maximum': max(learning_rates),
                'd11_learning_rate_last': learning_rates[-1],
                'd11_scheduler_last_epoch_at_train_start': float(
                    schedule_audit['scheduler_last_epoch_at_train_start']
                ),
                'd11_scheduler_t_cur_at_train_start': float(
                    schedule_audit['scheduler_t_cur_at_train_start']
                ),
            }
        )
        trace_path = Path(args.save_path) / 'effective_dose_update_trace.jsonl'
        with trace_path.open('w', encoding='utf-8') as trace_file:
            for optimizer_step, learning_rate in enumerate(learning_rates, start=1):
                trace_file.write(
                    json.dumps(
                        {
                            'optimizer_step': optimizer_step,
                            'learning_rate': learning_rate,
                        },
                        sort_keys=True,
                    )
                    + '\n'
                )
    return result


@torch.no_grad()
def test_one_epoch(args, test_dataset, test_loader, model, criterion, optimizer, epoch, device):
    model.eval()
    criterion.eval()
    
    memory_initialize(model, args)
    
    args.training = False
    metric_logger = loggers.MetricLogger(mode="test", delimiter="   ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=2, fmt='{value:.6f}'))
    
    space_fmt =str(len(str(args.epochs)))
    header = 'Epoch [{start_epoch: >{fill}}/{end_epoch}]'.format(start_epoch=epoch, end_epoch=args.epochs,
                                                                 fill=space_fmt)
    print_freq = len(test_loader)
    
    # output path
    proposal_file = args.proposal_path.format({},'test', str(epoch))
    proposal_txt_path = os.path.join(args.save_path, (proposal_file+'.txt'))
    Path(proposal_txt_path.format("pred")).write_text("", encoding="utf-8")
    for i, (inputs, targets, infos) in enumerate(metric_logger.log_every(test_loader, print_freq, header)):
        
        inputs, targets, infos = parrallel_collate_fn(inputs, targets, infos, args.p_videos)
        inputs = inputs.to(device) # batch x seq lens x feature size
        bs, seq_len, _ = inputs.shape
        targets = {k: v.to(device) for k, v in targets.items()}
        inputs = make_model_inputs(args, inputs, infos)
        
        # compute output
        outputs = model(inputs, device)
        _loss_dict = criterion(outputs, targets, infos, device)
        loss_weight = criterion.weight_dict
        
        loss = sum(_loss_dict[k] * loss_weight[k] for k in _loss_dict.keys() if k in loss_weight)
        
        loss_dict = {k: v for k,v in _loss_dict.items()}
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * loss_weight[k] for k, v in loss_dict_reduced.items() if k in loss_weight}
        losses_reduced_scaled =sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        
        if args.wandb:
            wandb.log(loss_dict_reduced_scaled)
            
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
            
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if args.make_output:
            write_model_predictions(
                args,
                model,
                infos,
                outputs,
                proposal_txt_path,
                test_dataset.label_name,
            )
    
    if epoch % args.test_eval_step == 0:
        proposal_json_path = os.path.join(args.save_path, (proposal_file+'.json')).format('pred')
        proposal_pred_txt_path = proposal_txt_path.format('pred')
        
        result_dict = online_nms(args, proposal_pred_txt_path, test_dataset)
        output_dict={"version":"VERSION 1", "results": result_dict, "external_data": {}}

        outfile=open(proposal_json_path, "w")
        json.dump(output_dict,outfile, indent=2)
        outfile.close()
        
        tiou_thresholds=np.linspace(0.3,0.70,5)
        mAP = evaluation_detection(args, proposal_json_path, subset='test', 
                                    tiou_thresholds=tiou_thresholds, verbose=True)
        # import pdb;pdb.set_trace()
        ###`
    else:
        mAP = np.array([0,0,0,0,0], dtype=np.float)
    
    metric_logger.synchronize_between_processes()  
    
    metric_logger.update(mAP_test=mAP.mean())
    metric_logger.update(mAP_03_test=mAP[0])
    metric_logger.update(mAP_04_test=mAP[1])
    metric_logger.update(mAP_05_test=mAP[2])
    metric_logger.update(mAP_06_test=mAP[3])
    metric_logger.update(mAP_07_test=mAP[4])
    
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
