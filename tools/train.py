import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import json
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import train_one_epoch, val_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from opentad.utils import (
    configure_strict_determinism,
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument(
        "--allow-unready-smoke",
        action="store_true",
        help="allow only an explicitly smoke_only config while formal training is locked",
    )
    parser.add_argument(
        "--allow-unready-screen",
        action="store_true",
        help=(
            "allow only an explicitly registered screening_only config while "
            "formal training remains locked"
        ),
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


def _streaming_shuffle(loader_cfg):
    return not bool(loader_cfg.get("streaming", False))


def _set_dataloader_epoch(loader, epoch):
    for sampler in (getattr(loader, "batch_sampler", None), getattr(loader, "sampler", None)):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
            return


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    formal_training_ready = bool(cfg.get("formal_training_ready", True))
    smoke_only = bool(cfg.get("smoke_only", False))
    screening_only = bool(cfg.get("screening_only", False))
    screening_training_ready = bool(
        cfg.get("screening_training_ready", False)
    )
    if args.allow_unready_smoke and args.allow_unready_screen:
        raise RuntimeError(
            "smoke and screen readiness overrides are mutually exclusive"
        )
    if args.allow_unready_smoke and not smoke_only:
        raise RuntimeError(
            "--allow-unready-smoke requires an explicit smoke_only config"
        )
    if args.allow_unready_screen and not (
        screening_only and screening_training_ready
    ):
        raise RuntimeError(
            "--allow-unready-screen requires an explicitly registered "
            "screening_only config"
        )
    smoke_authorized = args.allow_unready_smoke and smoke_only
    screen_authorized = (
        args.allow_unready_screen
        and screening_only
        and screening_training_ready
    )
    if not formal_training_ready and not (
        smoke_authorized or screen_authorized
    ):
        raise RuntimeError(
            "formal_training_ready is false; only an explicit smoke_only or "
            "registered screening_only config may use its matching override"
        )
    fit_only = bool(cfg.workflow.get("fit_only", False))
    if screen_authorized:
        contract = cfg.get("screening_contract", {})
        expected_seed = int(contract.get("seed", -1))
        expected_epochs = int(contract.get("epochs", -1))
        if args.seed != expected_seed:
            raise RuntimeError(
                f"screen seed must remain frozen at {expected_seed}"
            )
        if int(cfg.workflow.get("end_epoch", -1)) != expected_epochs:
            raise RuntimeError(
                "screen workflow does not match its registered epoch count"
            )
        if expected_epochs != 1:
            raise RuntimeError("the registered technical screen is one epoch")
        if not fit_only:
            raise RuntimeError("screen training must remain fit-only")
        if bool(cfg.get("raw_video_finetuning", False)):
            raise RuntimeError("the registered screen is feature-only")
        if args.resume is not None:
            raise RuntimeError("the registered screen must start from seed initialization")
    persistent_route = cfg.get("route_stage", "").startswith(
        "persistent_binding"
    )
    if persistent_route and args.disable_deterministic:
        raise RuntimeError(
            "persistent-binding execution forbids --disable_deterministic"
        )
    if fit_only and cfg.workflow.get("val_eval_interval", -1) > 0:
        raise RuntimeError(
            "fit-only training forbids reporting-set evaluation"
        )

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    if persistent_route and args.world_size != 1:
        raise RuntimeError(
            "persistent-binding chronological training requires world_size=1"
        )

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    determinism = (
        configure_strict_determinism()
        if persistent_route
        else None
    )
    cfg = update_workdir(cfg, args.id, args.world_size)
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    if determinism is not None:
        logger.info(f"Strict determinism: {determinism}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    train_loader = build_dataloader(
        train_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=_streaming_shuffle(cfg.solver.train),
        drop_last=True,
        **cfg.solver.train,
    )

    val_loader = None
    if cfg.workflow.get("val_loss_interval", -1) > 0:
        val_dataset = build_dataset(
            cfg.dataset.val,
            default_args=dict(logger=logger),
        )
        val_loader = build_dataloader(
            val_dataset,
            rank=args.rank,
            world_size=args.world_size,
            shuffle=False,
            drop_last=False,
            **cfg.solver.val,
        )

    test_loader = None
    if not fit_only and cfg.workflow.get("val_eval_interval", -1) > 0:
        test_dataset = build_dataset(
            cfg.dataset.test,
            default_args=dict(logger=logger),
        )
        test_loader = build_dataloader(
            test_dataset,
            rank=args.rank,
            world_size=args.world_size,
            shuffle=False,
            drop_last=False,
            **cfg.solver.test,
        )

    # build model
    model = build_detector(cfg.model)

    # DDP
    use_static_graph = getattr(cfg.solver, "static_graph", False)
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False if use_static_graph else True,
        static_graph=use_static_graph,  # default is False, should be true when use activation checkpointing in E2E
    )
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    # FP16 compression
    use_fp16_compress = getattr(cfg.solver, "fp16_compress", False)
    if use_fp16_compress:
        logger.info("Using FP16 compression ...")
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        logger.info("Using Model EMA...")
        model_ema = ModelEma(model)
    else:
        model_ema = None

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        scaler = GradScaler()
    else:
        scaler = None

    # build optimizer and scheduler
    optimizer = build_optimizer(cfg.optimizer, model, logger)
    scheduler, max_epoch = build_scheduler(cfg.scheduler, optimizer, len(train_loader))

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    # resume: reset epoch, load checkpoint / best rmse
    if args.resume != None:
        logger.info("Resume training from: {}".format(args.resume))
        device = f"cuda:{args.local_rank}"
        checkpoint = torch.load(args.resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if model_ema != None:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

        del checkpoint  #  save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    # train the detector
    logger.info("Training Starts...\n")
    val_loss_best = 1e6
    val_start_epoch = cfg.workflow.get("val_start_epoch", 0)
    training_audit_rows = []
    for epoch in range(resume_epoch + 1, max_epoch):
        _set_dataloader_epoch(train_loader, epoch)

        # train for one epoch
        epoch_audit = train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            logger,
            model_ema=model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=cfg.workflow.logging_interval,
            runtime_debug_interval=cfg.workflow.get("runtime_debug_interval", -1),
            scaler=scaler,
            fail_on_nonfinite=cfg.workflow.get("fail_on_nonfinite", False),
        )
        epoch_audit["epoch"] = int(epoch)
        training_audit_rows.append(epoch_audit)

        # save checkpoint
        save_checkpoint_enabled = not cfg.workflow.get("disable_checkpoint", False)
        if save_checkpoint_enabled and ((epoch == max_epoch - 1) or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0)):
            if args.rank == 0:
                save_checkpoint(model, model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)

        # val for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_loss_interval > 0) and ((epoch + 1) % cfg.workflow.val_loss_interval == 0):
                val_loss = val_one_epoch(
                    val_loader,
                    model,
                    logger,
                    args.rank,
                    epoch,
                    model_ema=model_ema,
                    use_amp=use_amp,
                )

                # save the best checkpoint
                if val_loss < val_loss_best:
                    logger.info(f"New best epoch {epoch}")
                    val_loss_best = val_loss
                    if args.rank == 0:
                        save_best_checkpoint(model, model_ema, epoch, work_dir=cfg.work_dir)

        # eval for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_eval_interval > 0) and ((epoch + 1) % cfg.workflow.val_eval_interval == 0):
                eval_one_epoch(
                    test_loader,
                    model,
                    cfg,
                    logger,
                    args.rank,
                    model_ema=model_ema,
                    use_amp=use_amp,
                    world_size=args.world_size,
                    not_eval=args.not_eval,
                )
    if args.rank == 0:
        audit_fields = (
            "expected_updates",
            "successful_updates",
            "scheduler_steps",
            "skipped_updates",
            "gt_supervision_exhaustions",
            "gt_birth_runtime_entry_free_collisions",
            "candidate_arbitration_suppressions",
            "candidate_cancellations",
            "active_abandonments",
            "deferred_birth_due_to_release",
        )
        totals = {
            field: sum(int(row[field]) for row in training_audit_rows)
            for field in audit_fields
        }
        audit_path = os.path.join(cfg.work_dir, "training_audit.json")
        with open(audit_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "schema_version": "persistent_binding_training_audit.v1",
                    "seed": int(args.seed),
                    "fit_only": fit_only,
                    "route_stage": str(cfg.get("route_stage", "")),
                    "binding_mode": str(
                        cfg.model.get("trajectory_binding_mode", "")
                    ),
                    "prior_bias_mode": str(
                        cfg.model.get("prior_bias_mode", "")
                    ),
                    "screening_only": bool(
                        cfg.get("screening_only", False)
                    ),
                    "epochs": training_audit_rows,
                    "totals": totals,
                },
                file,
                indent=2,
                sort_keys=True,
            )
    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
