import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import (
    build_optimizer,
    build_scheduler,
    optimizer_events_per_epoch,
    eval_one_epoch,
    resolve_amp_dtype,
    train_one_epoch,
    val_one_epoch,
)
from opentad.utils import (
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)
from opentad.utils.fixed_step_profile import FixedStepProfiler, TorchCudaProfileBackend
from opentad.utils.evidence_bundle import publish_exclusive_file
from opentad.utils.full_petal_launch import (
    FORMAL_MODE,
    PROFILE_MODE,
    build_fixed_step_profile_artifact,
    canonical_json_sha256,
    persist_launch_receipt,
    runtime_evidence_session,
    validate_full_petal_launch,
)
from opentad.utils.full_petal_training_evidence import (
    OptimizerEventTraceRecorder,
    VisualParameterEventRecorder,
)
from opentad.utils.full_petal_identity import derive_training_trace_identity


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    parser.add_argument("--launch-mode", choices=("profile", "formal"))
    parser.add_argument("--launch-ticket", type=str)
    args = parser.parse_args()
    return args


def _streaming_shuffle(loader_cfg):
    return not bool(loader_cfg.get("streaming", False))


def _set_dataloader_epoch(loader, epoch):
    sampler_updated_dataset = False
    for sampler in (getattr(loader, "batch_sampler", None), getattr(loader, "sampler", None)):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
            sampler_updated_dataset = getattr(sampler, "dataset", None) is loader.dataset
            break
    if hasattr(loader.dataset, "set_epoch") and not sampler_updated_dataset:
        loader.dataset.set_epoch(epoch)


def _precision_name(amp_dtype):
    if amp_dtype is torch.bfloat16:
        return "bf16"
    if amp_dtype is torch.float16:
        return "fp16"
    return "fp32"


def _data_order_identity(dataset, seed):
    manifests = getattr(dataset, "packet_manifests", None)
    if not isinstance(manifests, dict) or not manifests:
        raise RuntimeError("formal training requires an explicit packet manifest order")
    return canonical_json_sha256(
        {
            "seed": int(seed),
            "episodes": [
                {"video_id": str(video_id), "packet_indices": list(indices)}
                for video_id, indices in manifests.items()
            ],
        }
    )


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg_overrides = dict(args.cfg_options or {})
    execution_signing_key = os.environ.get("FULL_PETAL_EXECUTION_ATTESTATION_KEY")
    launch_authorization = validate_full_petal_launch(
        cfg,
        args.config,
        mode=args.launch_mode,
        ticket_path=args.launch_ticket,
        entrypoint="train",
        seed=args.seed,
        run_id=args.id,
        deterministic=not args.disable_deterministic,
        not_eval=args.not_eval,
        resume_path=args.resume,
        cfg_overrides=cfg_overrides,
        cfg_override_keys=tuple(cfg_overrides),
        environ=os.environ,
        repository_root=Path(__file__).resolve().parents[1],
        execution_signing_key_path=execution_signing_key,
    )
    if launch_authorization is not None and args.launch_mode == PROFILE_MODE and args.resume:
        raise RuntimeError("fixed-step profile cannot resume from a checkpoint")
    if launch_authorization is not None:
        receipt_path = persist_launch_receipt(
            launch_authorization,
            private_key_path=execution_signing_key,
        )
        print(f"FULL_PETAL_LAUNCH_RECEIPT={receipt_path}")

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    cfg = update_workdir(cfg, args.id, args.world_size)
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    if hasattr(train_dataset, "bind_sampling_seed"):
        train_dataset.bind_sampling_seed(args.seed)
    if hasattr(train_dataset, "bind_manifest_provenance"):
        if launch_authorization is None:
            raise RuntimeError("CRS-EPS training requires a validated launch authorization")
        train_dataset.bind_manifest_provenance(
            commit_sha=launch_authorization.commit_sha,
            scientific_config_sha256=launch_authorization.scientific_config_sha256,
            resolved_config_sha256=launch_authorization.resolved_config_sha256,
            launch_ticket_sha256=launch_authorization.ticket_sha256,
        )
    train_loader = build_dataloader(
        train_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=_streaming_shuffle(cfg.solver.train),
        drop_last=True,
        **cfg.solver.train,
    )

    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    val_loader = build_dataloader(
        val_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.val,
    )

    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
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
    amp_dtype = resolve_amp_dtype(
        use_amp,
        getattr(cfg.solver, "amp_dtype", "fp16"),
    )
    if use_amp:
        logger.info("Using Automatic Mixed Precision with dtype=%s...", amp_dtype)
        scaler = (
            GradScaler(enabled=amp_dtype is torch.float16)
            if amp_dtype is torch.float16
            else None
        )
    else:
        scaler = None

    fixed_step_profiler = None
    if launch_authorization is not None and launch_authorization.mode == PROFILE_MODE:
        fixed_step_profiler = FixedStepProfiler(
            launch_authorization.warmup_optimizer_events,
            launch_authorization.measured_optimizer_events,
            backend=TorchCudaProfileBackend(args.local_rank),
        )

    optimizer_event_recorder = None
    if launch_authorization is not None:
        execution_session = runtime_evidence_session(launch_authorization)
        expected_training_identity = derive_training_trace_identity(
            cfg,
            launch_authorization.data_identity,
            seed=args.seed,
            world_size=args.world_size,
        )
        runtime_data_order_sha256 = _data_order_identity(train_dataset, args.seed)
        if runtime_data_order_sha256 != expected_training_identity["data_order_sha256"]:
            raise RuntimeError(
                "runtime packet order differs from the data/config-derived training identity"
            )
        optimizer_event_recorder = OptimizerEventTraceRecorder(
            **expected_training_identity,
            runtime_session=execution_session,
            peak_memory_reader=torch.cuda.max_memory_allocated,
            synchronize=torch.cuda.synchronize,
        )

    # build optimizer and scheduler
    optimizer = build_optimizer(cfg.optimizer, model, logger)
    scheduler, max_epoch = build_scheduler(
        cfg.scheduler,
        optimizer,
        optimizer_events_per_epoch(train_loader),
    )
    visual_parameter_event_recorder = None
    visual_parameter_contract = getattr(cfg, "visual_parameter_contract", None)
    if (
        launch_authorization is not None
        and launch_authorization.mode == FORMAL_MODE
        and visual_parameter_contract is not None
    ):
        if set(visual_parameter_contract) != {
            "parameter_prefixes",
            "adapted_trainable_prefixes",
        }:
            raise RuntimeError("visual_parameter_contract fields differ")
        visual_parameter_event_recorder = VisualParameterEventRecorder(
            model,
            optimizer,
            parameter_prefixes=visual_parameter_contract["parameter_prefixes"],
            runtime_session=execution_session,
        )

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
    for epoch in range(resume_epoch + 1, max_epoch):
        _set_dataloader_epoch(train_loader, epoch)
        if hasattr(train_dataset, "persist_current_manifest"):
            if args.rank == 0:
                manifest_path = train_dataset.persist_current_manifest(
                    Path(cfg.work_dir) / "crs_eps_manifests"
                )
                logger.info("Published CRS-EPS epoch manifest: %s", manifest_path)
            dist.barrier()

        # train for one epoch
        train_stats = train_one_epoch(
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
            amp_dtype=amp_dtype,
            fixed_step_profiler=fixed_step_profiler,
            optimizer_event_recorder=optimizer_event_recorder,
            visual_parameter_event_recorder=visual_parameter_event_recorder,
        )

        if fixed_step_profiler is not None and train_stats["fixed_step_profile_complete"]:
            if args.rank == 0:
                profile_bundle_root = Path(launch_authorization.ticket_path).parent
                trace_path = profile_bundle_root / "fixed_step_optimizer_trace.jsonl"
                commitment_path = (
                    profile_bundle_root
                    / "fixed_step_optimizer_trace.commitment.json"
                )
                optimizer_event_recorder.persist(trace_path, commitment_path)
                precision = _precision_name(amp_dtype)
                profiler_measurements = fixed_step_profiler.measurements()
                if getattr(cfg, "crs_eps_contract", None) is not None:
                    profiler_measurements["workload"] = (
                        fixed_step_profiler.workload_measurements(
                            world_size=args.world_size
                        )
                    )
                artifact = build_fixed_step_profile_artifact(
                    launch_authorization,
                    cfg,
                    optimizer_event_trace_path=trace_path,
                    optimizer_event_commitment_path=commitment_path,
                    bundle_root=profile_bundle_root,
                    precision=precision,
                    gpu_name=torch.cuda.get_device_name(args.local_rank),
                    torch_version=torch.__version__,
                    cuda_version=torch.version.cuda,
                    profiler_measurements=profiler_measurements,
                )
                output = profile_bundle_root / "fixed_step_profile.json"
                publish_exclusive_file(
                    output,
                    (
                        json.dumps(
                            artifact,
                            allow_nan=False,
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode("utf-8"),
                )
                logger.info("Fixed-step profile PASS: %s", output)
            dist.barrier()
            logger.info("Profile completed without entering formal training")
            return

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
                    amp_dtype=amp_dtype,
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
                    amp_dtype=amp_dtype,
                    world_size=args.world_size,
                    not_eval=args.not_eval,
                )
    if fixed_step_profiler is not None:
        raise RuntimeError(
            "training schedule ended before the fixed-step profile reached its event budget"
        )
    if optimizer_event_recorder is not None:
        if args.rank == 0:
            trace_path = Path(cfg.work_dir) / "formal_training_trace.jsonl"
            commitment_path = (
                Path(cfg.work_dir) / "formal_training_trace.commitment.json"
            )
            optimizer_event_recorder.persist(trace_path, commitment_path)
            logger.info("Formal optimizer-event trace committed: %s", trace_path)
            if visual_parameter_event_recorder is not None:
                visual_trace_path = (
                    Path(cfg.work_dir) / "formal_visual_parameter_trace.jsonl"
                )
                visual_commitment_path = (
                    Path(cfg.work_dir)
                    / "formal_visual_parameter_trace.commitment.json"
                )
                visual_parameter_event_recorder.persist(
                    visual_trace_path, visual_commitment_path
                )
                logger.info(
                    "Formal visual parameter-event trace committed: %s",
                    visual_trace_path,
                )
        dist.barrier()
    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
