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
from opentad.utils.full_petal_launch import (
    PROFILE_MODE,
    build_fixed_step_profile_artifact,
    validate_full_petal_launch,
)


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

    launch_authorization = validate_full_petal_launch(
        cfg,
        args.config,
        mode=args.launch_mode,
        ticket_path=args.launch_ticket,
        entrypoint="train",
        cfg_override_keys=tuple((args.cfg_options or {}).keys()),
        environ=os.environ,
        repository_root=Path(__file__).resolve().parents[1],
    )
    if launch_authorization is not None and args.launch_mode == PROFILE_MODE and args.resume:
        raise RuntimeError("fixed-step profile cannot resume from a checkpoint")

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
    for epoch in range(resume_epoch + 1, max_epoch):
        _set_dataloader_epoch(train_loader, epoch)

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
        )

        if fixed_step_profiler is not None and train_stats["fixed_step_profile_complete"]:
            if args.rank == 0:
                precision = (
                    "bf16"
                    if amp_dtype is torch.bfloat16
                    else "fp16"
                    if amp_dtype is torch.float16
                    else "fp32"
                )
                artifact = build_fixed_step_profile_artifact(
                    launch_authorization,
                    cfg,
                    fixed_step_profiler.measurements(),
                    precision=precision,
                    gpu_name=torch.cuda.get_device_name(args.local_rank),
                    torch_version=torch.__version__,
                    cuda_version=torch.version.cuda,
                )
                output = Path(cfg.work_dir) / "fixed_step_profile.json"
                output.write_text(
                    json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
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
    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
