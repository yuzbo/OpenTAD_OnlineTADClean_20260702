import os
import copy
import json
import tqdm
import torch
import torch.distributed as dist

from opentad.utils import create_folder
from opentad.utils.device import get_model_device, move_data_to_device
from opentad.utils.online_protocol import (
    is_streaming_safe_emission,
    resolve_sliding_window_for_post_processing,
    should_run_video_level_nms,
    sort_emission_ledger,
    summarize_emission_ledger,
    validate_emission_ledger_summary,
    validate_streaming_safe_ext_cls,
    validate_streaming_safe_world_size,
)
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.evaluations import build_evaluator
from opentad.datasets.base import SlidingWindowDataset


def eval_one_epoch(
    test_loader,
    model,
    cfg,
    logger,
    rank,
    model_ema=None,
    use_amp=False,
    amp_dtype=None,
    world_size=0,
    not_eval=False,
):
    """Inference and Evaluation the model"""

    if amp_dtype is None and use_amp:
        amp_dtype = torch.float16
    use_amp = amp_dtype is not None

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    cfg.inference["folder"] = os.path.join(cfg.work_dir, "outputs")
    if cfg.inference.save_raw_prediction:
        create_folder(cfg.inference["folder"])

    # external classifier
    if "external_cls" in cfg.post_processing:
        if cfg.post_processing.external_cls != None:
            external_cls = build_classifier(cfg.post_processing.external_cls)
    else:
        external_cls = test_loader.dataset.class_map
    validate_streaming_safe_ext_cls(external_cls, cfg.post_processing)

    # whether video-level sliding-window merging is allowed. Streaming-safe
    # emission keeps each prefix/window as an auditable online output.
    cfg.post_processing.sliding_window = resolve_sliding_window_for_post_processing(
        cfg.post_processing,
        isinstance(test_loader.dataset, SlidingWindowDataset),
    )
    validate_streaming_safe_world_size(cfg.post_processing, world_size)

    # model forward
    model.eval()
    target_model = model.module if hasattr(model, "module") else model
    if hasattr(target_model, "reset_online_states"):
        target_model.reset_online_states()
    model_device = get_model_device(model)

    result_dict = {}
    for data_dict in tqdm.tqdm(test_loader, disable=(rank != 0)):
        data_dict = move_data_to_device(data_dict, model_device)
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=cfg.post_processing,
                    ext_cls=external_cls,
                )

        # update the result dict
        for k, v in results.items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    result_dict = gather_ddp_results(world_size, result_dict, cfg.post_processing)
    emission_summary = None
    if is_streaming_safe_emission(cfg.post_processing):
        emission_summary = summarize_emission_ledger(result_dict)
        validate_emission_ledger_summary(emission_summary)

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)

    if rank == 0:
        result_eval = dict(results=result_dict)
        if emission_summary is not None:
            latency = emission_summary["latency_sec"]
            logger.info(
                "[OnlineEval]: emissions=%d videos=%d streams=%d latency_mean=%s latency_p95=%s "
                "latency_max=%s no_future=%s",
                emission_summary["num_emissions"],
                emission_summary["num_videos"],
                emission_summary["num_streams"],
                latency["mean"],
                latency["p95"],
                latency["max"],
                emission_summary["no_future"],
            )
            if getattr(cfg.post_processing, "save_emission_ledger", True):
                ledger_path = os.path.join(
                    cfg.work_dir,
                    getattr(cfg.post_processing, "emission_ledger_filename", "emission_ledger.json"),
                )
                with open(ledger_path, "w") as out:
                    json.dump(dict(results=result_dict, summary=emission_summary), out, indent=2)
            if getattr(cfg.post_processing, "save_latency_summary", True):
                summary_path = os.path.join(
                    cfg.work_dir,
                    getattr(cfg.post_processing, "latency_summary_filename", "emission_latency_summary.json"),
                )
                with open(summary_path, "w") as out:
                    json.dump(emission_summary, out, indent=2)
        if cfg.post_processing.save_dict:
            result_path = os.path.join(cfg.work_dir, "result_detection.json")
            with open(result_path, "w") as out:
                json.dump(result_eval, out)

        if not not_eval:
            # build evaluator
            evaluator = build_evaluator(dict(prediction_filename=result_eval, **cfg.evaluation))
            # evaluate and output
            logger.info("Evaluation starts...")
            metrics_dict = evaluator.evaluate()
            evaluator.logging(logger)


def gather_ddp_results(world_size, result_dict, post_cfg):
    gather_dict_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_dict_list, result_dict)
    result_dict = {}
    for i in range(world_size):  # update the result dict
        for k, v in gather_dict_list[i].items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    if is_streaming_safe_emission(post_cfg):
        return sort_emission_ledger(result_dict)

    # do nms for sliding window, if needed
    if should_run_video_level_nms(post_cfg, getattr(post_cfg, "sliding_window", False)):
        # assert sliding_window=True
        tmp_result_dict = {}
        for k, v in result_dict.items():
            segments = torch.Tensor([data["segment"] for data in v])
            scores = torch.Tensor([data["score"] for data in v])
            labels = []
            class_idx = []
            for data in v:
                if data["label"] not in class_idx:
                    class_idx.append(data["label"])
                labels.append(class_idx.index(data["label"]))
            labels = torch.Tensor(labels)

            segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=class_idx[int(label.item())],
                        score=round(score.item(), 4),
                    )
                )
            tmp_result_dict[k] = results_per_video
        result_dict = tmp_result_dict
    return result_dict
