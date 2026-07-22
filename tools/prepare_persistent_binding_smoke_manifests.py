import argparse
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select small real videos for the persistent-binding Slurm smoke"
    )
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--fit-list", required=True)
    parser.add_argument("--calibration-list", required=True)
    parser.add_argument("--reporting-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-tokens", type=int, default=2)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_names(path):
    with open(path, "r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate video IDs in {path}")
    return names


def _select_video(names, database, cached_videos, subset, min_tokens):
    candidates = []
    for video_id in names:
        video = database.get(video_id)
        cached = cached_videos.get(video_id)
        if video is None or cached is None:
            continue
        annotations = tuple(video.get("annotations", ()))
        num_tokens = int(cached.get("num_tokens", 0))
        if video.get("subset") != subset or not annotations or num_tokens < min_tokens:
            continue
        candidates.append(
            {
                "video_id": video_id,
                "subset": subset,
                "num_tokens": num_tokens,
                "num_annotations": len(annotations),
            }
        )
    if not candidates:
        raise ValueError(
            f"no cached {subset} video with annotations and at least {min_tokens} tokens"
        )
    return min(candidates, key=lambda row: (row["num_tokens"], row["video_id"]))


def prepare_manifests(
    annotation,
    feature_manifest,
    fit_list,
    calibration_list,
    reporting_list,
    output_dir,
    min_tokens=2,
):
    source_paths = {
        "annotation": Path(annotation).resolve(),
        "feature_manifest": Path(feature_manifest).resolve(),
        "fit_list": Path(fit_list).resolve(),
        "calibration_list": Path(calibration_list).resolve(),
        "reporting_list": Path(reporting_list).resolve(),
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    with source_paths["annotation"].open("r", encoding="utf-8") as handle:
        database = json.load(handle)["database"]
    with source_paths["feature_manifest"].open("r", encoding="utf-8") as handle:
        cached_videos = json.load(handle)["videos"]

    selections = {
        "train": _select_video(
            _load_names(source_paths["fit_list"]),
            database,
            cached_videos,
            "training",
            min_tokens,
        ),
        "val": _select_video(
            _load_names(source_paths["calibration_list"]),
            database,
            cached_videos,
            "training",
            min_tokens,
        ),
        "test": _select_video(
            _load_names(source_paths["reporting_list"]),
            database,
            cached_videos,
            "validation",
            min_tokens,
        ),
    }
    selected_ids = [row["video_id"] for row in selections.values()]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("smoke train/val/test selections must be distinct")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_paths = {}
    for split, row in selections.items():
        path = output / f"{split}.txt"
        path.write_text(row["video_id"] + "\n", encoding="utf-8", newline="\n")
        manifest_paths[split] = str(path)

    payload = {
        "schema_version": 1,
        "selection_policy": (
            "fewest cached tokens among canonical split videos with at least one "
            "annotation; smoke-only and never used for scientific reporting"
        ),
        "min_tokens": int(min_tokens),
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "manifests": manifest_paths,
        "selections": selections,
    }
    selection_path = output / "selection.json"
    selection_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return payload


def main():
    args = parse_args()
    payload = prepare_manifests(
        annotation=args.annotation,
        feature_manifest=args.feature_manifest,
        fit_list=args.fit_list,
        calibration_list=args.calibration_list,
        reporting_list=args.reporting_list,
        output_dir=args.output_dir,
        min_tokens=args.min_tokens,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
