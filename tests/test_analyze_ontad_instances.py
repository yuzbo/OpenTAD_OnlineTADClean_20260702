import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "analyze_ontad_instances.py"
SPEC = importlib.util.spec_from_file_location("analyze_ontad_instances", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_concurrency_keeps_same_class_instances_separate_and_ignores_touching_boundaries():
    data = {
        "database": {
            "v1": {
                "subset": "validation",
                "annotations": [
                    {"segment": [0.0, 4.0], "label": "A"},
                    {"segment": [1.0, 3.0], "label": "A"},
                    {"segment": [3.0, 5.0], "label": "B"},
                ],
            },
            "v2": {
                "subset": "test",
                "annotations": [{"segment": [0.0, 1.0], "label": "C"}],
            },
        }
    }

    report = MODULE.analyze_database(data, subsets=["validation"])

    assert report["num_videos"] == 1
    assert report["num_instances"] == 3
    assert report["concurrency"]["global_max"] == 2
    assert report["same_class_concurrency"]["global_max"] == 2
    assert report["suggested_num_slots"] == 3


def test_invalid_segments_are_audited_instead_of_counted():
    data = {
        "database": {
            "v1": {
                "subset": "validation",
                "annotations": [
                    {"segment": [2.0, 2.0], "label": "A"},
                    {"segment": [1.0, 3.0], "label": "B"},
                ],
            }
        }
    }

    report = MODULE.analyze_database(data)

    assert report["num_instances"] == 1
    assert report["invalid_instances"] == [
        {"video_id": "v1", "index": 0, "segment": (2.0, 2.0)}
    ]
