_base_ = ["./thumos_internvideo2.py"]

annotation_path = "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json"
class_map = "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt"
data_path = "/data/run01/sczc063/yuzibo/thumos14/features/thumos14_6b/"

dataset = dict(
    train=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=data_path,
        block_list=None,
    ),
    val=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=data_path,
        block_list=None,
    ),
    test=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=data_path,
        block_list=None,
    ),
)

evaluation = dict(ground_truth_filename=annotation_path)

work_dir = "exps/thumos/causal_internvideo2_6b_n16r4"
