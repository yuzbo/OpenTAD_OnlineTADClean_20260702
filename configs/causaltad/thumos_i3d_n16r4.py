_base_ = ["./thumos_i3d.py"]

annotation_path = "/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json"
class_map = "/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt"
data_path = "/data/run01/sczc063/yuzibo/thumos14/features/i3d_actionformer_stride4_thumos/"
block_list = data_path + "missing_files.txt"

dataset = dict(
    train=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=data_path,
        block_list=block_list,
    ),
    val=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=data_path,
        block_list=block_list,
    ),
    test=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=data_path,
        block_list=block_list,
    ),
)

evaluation = dict(ground_truth_filename=annotation_path)

work_dir = "exps/thumos/causal_i3d_n16r4"
