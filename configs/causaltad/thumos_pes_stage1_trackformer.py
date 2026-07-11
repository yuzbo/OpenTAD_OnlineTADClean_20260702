_base_ = "./thumos_pes_stage1_base.py"

route_variant = "temporal_trackformer_reconstruction"

model = dict(
    assignment_mode="prefix",
    head=dict(
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
    ),
)

work_dir = "exps/thumos/pes_stage1_trackformer"
