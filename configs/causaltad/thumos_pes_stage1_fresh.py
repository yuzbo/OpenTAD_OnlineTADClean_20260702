_base_ = "./thumos_pes_stage1_base.py"

route_variant = "fresh_query_rediscovery"

model = dict(
    assignment_mode="per_step",
    head=dict(
        query_mode="fresh",
        start_mode="scalar",
        endpoint_mode="binary",
    ),
)

work_dir = "exps/thumos/pes_stage1_fresh"
