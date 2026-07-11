_base_ = "./thumos_pes_stage1_base.py"

route_variant = "prefix_observable_persistent_event_set"

model = dict(
    assignment_mode="prefix",
    head=dict(
        query_mode="persistent",
        start_mode="pointer",
        endpoint_mode="hazard",
    ),
)

work_dir = "exps/thumos/pes_stage1_persistent"
