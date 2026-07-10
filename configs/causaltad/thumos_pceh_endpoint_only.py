_base_ = "./thumos_pceh_ontad.py"

# Fair rolling endpoint-only control. It keeps the exact PCEH input, visual
# compute, cadence, cache, optimizer, and evaluator, but emits immediately when
# the endpoint hazard crosses threshold and disables completion/emission losses.
route_stage = "controlled_endpoint_only"
formal_training_ready = False

model = dict(
    head=dict(
        emission_policy="endpoint_only",
        completion_loss_weight=0.0,
        emission_loss_weight=0.0,
        delay_loss_weight=0.0,
        calibration_loss_weight=0.0,
    )
)

work_dir = "exps/thumos/pceh_endpoint_only"
