_base_ = ["./thumos_i3d_n16r4.py"]

formal_training_ready = True

model = dict(
    rpn_head=dict(
        type="MATRHead",
        online=True,
        memory_size=0,
        clamp_end_to_current=True,
        max_future_offset=0.0,
        online_censored_training=True,
        boundary_loss_weight=0.2,
        actionness_loss_weight=0.1,
        emit_loss_weight=0.1,
        use_boundary_scores=True,
        use_actionness_scores=True,
        use_emit_scores=True,
        cofie_enabled=False,
    )
)

work_dir = "exps/thumos/matr_i3d_n16r4"
