_base_ = ["./thumos_i3d_matr_n16r4.py"]

model = dict(
    rpn_head=dict(
        cofie_enabled=True,
        cofie_channels=64,
        cofie_loss_weight=1.0,
        cofie_threshold=0.5,
    )
)

work_dir = "exps/thumos/cofie_i3d_n16r4"
