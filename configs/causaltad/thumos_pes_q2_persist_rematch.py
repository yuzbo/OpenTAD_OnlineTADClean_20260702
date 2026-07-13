_base_ = ["./thumos_pes_q2_base.py"]

model = dict(trajectory_binding_mode="prefix_rematch_active_pool")
work_dir = "exps/thumos/pes_q2_persist_rematch"
