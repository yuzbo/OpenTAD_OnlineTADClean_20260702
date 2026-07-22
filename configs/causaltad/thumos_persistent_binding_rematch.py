_base_ = ["./thumos_persistent_binding_base.py"]

model = dict(trajectory_binding_mode="prefix_rematch_active_pool")
work_dir = "exps/thumos/persistent_binding_rematch"
