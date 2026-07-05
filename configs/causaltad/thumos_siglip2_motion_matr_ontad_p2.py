_base_ = "./thumos_siglip2_matr_ontad_p1.py"

# P2: paper-contribution route.
#
# The SigLIP2 vision tower remains frozen, but a trainable causal motion
# branch is enabled before the temporal adapter. This gives the method a
# streaming-safe motion cue instead of claiming novelty from SigLIP2 alone.

route_stage = "P2"
formal_training_ready = False

trainable_scope = dict(
    modules=[
        "backbone.backbone.motion_branch",
        "backbone.adapter",
        "backbone.out_proj",
        "backbone.norm",
        "projection",
        "neck",
        "rpn_head",
    ]
)

model = dict(
    backbone=dict(
        freeze_backbone=False,
        backbone=dict(
            use_motion_branch=True,
            motion_branch=dict(channels=768, kernel_size=3, causal=True, init_scale=0.1),
        ),
    ),
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=5e-5,
        weight_decay=0.05,
        custom=[
            dict(name="backbone.motion_branch", lr=1e-4, weight_decay=0.05),
            dict(name="adapter", lr=1e-4, weight_decay=0.05),
            dict(name="out_proj", lr=1e-4, weight_decay=0.05),
            dict(name="norm", lr=1e-4, weight_decay=0.0),
        ],
    ),
)

work_dir = "exps/thumos/siglip2_motion_matr_ontad_p2"
