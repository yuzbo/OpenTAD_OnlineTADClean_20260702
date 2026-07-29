import argparse

#argumentation setting
def make_parser():
    parser = argparse.ArgumentParser(description='hyperparameter options', add_help=False)

    # dataset
    parser.add_argument('--video_anno', default='./data/thumos14_v2.json', type=str)
    parser.add_argument('--video_len_file', default='./output/video_len_{}.json', type=str)
    parser.add_argument('--num_of_class', default=21, type=int)
    parser.add_argument('--num_frame', default=64, type=int)
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--flow', action='store_true')
    parser.add_argument('--video_feature_all_train', default='./data/thumos_all_feature_val_V3.pickle', type=str)
    parser.add_argument('--video_feature_all_test', default='./data/thumos_all_feature_test_V3.pickle', type=str)
    parser.add_argument('--ontal_label_file', default="./output/label_{}_{}_{}_{}_{}_{}_{}.h5", type=str)
    parser.add_argument('--num_queries', default=10, type=int)    
    
    parser.add_argument('--p_videos', default=1, type=int, help='number of parallel operation of videos')
    
    parser.add_argument('--detect_len', default=16, type=int)
    parser.add_argument('--anti_len', default=16, type=int)

    # model
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--training', default=True, type=bool)
    parser.add_argument('--feat_dim', default=4096, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--ffn_dim', default=2048, type=int)
    parser.add_argument('--e_nheads', default=8, type=int)
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--d_nheads', default=4, type=int)
    parser.add_argument('--dec_layers', default=5, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--activation', default="gelu", type=str)
    parser.add_argument('--max_memory_len', default=7, type=int)

    parser.add_argument('--memory_sampler', default='gap2', type=str, help='[gapN or all]')

    # Event-memory factorial.  The defaults are the untouched MATR behaviour:
    # detections are completed from the current segment plus past memory and
    # queries are freshly matched at every step.  The four experiment arms
    # change only these two factors.
    parser.add_argument(
        '--birth_mode',
        default='matr_delayed',
        choices=('matr_delayed', 'instant_transition'),
        type=str,
    )
    parser.add_argument(
        '--model_variant',
        default='native_matr',
        choices=('native_matr', 'eventmatr'),
        type=str,
        help='Exact official MATR or the eventized BxO model; never infer native from an event arm.',
    )
    parser.add_argument(
        '--ownership_mode',
        default='fresh_rematch',
        choices=('fresh_rematch', 'sticky_owner'),
        type=str,
    )
    parser.add_argument(
        '--event_arm',
        default=None,
        choices=('b0o0', 'b1o0', 'b0o1', 'b1o1'),
        type=str,
        help='Optional derived arm id; must agree with birth/ownership modes.',
    )
    parser.add_argument(
        '--event_birth_logit_threshold',
        default=None,
        type=float,
        help='Test-only threshold injection; formal EventMATR uses state argmax and leaves this unset.',
    )
    parser.add_argument(
        '--event_end_logit_threshold',
        default=None,
        type=float,
        help='Test-only threshold injection; formal EventMATR uses state argmax and leaves this unset.',
    )
    parser.add_argument(
        '--event_resource_limit',
        default=0,
        type=int,
        help='Physical fail-closed guard only; 0 means no semantic instance-count limit.',
    )
    parser.add_argument(
        '--event_lifecycle_version',
        default='v1_dense',
        choices=('v1_dense', 'd1_censored'),
        type=str,
        help='Frozen v1 dense prototypes or D1 chronological censored ragged learning.',
    )
    parser.add_argument(
        '--event_d1_lane',
        default='th',
        choices=('r', 't', 'h', 'th'),
        type=str,
        help='Registered D1 ablation: repair, trajectory, hazard, or their full combination.',
    )
    parser.add_argument(
        '--event_teacher_forcing_ratio',
        default=0.5,
        type=float,
        help='Pilot candidate for deterministic oracle/predicted track mixing; not a frozen paper hyperparameter.',
    )
    parser.add_argument(
        '--d11_effective_dose',
        action='store_true',
        help='D1.1-only control: train one epoch at the first registered warmup learning rate.',
    )
    parser.add_argument(
        '--study_protocol',
        default='upstream_native',
        choices=(
            'upstream_native',
            'matched_study',
            'd1_preexperiment',
            'd11_mechanism',
            'locked_test',
        ),
        type=str,
        help='Audit label separating upstream behaviour, formal training, D1 train-only gates, and one-shot test.',
    )
    
    parser.add_argument('--use_flag', action='store_true')
    parser.add_argument('--flag_threshold', default=0.5, type=float)

    # training
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=64, type=int)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--min_lr', default=1e-8, type=float)
    parser.add_argument('--max_lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_gamma', default=0.9, type=float)
    parser.add_argument('--lr_Tup', default=3, type=int)
    parser.add_argument('--lr_Tcycle', default=10, type=int)
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--drop_rate', default=0.3, type=float)    
    
    # criterion
    parser.add_argument('--use_empty_weight', action='store_true')
    parser.add_argument('--eos_coef', default=1e-8, type=float)
    parser.add_argument('--use_focal', action='store_true')
    parser.add_argument('--reduce', default=1, type=int)
    parser.add_argument('--cls_threshold', default=0.1, type=float)
    
    parser.add_argument('--cls_coef', default=1, type=int)
    parser.add_argument('--flag_coef', default=1, type=int)
    parser.add_argument('--reg_l1_coef', default=1, type=int)
    parser.add_argument('--reg_diou_coef', default=1, type=int)
    parser.add_argument('--reg_stcls_coef', default=1, type=int) 
    parser.add_argument('--event_identity_coef', default=1.0, type=float)
    
    # evaluation
    parser.add_argument('--nms_threshold', default=0.3, type=float)
    
    # output
    parser.add_argument('--make_output', action='store_true')
    parser.add_argument('--proposal_path', default='proposal_{}_{}_{}', type=str)
    
    # wandb
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--code_testing', action='store_true')
    
    # default setting
    parser.add_argument('--task', default="ontal", type=str)
    parser.add_argument('--dataset', default='thumos14', type=str)
    parser.add_argument('--random_seed', default=52, type=int)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--train_eval_step', default=5, type=int)
    parser.add_argument('--test_eval_step', default=1, type=int)
    
    # GPU
    parser.add_argument('--device', default="0", type=str)
    parser.add_argument('--num_workers', default=4, type=int)
     
    return parser.parse_args()
