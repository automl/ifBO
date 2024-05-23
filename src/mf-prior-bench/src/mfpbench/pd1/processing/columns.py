from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Column:
    name: str
    keep: bool = False
    type: type[str] | type[float] | type[list] | type | None = None
    rename: str | None = None
    hp: bool = False
    metric: bool = False


COLUMNS = [
    # =======
    # Describes a dataset
    # =======
    Column("dataset", keep=True, type=str),
    Column("model", keep=True, type=str),
    Column("hps.batch_size", keep=True, rename="batch_size"),
    # =======
    # Metrics
    # ==========
    # Both error rates in [0, 1] scale
    Column(
        "valid/error_rate",
        keep=True,
        type=list,
        rename="valid_error_rate",
        metric=True,
    ),  # list[float]
    # test error not available in imagnet, lm1b, translate_wmt, uniref50
    Column(
        "test/error_rate",
        keep=True,
        type=list,
        rename="test_error_rate",
        metric=True,
    ),
    Column("train_cost", keep=True, type=list, metric=True),
    # ==========
    # =======
    # Fidelity
    # ==========
    Column("epoch", keep=True, type=list),  # list[int]
    # =======
    # Some status about the model
    # ==============
    # In practice we can't use this at inference time but it would be nice to
    Column("status", keep=False, type=str, rename="completed"),
    # NOTE:
    # Could be useful to extract prior, need to be done before merging things and
    # then validate it's a valid prior. This is only useful in the context of the raw
    # data so I will leave it out for now
    Column("best_valid/error_rate_index", keep=False, type=int),
    # =======
    # Hps
    # =======
    # ---------------------------------
    # all
    Column(
        "hps.lr_hparams.decay_steps_factor",
        keep=True,
        type=float,
        rename="lr_decay_factor",
        hp=True,
    ),
    Column(
        "hps.lr_hparams.initial_value",
        keep=True,
        type=float,
        rename="lr_initial",
        hp=True,
    ),
    Column("hps.lr_hparams.power", keep=True, type=float, rename="lr_power", hp=True),
    Column(
        "hps.opt_hparams.momentum",
        keep=True,
        type=float,
        rename="opt_momentum",
        hp=True,
    ),
    # ---------------------------------
    # ---------------------------------
    # fashion_mnist-max_pooling_cnn-256
    # fashion_mnist-max_pooling_cnn-2048
    # mnist-max_pooling_cnn-256
    # mnist-max_pooling_cnn-2048
    Column(
        "hps.activation_fn",
        keep=True,
        type=str,
        rename="activation_fn",
    ),  # relu/tanh
    # ---------------------------------
    # =============
    # Drop: Constant per datasets, always NaN or irrelevant
    # =============
    Column("model_shape"),
    Column("trial_dir"),
    Column("global_step"),
    Column("learning_rate"),
    Column("preemption_count"),
    Column("steps_per_sec"),
    Column("study_dir"),
    Column("study_group"),
    Column("effective_lr"),
    Column("eval_time"),
    # metrics
    Column("test/ce_loss"),
    Column("test/denominator"),
    Column("train/ce_loss"),
    Column("train/denominator"),
    Column("train/error_rate"),
    Column("valid/ce_loss"),
    Column("valid/denominator"),
    # best
    Column("best_train_cost_index"),
    Column("best_train_cost"),
    Column("best_train_cost_step"),
    Column("best_train/error_rate_index"),
    Column("best_train/error_rate"),
    Column("best_train/error_rate_step"),
    Column("best_train/ce_loss_index"),
    Column("best_train/ce_loss"),
    Column("best_train/ce_loss_step"),
    Column("best_valid/error_rate"),
    Column("best_valid/error_rate_step"),
    Column("best_valid/ce_loss_index"),
    Column("best_valid/ce_loss"),
    Column("best_valid/ce_loss_step"),
    # hps
    Column("hparams"),
    Column("hps.attention_dropout_rate"),
    Column("hps.dropout_rate"),
    Column("hps.emb_dim"),
    Column("hps.l2_decay_factor"),
    Column("hps.l2_decay_rank_threshold"),
    Column("hps.label_smoothing"),
    Column("hps.lr_hparams.end_factor"),
    Column("hps.data_format"),
    Column("hps.lr_hparams.schedule"),
    Column("hps.model_dtype"),
    Column("hps.num_filters"),
    Column("hps.num_layers"),
    Column("hps.optimizer"),
    Column("hps.rng_seed"),
    Column("hps.use_shallue_label_smoothing"),
    Column("hps.virtual_batch_size"),
    Column("hps.input_shape"),
    Column("hps.output_shape"),
    Column("hps.train_size"),
    Column("hps.valid_size"),
    Column("hps.normalizer"),
    Column("hps.num_heads"),
    Column("hps.qkv_dim"),
    Column("hps.data_name"),
    Column("hps.max_eval_target_length"),
    Column("hps.max_target_length"),
    Column("hps.dec_num_layers"),
    Column("hps.enc_num_layers"),
    Column("hps.logits_via_embedding"),
    Column("hps.share_embeddings"),
    Column("hps.eval_split"),
    Column("hps.max_corpus_chars"),
    Column("hps.max_predict_length"),
    Column("hps.pack_examples"),
    Column("hps.reverse_translation"),
    Column("hps.tfds_dataset_key"),
    Column("hps.tfds_eval_dataset_key"),
    Column("hps.train_split"),
    Column("hps.vocab_size"),
    Column("hps.kernel_paddings"),
    Column("hps.kernel_sizes"),
    Column("hps.num_dense_units"),
    Column("hps.strides"),
    Column("hps.window_paddings"),
    Column("hps.window_sizes"),
    Column("hps.test_size"),
    Column("hps.mlp_dim"),
    Column("hps.activation_function"),  # This one is constant, unlike "activation_fn"
    Column("hps.blocks_per_group"),
    Column("hps.channel_multiplier"),
    Column("hps.conv_kernel_init"),
    Column("hps.conv_kernel_scale"),
    Column("hps.dense_kernel_init"),
    Column("hps.dense_kernel_scale"),
    Column("hps.alpha"),
    Column("hps.crop_num_pixels"),
    Column("hps.flip_probability"),
    Column("hps.use_mixup"),
    Column("hps.batch_norm_epsilon"),
    Column("hps.batch_norm_momentum"),
]

# TODO: Should really move this
DATASET_NAMES = [
    "cifar100-wide_resnet-2048",
    "cifar100-wide_resnet-256",
    "cifar10-wide_resnet-2048",
    "cifar10-wide_resnet-256",
    "fashion_mnist-max_pooling_cnn-2048",
    "fashion_mnist-max_pooling_cnn-256",
    "fashion_mnist-simple_cnn-2048",
    "fashion_mnist-simple_cnn-256",
    "imagenet-resnet-1024",
    "imagenet-resnet-256",
    "imagenet-resnet-512",
    "lm1b-transformer-2048",
    "mnist-max_pooling_cnn-2048",
    "mnist-max_pooling_cnn-256",
    "mnist-simple_cnn-2048",
    "mnist-simple_cnn-256 ",
    "svhn_no_extra-wide_resnet-1024",
    "svhn_no_extra-wide_resnet-256",
    "translate_wmt-xformer_translate-64",
    "uniref50-transformer-128",
]
