_base_ = [
    "..\..\..\configs\_base_\schedules\sgd_100e.py",
    "..\..\..\configs\_base_\default_runtime.py",
]


# dataset settings
dataset_type = "VideoDataset"
data_root = "E:\ITMO\CVProject\data\slovo\\train"
data_root_val = "E:\ITMO\CVProject\data\slovo\\test"
ann_file_train = "E:\ITMO\CVProject\mmaction_datasets\\time\\train_ann.txt"
ann_file_val = "E:\ITMO\CVProject\mmaction_datasets\\time\\val_ann.txt"

file_client_args = dict(io_backend="disk")

model = dict(
    type="Recognizer2D",
    backbone=dict(
        type="ResNet",
        pretrained="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        depth=50,
        norm_eval=False,
    ),
    cls_head=dict(
        type="TSNHead",
        num_classes=10,
        in_channels=2048,
        spatial_type="avg",
        consensus=dict(type="AvgConsensus", dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips="prob",
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape="NCHW",
    ),
    train_cfg=None,
    test_cfg=None,
)

train_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=3),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(
        type="MultiScaleCrop",
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
    ),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(
        type="SampleFrames", clip_len=1, frame_interval=1, num_clips=3, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]
test_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(
        type="SampleFrames", clip_len=1, frame_interval=1, num_clips=25, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="TenCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
test_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(
    type="F1Metric",
    label2id_mapping_path="E:\ITMO\CVProject\mmaction_datasets\\time\\label2id_mapping.json",
)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="weighted_avg_f1_score",
        rule="greater",
    ),
)

vis_backends = [
    # dict(type="LocalVisBackend"),
    dict(
        type="ExtendedClearMLVisBackend",
        init_kwargs=dict(project_name="CVProject", task_name="train_time_package"),
    ),
]
visualizer = dict(type="ClearMLExtendedVisualizer", vis_backends=vis_backends)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=256)
