_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/crowdhuman_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
work_dir = 'work_dirs/cascade_rcnn_r50_fpn_crowdhuman/'