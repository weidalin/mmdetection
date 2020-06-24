_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_crowdhuman.py',
    '../_base_/datasets/crowdhuman_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model = dict(roi_head=dict(bbox_head=dict(num_classes=2)))
# # optimizer
# optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9

# runtime settings

work_dir = 'work_dirs/cascade_rcnn_r50_fpn_crowdhuman1000*600/'