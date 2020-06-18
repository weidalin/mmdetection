_base_ = './faster_rcnn_r50_fpn_1x_crowdhuman.py'
classes = ('person', )
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
