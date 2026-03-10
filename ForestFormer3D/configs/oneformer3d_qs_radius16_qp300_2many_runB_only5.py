_base_ = ['./oneformer3d_qs_radius16_qp300_2many.py']

# Run B: biology-only taxonomy change
# - only class 2 is a thing/instance class
model = dict(
    thing_cls=[2],
    test_cfg=dict(
        stuff_cls=[0],
        thing_cls=[2]))

inst_mapping = [2]
val_evaluator = dict(
    thing_class_inds=[2],
    inst_mapping=inst_mapping)
test_evaluator = val_evaluator
