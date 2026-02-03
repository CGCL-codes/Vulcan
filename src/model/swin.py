from mmdet.apis import init_detector

def mask_rcnn_swin_t():
    # 绝对路径
    config = '/workspace/project/edgeseed/mmdetection/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
    checkpoint = '/workspace/project/edgeseed/model/weights/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth'
    model = init_detector(config, checkpoint)
    return model

# model = mask_rcnn_swin_t()
# print(model.backbone)

# module = model.backbone.stages[0].blocks[0].attn.w_msa.qkv
# print(module.weight.data.size())
# print(model.backbone.stages[0].blocks[0].attn.w_msa)

# print("分类层 fc_cls:", model.roi_head.bbox_head.fc_cls)
# print("分类层输出维度:", model.roi_head.bbox_head.fc_cls.out_features)

# print("回归层 fc_reg:", model.roi_head.bbox_head.fc_reg)
# print("回归层输出维度:", model.roi_head.bbox_head.fc_reg.out_features)
