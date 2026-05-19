import os

from torch.utils.data import DataLoader

from yolo_count.models.yolocount import build_yolocount_model_base
from yolo_count.utils.dataload import FSCData
from yolo_count.utils.fn import auto_load
from yolo_count.utils.validation import evaluate_on_fsc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ckpt_path = "F://YOLO_Count_Checkpoints/yolocnt_fsc147_epoch300.pth"

model = build_yolocount_model_base()
auto_load(model, ckpt_path)
model.eval().to("cuda")

large_threshold = 130
confidence_threshold = 0.0

# batch_size=1，显存占用约2.2G,CPU满载，推理耗时约70秒
# batch_size=8，显存占用约3.7G，CPU占用约35%，推理耗时约200秒
# batch_size=16，显存占用约6.5G，CPU占用约20%，推理耗时约240秒
# batch_size=32，显存占用约10.9G，CPU占用约10%，推理耗时约300秒
# GPU全程占用率约80%
# 说明该模型虽然参数量小，但推理步骤多
val_dataloader = DataLoader(
    # 这里是标签文件的路径
    # 图片文件的路径在dataloader文件中修改
    FSCData(root="../data/FSC", split="val", flip=False),
    batch_size=1,
    num_workers=0,
    shuffle=False,
)
test_dataloader = DataLoader(
    # 这里是标签文件的路径
    # 图片文件的路径在dataloader文件中修改
    FSCData(root="../data/FSC", split="test", flip=False),
    batch_size=1,
    num_workers=0,
    shuffle=False,
)

MAE_test, RMSE_test, Bias_test = evaluate_on_fsc(
    model,
    test_dataloader,
    large_threshold=large_threshold,
    confidence_threshold=confidence_threshold,
)
print(f"Test - MAE: {MAE_test:.4f}, RMSE: {RMSE_test:.4f}, Bias: {Bias_test:.4f}")

MAE_val, RMSE_val, Bias_val = evaluate_on_fsc(
    model,
    val_dataloader,
    large_threshold=large_threshold,
    confidence_threshold=confidence_threshold,
)
print(f"Validation - MAE: {MAE_val:.4f}, RMSE: {RMSE_val:.4f}, Bias: {Bias_val:.4f}")
