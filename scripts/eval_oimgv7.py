import os

from torch.utils.data import DataLoader

from yolo_count.models.yolocount import build_yolocount_model_base
from yolo_count.utils.dataload import OImgv7Data
from yolo_count.utils.fn import auto_load
from yolo_count.utils.validation import evaluate_on_oimgv7

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ckpt_path = "F://YOLO_Count_Checkpoints/yolocnt_lvis_obj365_oimgv7_epoch300.pth"

model = build_yolocount_model_base()
auto_load(model, ckpt_path)
model.eval().to("cuda")

confidence_threshold = 0.0

val_dataloader = DataLoader(
    # 这里是标签文件的路径
    # 图片文件的路径在dataloader文件中修改
    OImgv7Data(root="../data/OImgv7", split="validation"),
    batch_size=1,
    num_workers=0,
    shuffle=False,
)

MAE_val, RMSE_val = evaluate_on_oimgv7(
    model, val_dataloader, confidence_threshold=confidence_threshold
)
print(f"Validation - MAE: {MAE_val:.4f}, RMSE: {RMSE_val:.4f}")
