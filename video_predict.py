import os
import cv2
import torch
import numpy as np
import csv
from SPN.SPNet import SemiSPNet, SPNet
from ProUtils.misc import vl2ch
from SegUtils.transforms import vl2im

MODEL_PATH = "C:/Users/User/Desktop/TRP/MICCAI20_SiamParseNet/res2/snapshots-spn-semi-anneal-T0.40-B0.67/B0020_S060000.pth"
GPU_ID = "0"
INPUT_SIZE = (256, 256)
NUM_CLASSES = 5
TOPK = 20
OUTPUT_DIR = "C:/Users/User/Desktop/TRP/MICCAI20_SiamParseNet/KOTOVA"

masks_dir  = os.path.join(OUTPUT_DIR, "masks")
frames_dir = os.path.join(OUTPUT_DIR, "frames")
os.makedirs(masks_dir,  exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)

#Загрузка модели
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
device = torch.device("cuda")
model = SPNet(h=INPUT_SIZE[0]//8,
                  w=INPUT_SIZE[1]//8,
                  K=TOPK,
                  num_classes=NUM_CLASSES)
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["state_dict"])
model.to(device).eval()

#Открываем видео
video_path = "C:/Users/User/Desktop/TRP/video/Котова повтор 2 - Trim.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video {video_path}")

#Функция предобработки
def preprocess(frame):
    f = cv2.resize(frame, INPUT_SIZE, cv2.INTER_LINEAR).astype(np.float32)
    IMG_MEAN = np.array((101.84, 112.10, 111.66), dtype=np.float32)
    f -= IMG_MEAN
    f = f.transpose(2, 0, 1)
    return torch.from_numpy(f).unsqueeze(0).to(device)

#Фиктивная маска (one-hot) для SPN
dummy  = torch.zeros(1, *INPUT_SIZE, dtype=torch.uint8)  # CPU: [1,H,W]
onehot = vl2ch(dummy).to(device)                         # GPU: [1,H,W,C]

#Первый кадр и начальные маски
ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("Empty video")
prev_t = preprocess(prev_frame)
with torch.no_grad():
    _, seg_prev, _, _ = model(prev_t, prev_t, onehot, onehot)
class_prev = seg_prev.argmax(1).cpu().numpy()[0]    # H×W, значения 0–4
mask_prev   = vl2im(class_prev)                     # H×W×3, BGR
#Сохраним первую маску
cv2.imwrite(os.path.join(masks_dir,  f"mask_{1:04d}.png"), mask_prev)

movements = []
features  = []  #собираем [frame_idx, motion, area_head, area_hand, area_body, area_foot]
idx = 2         #следующий кадр

#Цикл по кадрам
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #сегментация
    cur_t = preprocess(frame)
    with torch.no_grad():
        _, seg_cur, _, _ = model(cur_t, cur_t, onehot, onehot)
    class_cur = seg_cur.argmax(1).cpu().numpy()[0]      # H×W

    #пересчёт движения и площадей
    motion = float((class_cur != class_prev).mean())
    areas  = [ int((class_cur == c).sum()) for c in (1,2,3,4) ]  # head, hand, body, foot

    movements.append((idx, motion))
    features.append([idx, motion] + areas)

    #строим цветную маску и оверлей
    mask_cur = vl2im(class_cur)  # H×W×3
    #масштабируем на оригинальный размер кадра
    mask_resized  = cv2.resize(mask_cur,
                               (frame.shape[1], frame.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(frame, 0.7, mask_resized, 0.3, 0)

    #сохраняем
    cv2.imwrite(os.path.join(masks_dir,  f"mask_{idx:04d}.png"), mask_resized)
    cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:04d}_overlay.png"), overlay)

    print(f"Processed {idx} frames.")

    #подготовка к следующей итерации
    class_prev = class_cur
    idx += 1

cap.release()

#Сохраняем features.csv
csv_path = os.path.join(OUTPUT_DIR, "features.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_idx","motion_score","area_head","area_hand","area_body","area_foot"])
    writer.writerows(features)

print(f"Processed {idx-1} frames.")
print(f"Features saved to {csv_path}")
