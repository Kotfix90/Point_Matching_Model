import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# CONSTANTS
# =====================
IMG_WIDTH, IMG_HEIGHT = 3200, 1800
RESIZE = 224

# =====================
# NORMALIZATION
# =====================
def normalize_point(p):
    return np.array([p[0] / IMG_WIDTH, p[1] / IMG_HEIGHT], dtype=np.float32)

def denormalize_point(p):
    return np.array([p[0] * IMG_WIDTH, p[1] * IMG_HEIGHT], dtype=np.float32)

# =====================
# MODEL
# =====================
class PointModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        self.encoder = backbone.features[:10]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.pool(self.encoder(dummy)).flatten(1)
            feat_dim = feat.shape[1]
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.pool(self.encoder(x)).flatten(1)

    def forward(self, door2_img, other_img, point, source):
        f1 = self.encode(door2_img)
        f2 = self.encode(other_img)
        x = torch.cat([f1, f2, point, source.unsqueeze(1)], dim=1)
        return self.fc(x)

# =====================
# TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((RESIZE, RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# CACHE
# =====================
image_cache = {}

def load_image(path):
    if path not in image_cache:
        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        image_cache[path] = img
    return image_cache[path]

# =====================
# GLOBAL CONTEXT
# =====================
model = PointModel().to(device)
door2_img = None
top_img = None
bottom_img = None

# =====================
# LOAD SESSION
# =====================
def load_session(session_path, model_relative_path="path/to/best_model.pth"):
    global door2_img, top_img, bottom_img

    if not os.path.exists(session_path):
        raise FileNotFoundError(f"Директория сессии не существует: {session_path}")

    print(f"Загрузка сессии: {session_path}")

    # Формируем путь к модели относительно текущей директории
    model_path = r"best_model.pth"

    print(f"Путь к модели: {model_path}")  # Для отладки

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    # Загружаем модель
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Загружаем изображения (остальная логика)
    for source, fname in [("top", "coords_top.json"), ("bottom", "coords_bottom.json")]:
        json_path = os.path.join(session_path, fname)
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        item = data[0]
        door2_path = os.path.join(session_path, item["file1_path"].replace("/", os.sep))
        other_path = os.path.join(session_path, item["file2_path"].replace("/", os.sep))

        door2_img = load_image(door2_path)

        if source == "top":
            top_img = load_image(other_path)
        else:
            bottom_img = load_image(other_path)

    print("Сессия загружена")

# =====================
# PREDICT INTERFACE
# =====================
def predict(x, y, source):
    if door2_img is None or (source == "top" and top_img is None) or (source == "bottom" and bottom_img is None):
        raise RuntimeError("Session not loaded. Call load_session() first.")

    if source == "top":
        img = top_img
        source_val = 1.0
    else:
        img = bottom_img
        source_val = 0.0

    point = torch.tensor(normalize_point([x, y]), dtype=torch.float32).unsqueeze(0).to(device)
    source_tensor = torch.tensor([source_val], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(door2_img, img, point, source_tensor)

    pred = pred.squeeze(0).cpu().numpy()
    pred = denormalize_point(pred)

    return float(pred[0]), float(pred[1])

# =====================
# MAIN (for CLI usage)
# =====================
if __name__ == "__main__":
    session_path = input("Введите путь к сессии: ").strip()
    load_session(session_path)

    mode = input("1 - одна точка, 2 - все точки: ")

    if mode == "1":
        x = float(input("x: "))
        y = float(input("y: "))
        source = input("source (top/bottom): ")
        px, py = predict(x, y, source)
        print("Predicted:", px, py)
    else:
        results = []
        for source, fname in [("top", "coords_top.json"), ("bottom", "coords_bottom.json")]:
            json_path = os.path.join(session_path, fname)
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                data = json.load(f)
            for item in data:
                for p in item["image2_coordinates"]:
                    x, y = p["x"], p["y"]
                    pred_x, pred_y = predict(x, y, source)
                    results.append({
                        "source": source,
                        "input_point": [x, y],
                        "predicted_point": [pred_x, pred_y]
                    })
        with open("predictions.json", "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved {len(results)} predictions")