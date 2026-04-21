import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =====================
# CONSTANTS
# =====================
IMG_WIDTH, IMG_HEIGHT = 3200, 1800
RESIZE = 224  # уменьшили для ускорения

# =====================
# NORMALIZATION
# =====================
def normalize_point(p):
    x, y = p
    return np.array([x / IMG_WIDTH, y / IMG_HEIGHT], dtype=np.float32)

def denormalize_point(p):
    x, y = p
    return np.array([x * IMG_WIDTH, y * IMG_HEIGHT], dtype=np.float32)

# =====================
# DATASET (НЕ МЕНЯЕМ)
# =====================
class PointDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None):
        with open(split_file, 'r') as f:
            split = json.load(f)

        self.sessions = split["train"] if "train" in root_dir else split["val"]
        self.root_dir = os.path.dirname(root_dir) if "train" in root_dir or "val" in root_dir else root_dir

        self.transform = transform
        self.data = []

        for session in self.sessions:
            session_path = os.path.join(self.root_dir, session)

            for source, fname in [("top", "coords_top.json"), ("bottom", "coords_bottom.json")]:
                path = os.path.join(session_path, fname)
                if not os.path.exists(path):
                    continue

                with open(path) as f:
                    coords_data = json.load(f)

                for item in coords_data:
                    pts1 = item["image1_coordinates"]
                    pts2 = item["image2_coordinates"]

                    if len(pts1) != len(pts2) or len(pts1) == 0:
                        continue

                    door2_path = os.path.join(session_path, item["file1_path"].replace("/", os.sep))
                    other_path = os.path.join(session_path, item["file2_path"].replace("/", os.sep))

                    if not (os.path.exists(door2_path) and os.path.exists(other_path)):
                        continue

                    for p1, p2 in zip(pts1, pts2):
                        self.data.append({
                            "door2_path": door2_path,
                            "other_path": other_path,
                            "p_door2": [p1["x"], p1["y"]],
                            "p_other": [p2["x"], p2["y"]],
                            "source": 1.0 if source == "top" else 0.0
                        })

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        door2_img = Image.open(item["door2_path"]).convert("RGB")
        other_img = Image.open(item["other_path"]).convert("RGB")

        if self.transform:
            door2_img = self.transform(door2_img)
            other_img = self.transform(other_img)

        p_other = torch.tensor(normalize_point(item["p_other"]), dtype=torch.float32)
        p_door2 = torch.tensor(normalize_point(item["p_door2"]), dtype=torch.float32)
        source = torch.tensor(item["source"], dtype=torch.float32)

        return door2_img, other_img, p_other, p_door2, source

# =====================
# LIGHT MODEL
# =====================
class PointModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        self.encoder = backbone.features[:10]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 🔥 автоматически считаем размер
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
        x = self.encoder(x)
        x = self.pool(x)
        return x.flatten(1)

    def forward(self, door2_img, other_img, point, source):
        f1 = self.encode(door2_img)
        f2 = self.encode(other_img)

        x = torch.cat([f1, f2, point, source.unsqueeze(1)], dim=1)
        return self.fc(x)

# =====================
# METRIC
# =====================
def calculate_med(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    pred = np.array([denormalize_point(p) for p in pred])
    target = np.array([denormalize_point(p) for p in target])

    dist = np.sqrt(np.sum((pred - target) ** 2, axis=1))
    return np.mean(dist)

# =====================
# TRAIN
# =====================
def train():
    transform = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_path = r"C:\Users\Vyacheslave\Desktop\test\Mapping\data_set\test-task\test-task\train"
    val_path = r"C:\Users\Vyacheslave\Desktop\test\Mapping\data_set\test-task\test-task\val"
    split_file = r"C:\Users\Vyacheslave\Desktop\test\Mapping\data_set\test-task\test-task\split.json"

    train_ds = PointDataset(train_path, split_file, transform)
    val_ds = PointDataset(val_path, split_file, transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    model = PointModel().to(device)

    # замораживаем encoder (ускорение)
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 100
    patience = 5
    min_delta = 1e-5

    best_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        # ===== TRAIN =====
        model.train()
        train_loss = 0

        for door2, other, p_other, p_door2, source in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            door2, other = door2.to(device), other.to(device)
            p_other, p_door2 = p_other.to(device), p_door2.to(device)
            source = source.to(device)

            optimizer.zero_grad()

            pred = model(door2, other, p_other, source)
            loss = criterion(pred, p_door2)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ===== VAL =====
        model.eval()
        val_loss = 0
        val_med = 0

        with torch.no_grad():
            for door2, other, p_other, p_door2, source in val_loader:
                door2, other = door2.to(device), other.to(device)
                p_other, p_door2 = p_other.to(device), p_door2.to(device)
                source = source.to(device)

                pred = model(door2, other, p_other, source)

                loss = criterion(pred, p_door2)
                val_loss += loss.item()

                val_med += calculate_med(pred, p_door2)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_med = val_med / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val MED: {avg_val_med:.2f}")

        # ===== EARLY STOPPING =====
        if best_loss - avg_val_loss > min_delta:
            best_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping triggered")
            break

    print("Training done")

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    train()