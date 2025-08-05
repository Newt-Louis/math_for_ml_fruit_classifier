import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from src.core.config import settings
from torchvision.models import ResNet18_Weights

# ✅ Load danh sách nhãn từ file để linh hoạt
import json
from pathlib import Path

LABELS_PATH = Path("models/labels.json")
if LABELS_PATH.exists():
    with open(LABELS_PATH, "r") as f:
        LABELS = json.load(f)
else:
    # fallback nếu chưa có file
    LABELS = ["apple", "banana", "orange", "grape", "watermelon"]

class FruitClassifierModel:
    def __init__(self, model_path: str, image_size=(100, 100), num_classes: int = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if num_classes is None:
            num_classes = len(LABELS)

        self.model = self._build_model(num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _build_model(self, num_classes: int):
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(self.device)

    def predict(self, image: Image.Image) -> str:
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
        return LABELS[predicted_idx]


# ✅ Singleton để tránh load lại model nhiều lần
_model_instance: FruitClassifierModel = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = FruitClassifierModel(
            model_path=settings.model_path,
            image_size=settings.image_size
        )
    return _model_instance
