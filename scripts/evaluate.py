import torch
from torchvision import datasets, transforms, models
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np

# -------- Paths (Cập nhật cho khớp với train.py) --------
BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"
BEST_MODEL_PATH = MODEL_DIR / "fruit_classifier_best.pt"
HISTORY_PATH = MODEL_DIR / "history.json"
LABELS_PATH = MODEL_DIR / "labels.json"
TEST_DIR = BASE / "data" / "raw" / "fruits-360_100x100" / "fruits-360" / "Test"

# -------- Load Data and Model --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load labels
with open(LABELS_PATH, 'r') as f:
    classes = json.load(f)

# Load history
with open(HISTORY_PATH, 'r') as f:
    history = json.load(f)

# [QUAN TRỌNG] Phép transform cho tập test/đánh giá PHẢI giống hệt lúc training
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# -------- 1. Vẽ đồ thị Loss và Accuracy --------
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Đồ thị Loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Đồ thị Accuracy
    train_acc_percent = [a * 100 for a in history['train_acc']]
    val_acc_percent = [a * 100 for a in history['val_acc']]
    ax2.plot(train_acc_percent, label='Training Accuracy')
    ax2.plot(val_acc_percent, label='Validation Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')  # Hiển thị theo %
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('Model Training History')
    plt.show()

    best_epoch = np.argmax(history['val_acc'])
    best_acc = history['val_acc'][best_epoch]
    print(f"🚀 Hiệu suất tốt nhất trên tập validation: {best_acc * 100:.2f}% tại Epoch thứ {best_epoch + 1}")


# -------- 2. Tạo ma trận nhầm lẫn (Confusion Matrix) --------
def generate_confusion_matrix():
    print("\n📊 Đang tạo ma trận nhầm lẫn...")
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_transform)  # Sử dụng eval_transform
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Generating Confusion Matrix"):
            x = x.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(25, 25))  # Tăng kích thước để hiển thị rõ các nhãn
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xticks(rotation=90)  # Xoay nhãn trục x
    plt.yticks(rotation=0)
    plt.tight_layout()  # Tự động điều chỉnh layout
    plt.show()


# -------- 3. Hiển thị ví dụ dự đoán --------
def show_prediction_examples(num_examples=10):
    print("\n🖼️ Đang hiển thị một vài ví dụ dự đoán...")
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_transform)  # Sử dụng eval_transform
    # Lấy ảnh ngẫu nhiên để hiển thị
    indices = torch.randperm(len(test_ds)).tolist()[:num_examples]
    images = [test_ds[i][0] for i in indices]
    labels = [test_ds[i][1] for i in indices]

    # Chuẩn bị tensor để đưa vào model
    images_tensor = torch.stack(images).to(device)

    with torch.no_grad():
        outputs = model(images_tensor)
        _, preds = torch.max(outputs, 1)

    # Để hiển thị, chúng ta cần "un-normalize" ảnh, nhưng để đơn giản, ta sẽ hiển thị ảnh gốc trước khi transform
    raw_ds = datasets.ImageFolder(TEST_DIR, transform=transforms.Compose([transforms.Resize((224, 224))]))

    plt.figure(figsize=(15, 7))
    for i in range(num_examples):
        plt.subplot(2, 5, i + 1)
        # Lấy ảnh gốc để hiển thị cho đẹp
        raw_img, _ = raw_ds[indices[i]]
        plt.imshow(raw_img)

        true_label = classes[labels[i]]
        pred_label = classes[preds[i]]
        color = "green" if true_label == pred_label else "red"
        plt.title(f"Thật: {true_label}\nDự đoán: {pred_label}", color=color, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# -------- Main Execution --------
if __name__ == "__main__":
    plot_history(history)
    generate_confusion_matrix()
    show_prediction_examples()