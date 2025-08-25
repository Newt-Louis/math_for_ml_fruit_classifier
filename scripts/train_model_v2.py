import os, sys, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import random_split
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import classification_report


def main():
    # -------- CLI args --------
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    # -------- Paths --------
    BASE = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE / "data" / "raw" / "fruits-360_100x100" / "fruits-360"
    TRAIN_DIR = DATA_DIR / "Training"
    TEST_DIR = DATA_DIR / "Test"
    SAVE_DIR = BASE / "models";
    SAVE_DIR.mkdir(exist_ok=True)

    BEST_MODEL_PATH = SAVE_DIR / "fruit_classifier_best.pt"
    HISTORY_PATH = SAVE_DIR / "history.json"
    LABELS_PATH = SAVE_DIR / "labels.json"
    # [CẬP NHẬT] Thêm đường dẫn cho báo cáo kết quả cuối cùng
    REPORT_PATH = SAVE_DIR / "classification_report.json"

    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        sys.exit(f"❌ Không tìm thấy thư mục dữ liệu Training hoặc Test")

    # -------- Data --------
    # [CẬP NHẬT] Đổi tên transform cho rõ ràng hơn
    # Transform cho training có augmentation để mô hình học tốt hơn
    train_val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet yêu cầu input cố định
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transform cho test và validation không có augmentation
    test_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # [CẬP NHẬT] Chia bộ Training gốc thành Train và Validation
    # 1. Load toàn bộ dữ liệu training gốc
    full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_val_tfm)

    # 2. Xác định tỉ lệ chia, ví dụ 80% cho training, 20% cho validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # 3. Thực hiện chia ngẫu nhiên
    train_ds, val_ds = random_split(full_train_dataset, [train_size, val_size])

    # Tạo bộ dữ liệu Test
    test_ds = datasets.ImageFolder(TEST_DIR, transform=test_tfm)

    # Tạo các DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=args.workers)
    # [CẬP NHẬT] Tạo val_loader từ val_ds
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=32, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=32, num_workers=args.workers)

    # Lấy class names từ full_train_dataset
    class_names = full_train_dataset.classes
    print(f"ℹ️ Saving {len(class_names)} labels to {LABELS_PATH}")
    with open(LABELS_PATH, "w") as f:
        json.dump(class_names, f)

    # -------- Model --------
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # -------- Train & Validate --------
    print(f"🔧 Bắt đầu training {args.epochs} epoch(s) trên {device}")

    best_val_accuracy = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Train Epoch {epoch + 1}")
        running_loss = correct = total = 0

        for i, (x, y) in loop:
            if 0 < args.max_batches <= i: break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
            loop.set_postfix(loss=running_loss / (i + 1),
                             acc=f"{(100 * correct / total):.2f}%")

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            # [CẬP NHẬT] Sử dụng val_loader thay vì test_loader
            for x_val, y_val in tqdm(val_loader, desc=f"Validate Epoch {epoch + 1}"):
                x_val, y_val = x_val.to(device), y_val.to(device)
                out_val = model(x_val)
                loss_val = criterion(out_val, y_val)

                val_loss += loss_val.item()
                _, pred_val = out_val.max(1)
                val_total += y_val.size(0)
                val_correct += pred_val.eq(y_val).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        print(f"✅ Validation Epoch {epoch + 1}: Avg Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy * 100:.2f}%")

        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(
                f"🚀 New best model saved to {BEST_MODEL_PATH} with validation accuracy: {best_val_accuracy * 100:.2f}%")

    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)
    print(f"\n📈 Training history saved to {HISTORY_PATH}")

    # -------- [CẬP NHẬT] Final Testing Phase on Unseen Data --------
    print("\n" + "=" * 50)
    print("🏁 Bắt đầu đánh giá cuối cùng trên bộ Test (dữ liệu chưa từng thấy)")

    # Load lại model tốt nhất đã lưu
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_test, y_test in tqdm(test_loader, desc="Testing"):
            x_test, y_test = x_test.to(device), y_test.to(device)
            out_test = model(x_test)
            _, pred_test = out_test.max(1)

            all_preds.extend(pred_test.cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())

    # Tạo báo cáo classification
    print("\n📊 Báo cáo kết quả chi tiết (Classification Report):")
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # In ra màn hình một cách dễ đọc
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 75)
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            print(
                f"{class_name:<25} {metrics['precision']:<12.2f} {metrics['recall']:<12.2f} {metrics['f1-score']:<12.2f} {metrics['support']:<10}")
    print("-" * 75)

    # Lưu báo cáo ra file json
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"💾 Báo cáo đã được lưu vào: {REPORT_PATH}")

    print(
        f"\n✅ Hoàn tất! Model tốt nhất được lưu tại: {BEST_MODEL_PATH} với validation accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Accuracy trên tập test cuối cùng là: {report['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()