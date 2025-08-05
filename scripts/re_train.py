import os, sys, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
import json


def main():
    # -------- CLI args --------
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    # -------- Paths --------
    BASE = Path(__file__).resolve().parents[1]
    TRAIN_DIR = BASE / "data" / "raw" / "fruits-360_100x100" / "fruits-360" / "Training"
    TEST_DIR = BASE / "data" / "raw" / "fruits-360_100x100" / "fruits-360" / "Test"
    SAVE_DIR = BASE / "models";
    SAVE_DIR.mkdir(exist_ok=True)

    # [CẬP NHẬT] Thêm các đường dẫn cho file history và labels
    BEST_MODEL_PATH = SAVE_DIR / "fruit_classifier_best.pt"
    HISTORY_PATH = SAVE_DIR / "history.json"
    LABELS_PATH = SAVE_DIR / "labels.json"

    if not TRAIN_DIR.exists():
        sys.exit(f"❌ Không tìm thấy {TRAIN_DIR}")

    # -------- Data --------
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfm)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=test_tfm)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=32, num_workers=args.workers)

    print(f"ℹ️ Saving labels to {LABELS_PATH}")
    with open(LABELS_PATH, "w") as f:
        json.dump(train_ds.classes, f)

    # -------- Model --------
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # -------- Train & Validate --------
    print(f"🔧 Train {args.epochs} epoch(s) on {device}")

    # [CẬP NHẬT] Thêm history dictionary và sửa tên biến best_accuracy
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
            loss.backward();
            optimizer.step()

            running_loss += loss.item()
            _, pred = out.max(1)
            total += y.size(0);
            correct += pred.eq(y).sum().item()
            loop.set_postfix(loss=running_loss / (i + 1),
                             acc=f"{(100 * correct / total):.2f}%")

        # [CẬP NHẬT] Lưu kết quả training vào history
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        torch.save(model.state_dict(), SAVE_DIR / f"ckpt_epoch{epoch + 1}.pt")
        print(f"💾 Saved ckpt_epoch{epoch + 1}.pt")

        # --- Validation Phase ---
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for x_val, y_val in tqdm(test_loader, desc=f"Validate Epoch {epoch + 1}"):
                x_val, y_val = x_val.to(device), y_val.to(device)
                out_val = model(x_val)
                loss_val = criterion(out_val, y_val)

                val_loss += loss_val.item()
                _, pred_val = out_val.max(1)
                val_total += y_val.size(0)
                val_correct += pred_val.eq(y_val).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = val_correct / val_total  # Dùng tỉ lệ 0-1 để lưu
        print(f"✅ Validation Epoch {epoch + 1}: Avg Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy * 100:.2f}%")

        # [CẬP NHẬT] Lưu kết quả validation vào history
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🚀 New best model saved to {BEST_MODEL_PATH} with accuracy: {best_val_accuracy * 100:.2f}%")

    # [CẬP NHẬT] Lưu file history và in ra kết quả cuối cùng
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)
    print(f"📈 Training history saved to {HISTORY_PATH}")

    print(
        f"✅ Finished training. Best model saved at: {BEST_MODEL_PATH} with validation accuracy: {best_val_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()