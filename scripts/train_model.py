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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)   # có thể đặt 0 nếu muốn
    args = parser.parse_args()

    # -------- Paths --------
    BASE = Path(__file__).resolve().parents[1]
    TRAIN_DIR = BASE / "data" / "raw" / "fruits-360_100x100" / "fruits-360" / "Training"
    TEST_DIR  = BASE / "data" / "raw" / "fruits-360_100x100" / "fruits-360" / "Test"
    SAVE_DIR  = BASE / "models"; SAVE_DIR.mkdir(exist_ok=True)
    SAVE_PATH = SAVE_DIR / "fruit_classifier.pt"

    if not TRAIN_DIR.exists():
        sys.exit(f"❌ Không tìm thấy {TRAIN_DIR}")

    # -------- Data --------
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=tfm)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=tfm)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=args.workers)
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=32, num_workers=args.workers)
    
    # ✅ Save labels ngay sau khi train_dataset được tạo
    print(f"ℹ️ Saving labels to models/labels.json")
    with open("models/labels.json", "w") as f:
        json.dump(train_ds.classes, f)

    # -------- Model --------
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # -------- Train --------
    print(f"🔧 Train {args.epochs} epoch(s) on {device}")
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}")
        running_loss = correct = total = 0

        for i, (x, y) in loop:
            if 0 < args.max_batches <= i: break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()

            running_loss += loss.item()
            _, pred = out.max(1)
            total += y.size(0); correct += pred.eq(y).sum().item()
            loop.set_postfix(loss=running_loss/(i+1),
                             acc=100*correct/total)

        torch.save(model.state_dict(), SAVE_DIR / f"ckpt_epoch{epoch+1}.pt")
        print(f"💾 Saved ckpt_epoch{epoch+1}.pt")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Finished training. Final model: {SAVE_PATH}")

if __name__ == "__main__":
    main()
