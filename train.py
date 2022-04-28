import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from utils import *
from dataset import VOCDataset
from loss1 import YoloLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 32
WEIGHT_DECAY = 0
EPOCHS = 50
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR = "/kaggle/input/pascalvoc-yolo/images"
LABEL_DIR = "/kaggle/input/pascalvoc-yolo/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])


def train_model(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def valid_model(test_loader, model, loss_fn):
    loop = tqdm(test_loader, leave=True)
    mean_loss = []

    with torch.no_grad:
        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            mean_loss.append(loss.item())
            # update progress bar
            loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    lr_scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3)

    train_dataset = VOCDataset(
        "/kaggle/input/pascalvoc-yolo/train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "/kaggle/input/pascalvoc-yolo/100examples.csv", img_dir=IMG_DIR, label_dir=LABEL_DIR, transform=transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    early_stopping = EarlyStopping(patience=3, min_delta=0.01)

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = calculate_mAP(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"mAP: {mean_avg_prec}")
        early_stopping(mean_avg_prec)
        if early_stopping.early_stop:
            break
        else:
            print(f'Epoch: {epoch + 1}/{EPOCHS}')
            train_model(train_loader, model, optimizer, loss_fn)
            lr_scheduler.step(mean_avg_prec)
            if (epoch + 1) % 10 == 0:
                save_checkpoint(state=model.state_dict(), filename='yolov1' + str(epoch + 1) + '.pth.tar')


if __name__ == "__main__":
    main()
