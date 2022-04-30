from torch.utils.data import DataLoader
from utils import *
from dataset import VOCDataset
from model import Yolov1
from train import Compose
from torchvision import transforms as transforms
import torch
import cv2
import argparse

IMG_DIR = "/kaggle/input/pascalvoc-yolo/images"
LABEL_DIR = "/kaggle/input/pascalvoc-yolo/labels"
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 32
WEIGHT_DECAY = 0
EPOCHS = 50
PIN_MEMORY = True




if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_dir',type=str)
    args = parser.parse_args()

    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])
    test_dataset = VOCDataset(
        "/kaggle/input/pascalvoc-yolo/100examples.csv", img_dir=IMG_DIR, label_dir=LABEL_DIR, transform=transform,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

    model.load_state_dict(torch.load(args.weight_dir,map_location=torch.device('cpu')))

    for x, y in test_loader:
                x = x.to(DEVICE)
                for idx in range(8):
                    bboxes = cellboxes_to_boxes(model(x))
                    bboxes = non_max_supression(bboxes[idx], iou_threshold=0.5, prob_threshold=0.4, box_format="midpoint")
                    plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)