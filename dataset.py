import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VOCDataset(Dataset):
    def __init__(self, csv, img_dir, label_dir,transform = None, S=7, B=2, C=20):
        super(VOCDataset, self).__init__()
        self.annotations = pd.read_csv(csv)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        if transform:
            self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Get label file path for each image
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])

        # list of all bounding boxes
        boxes = []

        with open(label_path) as lb:
            for line in lb.readlines():
                class_label, x, y, w, h = [float(num) if float(num) != int(float(num)) else int(num)
                                           for num in line.replace("\n", "").split()]

                boxes.append([class_label, x, y, w, h])

        boxes = torch.tensor(boxes)

        # Get image
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        image, boxes = self.transform(image,boxes)

        # Label tensor for image
        label_tensor = torch.zeros(self.S, self.S, (self.C + self.B * 5))

        # get the label and convert from image to relative to each cell
        for box in boxes:
            class_label, x, y, w_image, h_image = box.tolist()
            class_label = int(class_label)

            # get i,j represents the row and collumn of this cell
            i, j = int(self.S * y), int(self.S * x)

            # Get x_cell,y_cell are coordinates of center point relative to this cell
            x_cell, y_cell = self.S * x - j, self.S * y - i

            # width_cell,height_cell are size of bounding box convert to relative with each cell
            """
            w_box = w * image_width
            cell_width = image_width / S
            => width_cell = w *  S
            """

            width_cell, height_cell = w_image * self.S, h_image * self.S

            if label_tensor[i, j, 20] == 0:
                label_tensor[i, j, 20] = 1

                bbox_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                label_tensor[i,j,21:25] = bbox_coordinates
                label_tensor[i,j,class_label] = 1

        return image,label_tensor
