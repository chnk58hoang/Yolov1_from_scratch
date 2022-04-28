import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


def calculate_iou(box_pred, box_label, box_format='midpoint'):
    """
    :param box_preds: tensor[x,y,w,h] if midpoint or [x1,y1,x2,y2] if corner
    :param box_label:
    :param box_format: midpoint or corner
    :return:
    """
    if box_format == 'midpoint':
        box_preds_x1 = box_pred[..., 0:1] - box_pred[..., 2:3] / 2
        box_preds_y1 = box_pred[..., 1:2] - box_pred[..., 3:4] / 2
        box_preds_x2 = box_pred[..., 0:1] + box_pred[..., 2:3] / 2
        box_preds_y2 = box_pred[..., 1:2] + box_pred[..., 3:4] / 2


        box_label_x1 = box_label[..., 0:1] - box_label[..., 2:3] / 2
        box_label_y1 = box_label[..., 1:2] - box_label[..., 3:4] / 2
        box_label_x2 = box_label[..., 0:1] + box_label[..., 2:3] / 2
        box_label_y2 = box_label[..., 1:2] + box_label[..., 3:4] / 2

    if box_format == 'corners':
        box_preds_x1 = box_pred[..., 0:1]
        box_preds_y1 = box_pred[..., 1:2]
        box_preds_x2 = box_pred[..., 2:3]
        box_preds_y2 = box_pred[..., 3:4]

        box_label_x1 = box_label[..., 0:1]
        box_label_y1 = box_label[..., 1:2]
        box_label_x2 = box_label[..., 2:3]
        box_label_y2 = box_label[..., 3:4]

    x1 = torch.max(box_preds_x1, box_label_x1)
    y1 = torch.max(box_label_y1, box_preds_y1)
    x2 = torch.min(box_preds_x2, box_label_x2)
    y2 = torch.min(box_label_y2, box_preds_y2)

    intersection = torch.clamp((x2 - x1), 0) * torch.clamp((y2 - y1), 0)

    box_pred_area = abs(box_preds_x2 - box_preds_x1) * abs(box_preds_y2 - box_preds_y1)
    box_label_area = abs(box_label_x2 - box_label_x1) * abs(box_label_y2 - box_label_y1)

    return intersection / (box_label_area + box_pred_area - intersection + 0.00001)


def non_max_supression(pred_bboxes, iou_threshold, prob_threshold, box_format='corners'):
    """

    :param pred_bboxes: list of bounding boxes. Each bbox is represented by a list [class,probability,x1,y1,x2,y2]
    :param iou_threshold:
    :param prob_threshold:
    :param box_format:
    :return:
    """
    assert type(pred_bboxes) == list

    pred_bboxes_after_nms = []

    # Discard bounding boxes have probability smaller than prob_threshold
    pred_bboxes = [bbox for bbox in pred_bboxes if bbox[1] >= prob_threshold]
    # Sort the bboxes with descending of probability
    pred_bboxes = sorted(pred_bboxes, key=lambda x: x[1], reverse=True)

    while pred_bboxes:
        chosen_bbox = pred_bboxes.pop(0)

        # keep bounding boxes which have different class with current chosen bbox or iou with chosen_bbox < iou_threshold
        pred_bboxes = [bbox for bbox in pred_bboxes
                       if
                       bbox[0] != chosen_bbox[0] or calculate_iou(torch.tensor(chosen_bbox[2:]), torch.tensor(bbox[2:]),
                                                                 box_format=box_format) < iou_threshold]

        pred_bboxes_after_nms.append(chosen_bbox)

    return pred_bboxes_after_nms


def calculate_mAP(pred_bboxes, target_bboxes, iou_threshold=0.5, box_format='corners', num_classes=20):
    """

    :param pred_bboxes: list of [idx,class_label,probability,x1,y1,x2,y2]
    :param target_bboxes: list of [idx,class_label,x1,y1,x2,y2]
    :param iou_threshold:
    :param box_format:
    :param num_classes:
    :return:
    """

    average_precisions = []
    epsilon = 1e-6

    # Compute AP for each class
    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c

        for detection in pred_bboxes:
            if detection[1] == c:
                detections.append(detection)

        for target_bbox in target_bboxes:
            if target_bbox[1] == c:
                ground_truths.append(target_bbox)

        # Count number of target bboxes for each image
        # {index:number of bboxes}

        amount_bboxes = Counter(gt[0] for gt in ground_truths)

        # amount_bboxes: {0:torch.tensor([0,0,0]}. etc ....
        # 0: undetected bbox, 1: detected bbox
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # Sort all detection with descending of confidence score aka probability
        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_ground_truth = len(ground_truths)

        # Skip if none exist this class
        if total_ground_truth == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Get ground_truths of same image index with current detection
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = calculate_iou(torch.tensor(gt[3:]), torch.tensor(detection[3:]), box_format=box_format)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_ground_truth + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


def get_bboxes(
        loader,
        model,
        iou_threshold,
        threshold,
        pred_format="cells",
        box_format="midpoint",
        device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_supression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=threshold,
                box_format=box_format,
            )

            # if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


class EarlyStopping():
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_mAP = None
        self.early_stop = False
    def __call__(self, mAP):
        if self.best_mAP == None:
            self.best_mAP = mAP
        elif self.best_mAP - mAP > self.min_delta:
            self.best_mAP = mAP
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_mAP - mAP < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


