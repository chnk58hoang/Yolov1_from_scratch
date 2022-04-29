# Yolov1_from_scratch
Yolov1 implemented from scratch with PyTorch


# Paper and Architecture
- This is the [link](https://arxiv.org/pdf/1506.02640.pdf) to the paper of Yolo V1
![image](https://user-images.githubusercontent.com/71833423/165345528-62e5a868-8415-4e6e-aff1-4b3202e6d29f.png)

- Architecture
![image](https://user-images.githubusercontent.com/71833423/165345794-2d9b1fb7-b4c5-4468-850a-50365bcbb195.png)


# Dataset
I use the [PascalVOC](https://www.kaggle.com/datasets/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2) dataset on Kaggle for this task.
![image](https://user-images.githubusercontent.com/71833423/165568897-ff5c338b-a74a-47cf-be45-c190b064d4ed.png)

# Requirements
I highly recommend using conda virtual environment
```bash
conda install pytorch torchvision
```
# Training 
I use GPU on Kaggle and evaluate metric is Mean Average Precision (mAP)
![image](https://user-images.githubusercontent.com/71833423/166062585-16bf531c-79dd-425f-9b3e-7c907b1866bd.png)


# Result 
![image](https://user-images.githubusercontent.com/71833423/165938222-3a4b59ef-6f8b-49e9-bfee-06284aee0d14.png)
