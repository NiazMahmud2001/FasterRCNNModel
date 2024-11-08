from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score , confusion_matrix 

from PIL import Image
import cv2 

import torch 
import torch.nn as nn 

from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import os
import sys

from timeit import default_timer as timer 
from tqdm.auto import tqdm
from pathlib import Path



sys.stdout.reconfigure(encoding='utf-8') 
#this is for torchinfo=> summary

number_cpu_worker = os.cpu_count() 
print("number of cpu count: ", number_cpu_worker)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(device)


'''
labels: ObjectType      Truncation      Occlusion     Alpha      x1       y1        x2         y2        Height          Width        Length      X       Y       Z        RotationY
****   Pedestrian          0.00            0          -0.20    712.40   143.00    810.73    307.92        1.89           0.48           1.20     1.84    1.47    8.41         0.01


** ObjectType => Truck, Car, Cyclist, DontCare, etc 
** Truncation => 0.0 = fully visible , 1.0 = fully truncated
** Occlusion => 0:Fully visible.  1:Partially occluded.  2:Largely occluded.  3:Unknown (e.g., out of view).
** 2D Bounding Box Coordinates (x1, y1, x2, y2) 
** 3D Dimensions (Height, Width, Length)
** 3D Location (X, Y, Z)
** RotationY (-1.56 etc.)


*** Truck 0.00 0 -1.57 599.41 156.40 629.75 189.25 2.85 2.63 12.34 0.47 1.49 69.44 -1.56
* ObjectType: Truck
* Truncation: 0.00 (not truncated)
* Occlusion: 0 (fully visible)
* Alpha: -1.57 radians
* Bounding Box: (x1=599.41, y1=156.40, x2=629.75, y2=189.25)
* 3D Dimensions: Height=2.85m, Width=2.63m, Length=12.34m
* 3D Location: (X=0.47m, Y=1.49m, Z=69.44m)
* RotationY: -1.56 radians
'''

train_path = "G:/9th _semister/topics in cs/project/base4/data_object_image_2/training/image_2"
test_path = "G:/9th _semister/topics in cs/project/base4/data_object_image_2/testing/image_2"
root_calib = "G:/9th _semister/topics in cs/project/base4/data_object_calib"
train_label_path = "G:/9th _semister/topics in cs/project/base4/data_object_label_2/training/label_2"

label_colors = {
    'Cyclist': (0,128,255),
    'Van': (255,255,0),
    'Person_sitting': (0,255,255),
    'Pedestrian': (0,255,255),
    'Tram': (128,0,0),
    'Car': (255,0,0),
    'DontCare': (255,255,0),
    'Truck': (255,255,255),
    'Misc': (0,255,255)
}

CLASS_MAPPING = {
    "Car": 1,
    "Van": 2,
    "Truck": 3,
    "Pedestrian": 4,
    "Person_sitting": 5,
    "Cyclist": 6,
    "Tram": 7,
    "Misc": 8,
    "DontCare": 0
}
    
all_train_img_name = os.listdir(train_path)
all_train_label_name = os.listdir(train_label_path)

all_test_img_name = os.listdir(test_path)
len_of_test_img = len(all_train_img_name)

# analyze my datasets: ==============================================================
def analyzeImgShape(): 
    all_img_shape_row = [] # height
    all_img_shape_y_cols = []  # width
    all_img_shape_color_channel = [] 
    
    for x in all_train_img_name: 
        img_load = cv2.imread(train_path + "/" + x).shape
        all_img_shape_row.append(img_load[0])
        all_img_shape_y_cols.append(img_load[1])
        all_img_shape_color_channel.append(img_load[2])
    
    plt.figure(figsize=(12, 9))
    
    plt.subplot(2, 2, 1)
    plt.hist(all_img_shape_row, bins=20, color='blue', alpha=0.7)
    plt.title('Height Dist')
    plt.xlabel('Height')
    
    plt.subplot(2, 2, 2)
    plt.hist(all_img_shape_y_cols, bins=20, color='green', alpha=0.7)
    plt.title('Width Dist')
    plt.xlabel('Width')

    plt.subplot(2, 2, 3)
    plt.hist(all_img_shape_color_channel, bins=20, color='green', alpha=0.7)
    plt.title('COlor Channel Dist')
    plt.xlabel('COlor')
    plt.show()
    
# analyzeImgShape()

def get_classes(): 
    classAllSet = set()
    classAll = []
    
    for x in all_train_label_name: 
        label_path = train_label_path +"/"+ x
        with open(label_path, "r") as f : 
            for all_line in f: 
                word = all_line.strip().split(" ")
                classAllSet.add(word[0])
                classAll.append(word[0])
                
    plt.hist(classAll, bins=20)
    plt.title('Labels distributions')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()
    
# get_classes()      

def viewImages(img , vertices): 
    img = np.transpose(img, (1, 2, 0)).copy() 
    for box in vertices:
        x1, y1, x2, y2 = box 
        print(x1, y1, x2, y2)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=2)
        
    cv2.imshow("Bounding Boxes", img)
    cv2.waitKey(0)
              
              
              
class CustomImageFolderLoader(Dataset): 
    def __init__(self, transform=None, list_all_img_name=None , list_all_label_name=None): 
        self.transform = transform
        self.all_train_img_name = list_all_img_name
        self.all_train_label_name = list_all_label_name
        self.road_object = ["Car",
                            "Van", 
                            "Tram",
                            "Pedestrian",
                            "Cyclist",
                            "Truck", 
                            "Person_sitting", 
                            "Misc", 
                            ]
        # print(type(all_train_img_name) , " ", type(all_train_label_name))
        # print(self.all_train_img_name[0:5:1])
        # print(self.all_train_label_name[0:5:1])
        
        
        
    def __len__(self): 
        return len(self.all_train_label_name)
    
    def __getitem__(self, img_indx): 
        # 1st load the img based on idx=> index 
        # print(img_indx)
        
        img_path = train_path + "/" + all_train_img_name[img_indx]
        label_path = train_label_path +"/"+ all_train_label_name[img_indx]
        
        img = Image.open(img_path).convert("RGB")
        # print("the image path in __getitem__() is: ", img_path)
        # plt.imshow(img)
        # plt.show()

        bounding_box , labels = [] , [] 
        # one pic can have multiple object ... so add these bounding box and labels to array
        with open(label_path, "r") as f : 
            for all_line in f: 
                word = all_line.strip().split(" ") 
                '''
                all_line.strip().split(" ") is same as all_line.split(" ") but the difference is: all_line.split(" ") have "\n" in the array 
                but all_line.strip().split(" ") dont have "\n"
                '''
                # print(word[0])
                for index , val in enumerate(self.road_object):
                    if word[0] == val: 
                        x1, y1, x2, y2 = map(float, word[4:8]) # x1 , x2, y1, y2 are the boinding box points of each object 
                        bounding_box.append([x1, y1, x2, y2])
                        labels.append(torch.tensor(index+1))
                        # print(word[4], word[5], word[6], word[7], word[8], word[9])
                    
        
        target = {
            "boxes": bounding_box, 
            "labels": labels
        } 
        return self.transform(img) , target
            


# create the dataset and dataloader: ============================================================

custom_transform = transforms.Compose([
    # transforms.Resize((720,720)),
    transforms.ToTensor(), 
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # if needed then enable this line                   
])

def cusFinc(data): 
    return data


# initialize the file loader class: -================================================================
trainDataFolder = CustomImageFolderLoader(custom_transform, all_train_img_name, all_train_label_name)
train_dataloder = DataLoader(trainDataFolder, batch_size=4, shuffle=True, collate_fn=cusFinc)


temp_data = next(iter(train_dataloder))
print(type(temp_data), len(temp_data))
print(len(temp_data[0]))
# viewImages(iimg, bbox)
# print(len(train_img_batch[0]) , len(train_bounding_box_label_batch[0]))



# now import a pretained model for object detection from torchvision: =========================================
weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model0 = models.detection.fasterrcnn_resnet50_fpn(weights=weights, pretrained=True).to(device)

# visualize pre-trained model structure :=======================================================================
print("\n")
summary(model=model0.backbone,# just model0 for classification and model0.backbone=> for  object detection       
        input_size=(1, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        #col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)
print(model0)


# adjust the model input output classes: =====================================================================
num_classes = len(CLASS_MAPPING)
in_features = model0.roi_heads.box_predictor.cls_score.in_features
model0.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

for param in model0.backbone.parameters():
    param.requires_grad = False
for param in model0.rpn.parameters():
    param.requires_grad = False



# move all the sub-classes of pretrained model to device: =====================================================
model0.backbone.to(device)
model0.rpn.to(device)
model0.roi_heads.to(device)
model0.transform.to(device)


# print("\nSummary after adjusting the heads: ")
# summary(model=model0.backbone,# just model0 for classification and model0.backbone=> for  object detection       
#         input_size=(1, 3, 224, 224), # make sure this is "input_size", not "input_shape"
#         #col_names=["input_size"], # uncomment for smaller output
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )


# just for test: =========================================
# pred = model0(iimg)

# =========================================================

optimizer = torch.optim.Adam(model0.parameters(), lr=0.001)
num_epochs = 10


for epoch in range(num_epochs):
    model0.train()
    train_loss = 0
    
    # below for each single batch loop: =============================
    iter_img_till_now = 0
    total_loss_till_now = 0
    
    for indx, temp_batch in tqdm(enumerate(train_dataloder)):
        model0.train()
        images = []
        targets = []
        
        for datas_batch in temp_batch: 
            images.append(datas_batch[0].to(device))
            d = {}
            d['boxes'] = torch.as_tensor(datas_batch[1]["boxes"], dtype=torch.float32).to(device)
            d['labels'] = torch.as_tensor(datas_batch[1]["labels"], dtype=torch.int64).to(device)
            targets.append(d)
        
        #images = torch.as_tensor(images, dtype=torch.float).to(device)
        loss_dict = model0(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        train_loss += losses.item()
        
        # below for each single batch loop: =============================
        iter_img_till_now +=1
        total_loss_till_now += losses.item()
        print(f"\nloss per batch:{losses.item():.5f}  || and avg loss till now:{total_loss_till_now/iter_img_till_now:.3f}")
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if indx % 20 == 0:
            model0.eval()
            with torch.inference_mode(): 
                rand_img = np.random.randint(0 , len_of_test_img)
                img_view_name = all_test_img_name[rand_img]
                img_view = Image.open("G:/9th _semister/topics in cs/project/base4/data_object_image_2/testing/image_2/"+img_view_name).convert("RGB")
                image_np = np.array(img_view)
                img_view = [custom_transform(img_view).to("cuda")]
                
                
                predictions = model0(img_view)
                #print(predictions)
                threshold = 0.5
                boxes = predictions[0]['boxes']
                labels = predictions[0]['labels']
                scores = predictions[0]['scores']

                filtered_boxes = boxes[boxes > threshold]
                filtered_labels = labels[labels > threshold]
                filtered_scores = scores[scores > threshold] # scores are very much small ..... so you may not see them at first level

                print("Filtered Boxes:", filtered_boxes)
                print("Filtered Labels:", filtered_labels)
                print("Filtered Scores:", filtered_scores)


        
    iter_img_till_now = 0
    total_loss_till_now = 0
    print(f"loss per epoch:  [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_dataloder)}")
    
    # now get a random picture after 20 epoch from "testing" folder and view the pic 
    
    
# check predictipn after training: ============================================
image = Image.open("G:/9th _semister/topics in cs/project/base4/data_object_image_2/training/image_2/000002.png").convert("RGB")
image1 = Image.open("G:/9th _semister/topics in cs/project/base4/data_object_image_2/training/image_2/000003.png").convert("RGB")
image2 = Image.open("G:/9th _semister/topics in cs/project/base4/data_object_image_2/training/image_2/000004.png").convert("RGB")
image3 = Image.open("G:/9th _semister/topics in cs/project/base4/data_object_image_2/training/image_2/000005.png").convert("RGB")

image = custom_transform(image).unsqueeze(0).to("cuda")
image1 = custom_transform(image1).unsqueeze(0).to("cuda")
image2 = custom_transform(image2).unsqueeze(0).to("cuda")
image3 = custom_transform(image3).unsqueeze(0).to("cuda")



# Npow Set the model to evaluation mode and predict: ===>> 
model0.eval()
with torch.no_grad():  
    # Disable gradient computation and predict
    predictions = model0(image)
    predictions1 = model0(image1)
    predictions2 = model0(image2)
    predictions3 = model0(image3)

print(predictions)
print(predictions1)
print(predictions2)
print(predictions3)



# save the model: ===================================================================
modelPath ="G:/9th _semister/topics in cs/project/base4/saved_model/model2.pth"
#  Save the models state dict 
print(f"Saving model to: {modelPath}")
torch.save(obj=model0.state_dict(), f=modelPath) 
#"state_dict()" return all the parameters(learnable tensors) and buffers (non-learnable tensors)


cv2.destroyAllWindows()


# model1 have: avg loss till now:0.371 : paused the backbone and roi learning 

















































