import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
import torchvision
from PIL import Image
from pathlib import Path
import numpy as np 
import cv2

num_classes = 9

# Load the saved model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("G:/9th _semister/topics in cs/project/base4/saved_model/model0.pth"))
model = model.eval().to("cuda")

# print("new created loaded model: ", model.state_dict())
# Now Prepare your input image
transform = transforms.Compose([
    transforms.ToTensor(),
])


vid = cv2.VideoCapture(1)
crop_width  = 1242 
crop_height = 375 

width = 1280 
height = 720 
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

start_x = (width - crop_width) // 2
start_y = (height - crop_height) // 2

cam = True
labels_name = ["DontCare","Car","Van","Truck","Pedestrian","Person_sitting","Cyclist","Tram","Misc"]
while cam==True:
    ret, frame = vid.read()
    frame = frame[start_y:start_y+crop_height, start_x:start_x+crop_width] 
    # croped the captured image for matching out model requirement
    height, width, shape = frame.shape
    
    imgPil = Image.fromarray(frame).convert("RGB")
    transform_img = transform(imgPil) 
    # transform_img [color channel , height, width] (for pytorch and PIL) and np requires:(height , width , color channel)
    pil_transformed_img = [transform_img.to("cuda")]

    
    model.eval()
    with torch.no_grad():  
        # Disable gradient computation and then predict
        predictions = model(pil_transformed_img)
    
    threshold = 0.38
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    filtered_boxes = boxes[scores > threshold]
    filtered_labels = labels[scores > threshold]
    filtered_scores = scores[scores > threshold]
    
    for i in range(len(filtered_boxes)):
        box = filtered_boxes[i].cpu().numpy().astype(int)  # Convert box to integer coordinates
        label = filtered_labels[i].item()
        score = filtered_scores[i].item()
        
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
        
        # Add the label and score above the bounding box
        cv2.putText(frame, labels_name[int(label)], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    image_np = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("My cam1 Image: ", image_np)
    k = cv2.waitKey(100)
    if k == ord("q"):
        break

# Release the video capture object and close the windows
vid.release()
cv2.destroyAllWindows()









































