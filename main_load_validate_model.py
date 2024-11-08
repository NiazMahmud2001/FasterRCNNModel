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
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("G:/9th _semister/topics in cs/project/base4/data_object_image_2/training/image_2/000010.png").convert("RGB")
image_np = np.array(image)
print("np image shape: ", transform(image).shape)
image = [transform(image).to("cuda")]
print("img shaep: ", len(image))


# Npow Set the model to evaluation mode and predict: ===>> 
model.eval()
with torch.no_grad():  
    # Disable gradient computation and predict
    predictions = model(image)

# print(predictions)

# Adjust threshold and print results
threshold = 0.5
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

filtered_boxes = boxes[scores > threshold]
filtered_labels = labels[scores > threshold]
filtered_scores = scores[scores > threshold]

print("Filtered Boxes:", filtered_boxes)
print("Filtered Labels:", filtered_labels)
print("Filtered Scores:", filtered_scores)





# now draw the  bounding box on the picture: ========================================

for i in range(len(filtered_boxes)):
    box = filtered_boxes[i].cpu().numpy().astype(int)  # Convert box to integer coordinates
    label = filtered_labels[i].item()
    score = filtered_scores[i].item()
    
    cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
    
    # Add the label and score above the bounding box
    label_text = f"{label}: {score:.2f}"
    cv2.putText(image_np, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Convert image to BGR (from RGB) for OpenCV compatibility and display
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
cv2.imshow("Image with Bounding Boxes", image_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
# model1 have: avg loss till now:0.371









































