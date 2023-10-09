import yolov5
import pandas as pd
import os
import cv2
from tqdm import tqdm

model_num = input("Enter the model version number: ") #Mention the model version eg: v132 would be '132'
model_path = rf'..\Trained models\v{model_num}\best.pt'
image_folder = r'..\Test images\images'
csv_path = rf'..\Tested models result csv\Test Result x{model_num} (v5).csv'
txt_folder = r'..\Test images\labels'

model = yolov5.load(model_path)  # Loading the custom model
image_files = os.listdir(image_folder)

df = pd.DataFrame(columns=['Filename', 'Ground Truth', 'Ground Truth BBox Area', 'Identified Class', 'Confidence Score', 'Det Counts Metric', 'BBox Area Metric']) # Create an empty DataFrame to store the results

# Read the YOLO text files and extract the information
for image_file in tqdm(image_files):
   image_name = os.path.splitext(image_file)[0]
   txt_file = os.path.join(txt_folder, image_name + '.txt')

   # Read the YOLO text file if it exists
   if os.path.isfile(txt_file):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        # Extract the class names and bounding box areas
        class_names = []
        bbox_areas = []
        for line in lines:
            class_id, x, y, w, h = line.strip().split(' ')
            if int(class_id) < 0 or int(class_id) > len(model.names):
                continue
            else:
                class_name = model.names[int(class_id)]
            class_names.append(class_name)
            bbox_area = float(w) * float(h)
            bbox_areas.append(bbox_area)
   else:
        class_names = []
        bbox_areas = []

    # Combine the class names and bounding box areas into a single string
   class_names_str = ','.join(class_names)
   bbox_areas_str = ','.join([str(area) for area in bbox_areas])

   # Add the file name, class names, and bounding box areas to the DataFrame
   df = df._append({'Filename': image_file, 'Ground Truth': class_names_str, 'Ground Truth BBox Area': bbox_areas_str}, ignore_index=True)
   image_path = os.path.join(image_folder, image_file)
   image = cv2.imread(image_path)
   results = model(image_path)
   scores = results.xyxy[0][:, 4].tolist()  # confidence scores as a list
   class_ids = results.xyxy[0][:, 5].tolist()  # Extract the class IDs and names from YOLO results
   class_names = results.names
   classes = [class_names[int(i)] for i in class_ids]  # convert class IDs to class names
   image_index = df.index[df['Filename'] == image_file].tolist()[0]
   ground_truth_classes = str(df.loc[image_index, 'Ground Truth']).split(',') #Get the ground truth classes for this image
   df.at[image_index, 'Identified Class'] = ','.join(classes)
   classes = list(classes)
   classes = [c for c in classes if c in ground_truth_classes] # Remove any wrong classes from identified_classes
   if len(classes) > len(ground_truth_classes):
       comparision_metrics = 1
   else:
       comparision_metrics = len(classes) / len(ground_truth_classes) #Calculating the comparision metrics
   df.at[image_index, 'Confidence Score'] = ','.join(map(str, scores))
   df.at[image_index, 'Det Counts Metric'] = comparision_metrics
   bboxes = results.xyxy[0][:, :4].tolist()  # bounding boxes as a list of tuples (x1, y1, x2, y2)
   bbox_areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]  # calculating bounding box areas
   norm = [x/(image.shape[0]*image.shape[1]) for x in bbox_areas] #Normalising the bbox area
   df.at[image_index, 'BBox Area Metric'] = ','.join(map(str, norm))
df.to_csv(csv_path, index=False)
df = pd.read_csv(csv_path)