#Code to merge yolo result images to one folder, and also to write a csv file to store class detections in an image and to move images to folders for which the confidence score is high
import yolov5
import os
import csv
import shutil
from PIL import Image
from tqdm import tqdm

model_num = input("Enter the model version number: ")
model_path = rf'..\Trained models\v{model_num}\best.pt'
model = yolov5.load(model_path)
conf_score = float(input("Enter the confidence score: "))
# for folder_num in range (1,33):
    # images_folder = rf"D:\Documents\Ayruz\yolo models\FP test folders\FP test images {folder_num}"
    # output_folder = rf'D:\Documents\Ayruz\yolo models\FP test results\FP result {folder_num}'
    # output_file = f'{output_folder}\FP result {folder_num}.csv'
images_folder = r'..\Test images\images' 
output_folder = rf'..\Trained models\v{model_num}\output'
output_file = f'{output_folder}\Background.csv'

destination_folder = fr'{output_folder}\back'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
result_dir = os.path.join(output_folder, "results")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

header = ['file_name', 'confidence_score', 'detections','image']

with open(output_file, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)
    for image_file in tqdm(os.listdir(images_folder)):
        if os.path.exists(os.path.join(result_dir,image_file)):
            print("Skipping:",image_file)
            continue
        else:
            image_path = os.path.join(images_folder, image_file)
            image = Image.open(image_path).convert('RGB')
            results = model(image)
            output_path = os.path.join(result_dir, image_file)
            results.save(save_dir=output_path)
            detections = results.pandas().xyxy[0]
            scores = results.xyxy[0][:, 4].numpy()
            class_ids = results.xyxy[0][:, 5].tolist()  # Extract the class IDs and names from YOLO results
            class_names = results.names
            classes = [class_names[int(i)] for i in class_ids]  # convert class IDs to class names
            # get the number of detections
            num_detections = len(detections)
            for i in range(num_detections):
                # set the value of fp
                image_status = 'take' if scores[i] > conf_score else 'no'
                # get the ith detection
                detection = detections.iloc[i]
                class_name = classes[i]
                # write the results to the CSV file
                writer.writerow([image_file, scores[i], class_name, image_status])
# Loop through each row in the CSV file
with open(output_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Check if the image is taken
        if row['image'] == 'take':
            # Get the file name
            file_name = row['file_name']
            # Get the destination folder
            class_name = row['detections']
            class_folder = os.path.join(destination_folder, class_name)
            # Create the folder if it doesn't exist
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            # Copy the file to the destination folder
            shutil.copy(os.path.join(images_folder, file_name), os.path.join(destination_folder, class_name, os.path.basename(file_name)))
# Merge all the output images into a single folder
merged_output_folder = os.path.join(output_folder, 'model predicted output')
if not os.path.exists(merged_output_folder):
    os.makedirs(merged_output_folder)
for dir in os.listdir(result_dir):
    if os.path.isdir(os.path.join(result_dir, dir)):
        for file in os.listdir(os.path.join(result_dir, dir)):
            # Get the source file path
            src_file_path = os.path.join(result_dir, dir, file)
            # Get the destination file path
            dest_file_path = os.path.join(merged_output_folder, dir)
            # Rename the file to the folder name + original file name
            os.rename(src_file_path, dest_file_path)
        # Delete the folder
        try:
            os.rmdir(os.path.join(result_dir, dir))
        except OSError:
            print("Directory not empty.")
os.rmdir(result_dir)