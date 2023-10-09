#if importing images locallly
#in cmd go to the path where we have the images to be imported
# cd <the image_path>
#then type python -m http.server and press enter

import os
import json
import requests
import shutil
from PIL import Image


class LabelStudioImporter:
    def __init__(self):
        pass
    
    def create_project(self, label_studio_url, auth_token, project_name, project_description):
        headers = {
            "Authorization": f"Token {auth_token}",
            "Content-Type": "application/json"
        }

        # Create the project
        data = {
            "title": project_name,
            "description": project_description
        }

        response = requests.post(f"{label_studio_url}/api/projects/", headers=headers, data=json.dumps(data))

        if response.status_code == 201:
            self.project_id = response.json()["id"]
            print(f"New project created with ID: {self.project_id}")
        else:
            print(f"Failed to create project. Status code: {response.status_code}")

    def import_json_files(self, label_studio_url, auth_token, project_id, folder_path):
        headers = {
            'Authorization': f'Token {auth_token}'
        }

        # Get a list of JSON files in the folder
        json_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.json')]

        # Process each JSON file
        for json_file in json_files:
            json_file_path = os.path.join(folder_path, json_file)

            # Read the JSON data from the file
            with open(json_file_path, 'r') as file:
                json_data = json.load(file)

            # Create the payload for importing the   
            payload = {
                'data': json_data['data'],
                'annotations': json_data['annotations']
            }

            # Send a POST request to import the JSON data
            response = requests.post(f'{label_studio_url}/api/projects/{project_id}/import',
                                     headers=headers, json=payload)

            # Check the response status code
            if response.status_code == 201:
                print(f'Successfully imported {json_file} to project {project_id}')
            else:
                print(f'Failed to import {json_file}. Status code: {response.status_code}')

    def convert_to_json(self, labels_folder, server_url, output_folder, images_folder, classes_file):
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # Get a list of image files in the folder
        image_files = [file for file in os.listdir(images_folder) if file.lower().endswith(('.jpg', '.png', '.bmp', 'jpeg'))]

        # Process each image file
        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            with Image.open(image_path) as img:
                image_width, image_height = img.size

            txt_filename = os.path.splitext(image_file)[0] + ".txt"
            txt_path = os.path.join(labels_folder, txt_filename)
            json_path = os.path.join(output_folder, os.path.splitext(txt_filename)[0] + ".json")

            with open(txt_path, "r") as f:
                lines = f.readlines()

            annotations = []
            for idx, line in enumerate(lines):
                label, x, y, width, height = map(float, line.strip().split())

                x_min = (x - (width / 2)) * image_width
                y_min = (y - (height / 2)) * image_height
                box_width = width * image_width
                box_height = height * image_height
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(image_width, x_min + box_width)
                y_max = min(image_height, y_min + box_height)
                x_min_percentage = (x_min / image_width) * 100
                y_min_percentage = (y_min / image_height) * 100
                box_width_percentage = (box_width / image_width) * 100
                box_height_percentage = (box_height / image_height) * 100

                annotation = {
                    "id": f"result{idx + 1}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": image_width,
                    "original_height": image_height,
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": x_min_percentage,
                        "y": y_min_percentage,
                        "width": box_width_percentage,
                        "height": box_height_percentage,
                        "rectanglelabels": [classes[int(label)]]
                    }
                }
                annotations.append(annotation)

            image_url = os.path.join(server_url, image_file)

            data = {
                "data": {
                    "image": image_url
                },
                "annotations": [
                    {
                        "result": annotations
                    }
                ]
            }

            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)

            print(f"Converted {txt_filename} to JSON format")


# Set the necessary information
label_studio_url = 'http://localhost:8080'
server_url = 'http://localhost:8000/'
auth_token = 'f1b60b9ac1ebc0e53e9ed17672d2fbbc0bb9f8ef'
project_name = 'clover'
project_description = 'A new Label Studio project'

# Create an instance of the LabelStudioImporter class
importer = LabelStudioImporter()

# Create the project
importer.create_project(label_studio_url, auth_token, project_name, project_description)

# Set the folder paths and files
folder_path = r'D:\abcd\mer'
labels_folder = os.path.join(folder_path, 'labels')
output_folder = os.path.join(folder_path, 'labels_json')
images_folder = os.path.join(folder_path, 'images')
classes_file = os.path.join(folder_path, 'D:\Dropbox\DUK\classes.txt')

# Convert the files to JSON format
importer.convert_to_json(labels_folder, server_url, output_folder, images_folder, classes_file)

# Import the JSON files to the project
importer.import_json_files(label_studio_url, auth_token, importer.project_id, output_folder)
shutil.rmtree(output_folder)