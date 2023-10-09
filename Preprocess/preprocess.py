import shutil
import json
import os
from sklearn.model_selection import train_test_split

#modify the paths as needed
logo_dir = 'logo'
new_brands_dir = r'\path\to\New Brands'
label_dict = {'0': '11', '1': '4', '2': '1', '3': '2', '4': '3', '5': '6', '6': '7', '7': '8', '8': '9', '9': '10', '10': '19', '11': '15', '12': '16', '13': '13', '14': '17', '15': '18', '16': '20', '17': '25', '18': '5', '19': '22', '20': '23', '21': '28', '22': '24', '23': '29', '24': '30', '25': '14', '26': '26', '27': '27', '28': '0', '29': '12'}
merged_dir = r'path\to\merged'
class_names = [0,1,2]
folders_to_merge = ['path/to/folder1','path/to/folder2','path/to/folder3']
json_file = "/path/to/data.json"

class Preprocess:
    def __init__(self):
        pass
        
#Removes image files if their corresponding label files are not present and vice versa
    def remove_label(self, logo_folder):
        image_folder = os.path.join(logo_folder,'images')
        label_folder = os.path.join(logo_folder,'labels')
        image_files = os.listdir(image_folder)
        labels_files = os.listdir(label_folder)
        
        for image_file in image_files:
            image_basename, image_ext = os.path.splitext(image_file) #Splitting the image file names to get the corresponding label file
            matching_labels_files = [f for f in labels_files if os.path.splitext(f)[0] == (image_basename)] #Getting the label files whose image files are present
            if not matching_labels_files: #Checking if the label file is present or not
                image_path = os.path.join(image_folder, image_file)
                os.remove(image_path) #Removing the image file for which the label is not present
                print(f"Removed image file {image_file} because its corresponding label file was not found.")
            else:
                label_path = os.path.join(label_folder, matching_labels_files[0])
                if os.path.isfile(label_path) and os.path.getsize(label_path) == 0:
                    os.remove(os.path.join(image_folder, image_file))
                    os.remove(label_path)
                    print(f"Removed image file {image_file} because its corresponding label file is empty.")
        
        for labels_file in labels_files:
            labels_basename, labels_ext = os.path.splitext(labels_file) #Splitting the label file names to get the corresponding image file
            matching_image_files = [f for f in image_files if os.path.splitext(f)[0] == labels_basename] #Getting the image files whose label files are present
            if not matching_image_files: #Checking if the image file is present or not
                labels_path = os.path.join(label_folder, labels_file)
                os.remove(labels_path) #Removing the label file for which the image is not present
                print(f"Removed label file {labels_file} because its corresponding image file was not found.")

#Replacing the labelled classes based on the mapping given by a dictionary
    def remap_class(self, logo_folder, label_dict): #The dictionary contains old labels as keys and new labels as values
        labels_folder = os.path.join(logo_folder,'labels')
        for filename in os.listdir(labels_folder):
            if filename.endswith(".txt"):
                with open(os.path.join(labels_folder, filename), "r+") as file: #Giving both read and write access to the label file
                    lines = file.readlines()
                    file.seek(0) #Offsetting the position to the beginning of the file
                    for line in lines:
                        items = line.strip().split()
                        if len(items) > 0: # Check if the line has at least one item
                            old_label = items[0] #The labels are present at the first index
                            if old_label in label_dict:
                                new_label = label_dict[old_label]
                                items[0] = str(new_label) #Replacing old label value with new
                            new_line = " ".join(items)
                            file.write(new_line + "\n")
                    file.truncate() #Rewritting the values in the file
                file.close() #Close the file after reading and writing to it

#Method to split the dataset for training and validation
    def split_data(self, new_brands_folder, val_size=0.25,random_state=42):
        #Iterating over all the subfolders in "New Brands"
        for subfolder in os.listdir(new_brands_folder):
            subfolder_path = os.path.join(new_brands_folder, subfolder)
            image_folder = os.path.join(subfolder_path, 'images')
            labels_folder = os.path.join(subfolder_path, 'labels')
            file_names = os.listdir(image_folder)
        #Creating train and validation folders within each subfolder if they don't exist
            train_folder = os.path.join(subfolder_path, 'train')
            val_folder = os.path.join(subfolder_path, 'val')
            if not os.path.exists(train_folder):
                os.makedirs(train_folder)
                os.makedirs(os.path.join(train_folder, 'images'))
                os.makedirs(os.path.join(train_folder, 'labels'))
            if not os.path.exists(val_folder):
                os.makedirs(val_folder)
                os.makedirs(os.path.join(val_folder, 'images'))
                os.makedirs(os.path.join(val_folder, 'labels'))
        
            #Splitting the dataset to train and validation
            train_images, val_images = train_test_split(list(file_names), test_size=val_size, random_state=random_state)

            #Copying the dataset of images and its corresponding labels to newly created Train, Test and Validation folders
            for basename in train_images:            
                shutil.copy2(os.path.join(image_folder, basename), os.path.join(train_folder, 'images', basename))
                shutil.copy2(os.path.join(labels_folder, os.path.splitext(basename)[0]+'.txt'), os.path.join(train_folder, 'labels', os.path.splitext(basename)[0]+'.txt'))
            for basename in val_images: 
                shutil.copy2(os.path.join(image_folder, basename), os.path.join(val_folder, 'images', basename))
                shutil.copy2(os.path.join(labels_folder, os.path.splitext(basename)[0]+'.txt'), os.path.join(val_folder, 'labels', os.path.splitext(basename)[0]+'.txt'))
        
#Method to merge multiple training folders into one
    def merge_training_data(self, training_path, folder_names):
        if not os.path.exists(training_path):
            os.makedirs(training_path) #Creating a new training directory
        for folder_name in folder_names: #Iterating through each split folders
            subfolders = [name for name in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, name))] #Accessing the image and label folders
            for subfolder in subfolders:
                source = os.path.join(folder_name, subfolder)
                destination = os.path.join(training_path, subfolder) #Giving the path to folder where the files needs to be merged
                if not os.path.exists(destination):
                    os.makedirs(destination)
                files = os.listdir(source)
                for file in files:
                    shutil.copy2(os.path.join(source, file), destination) #Moving the files to the destination folder
          
#Returns a list of filenames from folder that have at least one bounding box with a class label in class_combination
    def fetch_data_class_combination(self, logo_folder, class_combination):
        labels_folder = os.path.join(logo_folder,'labels')
        file_list = []
        files = os.listdir(labels_folder)
        for file in files:
            with open(os.path.join(labels_folder, file), "r") as f: #Giving only read access to the label file
                lines = f.readlines()
                for line in lines:
                    items = line.split()
                    class_name = int(items[0])
                    if class_name in class_combination: #Checking if the label is present in the class combination
                        file_list.append(file) #Appending the names of the files that have a class combination
                        break
        return file_list

#Generates a JSON file with filenames grouped by class labels
    def generate_json(self, logo_folder, class_names):
        labels_folder = os.path.join(logo_folder,'labels')
        data ={}
        for class_name in class_names:
            file_list = self.fetch_data_class_combination(labels_folder, [class_name]) #Returns a list of files with the given class label
            data[str(class_name)] = file_list #Stores the labels and the files that contain it in the form of dictionary
        json_data = json.dumps(data, indent=4) #Dumping the values in json format
        return json_data

preprocess = Preprocess()
preprocess.remove_label(f"{new_brands_dir}\\{logo_dir}")
preprocess.remap_class(f"{new_brands_dir}\\{logo_dir}", label_dict)
preprocess.split_data(new_brands_dir)
preprocess.merge_training_data(merged_dir,folders_to_merge)
print(preprocess.fetch_data_class_combination(f"{new_brands_dir}\\{logo_dir}",class_names))
json_data = preprocess.generate_json(f"{new_brands_dir}\\{logo_dir}", class_names)
with open(json_file, "w") as f: #Saving the returned json data in data.json
    f.write(json_data)