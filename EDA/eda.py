#Importing essential libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import datetime
import yolov5


class YOLOtextfiles:
    def __init__(self):
        # Constructor initializes the object with a folder path
        pass
    #Method to count the total class counts    
    def get_class_counts(self,folder_path):
        # Create an empty dictionary to store the class counts
        class_counts = {}
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    # Get the path to the label_file_path
                    label_file_path = os.path.join(root, file)
                    with open(label_file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            class_label = line.strip().split()[0]
                            if class_label not in class_counts:
                                class_counts[class_label] = 1
                            else:
                                class_counts[class_label] += 1
        # Return the dictionary containing the class counts
        return class_counts
        
    
    #method to find the count of the number of unique class labels in each images
    def count_instances(self,folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    print(file)
                    # Create an empty list to store the class labels
                    count_instance=[]
                    label_file_path = os.path.join(root, file)
                    with open(label_file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            class_label = int(line.strip().split()[0])
                            if class_label in count_instance:
                                continue
                            else:
                                count_instance.append(class_label)
                        print("Class Instances per Image:",count_instance) 


    # method to find groups of objects in the same image that belong to different classes.
    def count_grouped_instances(self,folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    # Create an empty list to store the class labels for each group
                    group_instance=[]
                    label_file_path = os.path.join(root, file)
                    with open(label_file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            class_label = int(line.strip().split()[0])                        
                            if class_label in group_instance:
                                continue
                            else:
                                group_instance.append(class_label)
                            # If there is more than one unique class label in the group_instance list, print the group to the console
                            if len(group_instance)>1:
                                print(root.split("\\")[4],"--",file,"--  Grouped instances:",group_instance)
                

    #method to generate a graph to get the Class counts
    def plot_class_counts(self,folder_path):
        # Get a dictionary of class labels and their counts in the dataset.
        class_counts = self.get_class_counts(location)
        labels=np.array(list(class_counts.keys()))
        values=np.array(list(class_counts.values()))
        sort_idx=np.argsort(values)[::-1]
        labels=labels[sort_idx]
        values=values[sort_idx]
        plt.bar(range(len(labels)),values)
        plt.xticks(range(len(labels)),labels)
        plt.xlabel("Class Labels")
        plt.ylabel("Count of labels")
        plt.title("Class  vs Counts")
        plt.show()
    

    #method to generate a graph to get the train/test/validation counts
    def train_test_val_count(self,folder_path):
        # Create an empty list to store the names of all the text files
        textfiles=[]        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                
                if file.endswith(".txt"):
                    textfiles.append(file)        
        train=(len(textfiles)*0.8)
        test=(len(textfiles)*0.10)
        val=(len(textfiles)*0.10)        
        labels = ['Training', 'Validation', 'Test']
        counts = [train, val, test]        
        fig, ax = plt.subplots()
        ax.bar(labels, counts)
        ax.set_xlabel('Train/test/val')
        ax.set_ylabel('Count')
        ax.set_title('Train/test/val counts')
        plt.show()
        print("Training dataset:",int(train))
        print("Testing dataset:",int(test))
        print("Validation dataset:",int(val))
    


    # Define a function to plot object positions colored by size                    
    def scatter_plot(self,folder_path):
        # Initialize empty lists to store data points
        sizes = []
        x_coords = []
        y_coords = []
        class_labels = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    label_file_path = os.path.join(root, file)
                    with open(label_file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            # Extract data from each line of the text file
                            class_label, x, y, w, h = line.strip().split()
                            x = float(x)
                            y = float(y)
                            w = float(w)
                            h = float(h)
                            sizes.append(w*h)
                            x_coords.append(x)
                            y_coords.append(y)
                            class_labels.append(class_label)       
        fig, ax = plt.subplots()
        scatter = ax.scatter(x_coords, y_coords, c=sizes, cmap='viridis')      
        legend = ax.legend(*scatter.legend_elements(),
                            loc="lower left", title="Sizes")
        ax.add_artist(legend)      
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Object Positions Colored by Size')
        plt.show()

    # Method to plot the folder names with the count of images and labels (stacked bar graph)  
    def plot_folder_count(self,folder_path):
        folder_counts = {}
        for subdir in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, subdir)
            if os.path.isdir(subdir_path):
                images_count = len(os.listdir(os.path.join(subdir_path, 'images')))
                labels_count = len(os.listdir(os.path.join(subdir_path, 'labels')))
                folder_counts[subdir] = {'images': images_count, 'labels': labels_count}
        folder_names = list(folder_counts.keys())
        image_counts = [folder_counts[k]['images'] for k in folder_names]
        label_counts = [folder_counts[k]['labels'] for k in folder_names]
        plt.figure(figsize=(10,10))
        plt.bar(folder_names, image_counts, label='Images')
        plt.bar(folder_names, label_counts, bottom=image_counts, label='Labels')
        plt.xticks(rotation=90,fontsize=14)
        plt.xlabel('Folder Names',fontsize=14)
        plt.ylabel('Counts',fontsize=14)
        plt.title('Counts of Images and Labels for each Folder')
        plt.legend()
        plt.show()
        return folder_counts                

    #stacked bar graph to find the brand and its variants from each folder
    def stacked_graph_brand_variants(self,folder_path):    
        class_counts = {}
        for root, dirs, files in os.walk(folder_path):
            # If 'labels' folder is found in the current directory
            if 'labels' in dirs:
                # Get the brand name from the parent directory
                directory_name = os.path.basename(root)
                brand_counts = {}
                label_dir_path = os.path.join(root, 'labels')
                for file in os.listdir(label_dir_path):
                    if file.endswith(".txt"):
                        label_file_path = os.path.join(label_dir_path, file)
                        with open(label_file_path, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                class_label = line.strip().split()[0]
                                if class_label not in brand_counts:
                                    brand_counts[class_label] = 1
                                # If the class label is already in the dictionary, increment its count by 1
                                else:
                                    brand_counts[class_label] += 1
                 
                # Add the brand counts to the class_counts dictionary
                class_counts[directory_name] = brand_counts
                 
        for folder in class_counts:
            classes = list(class_counts[folder].keys())    
            counts = list(class_counts[folder].values())        
        # Return the dictionary containing the class counts
            # create the stacked bar graph
            plt.bar(folder, counts[0], label=classes[0])
            for i in range(1, len(classes)):
                plt.bar(folder, counts[i], bottom=sum(counts[:i]), label=classes[i])
            plt.xlabel('Class Labels')
            plt.ylabel('Counts')
            plt.title('Stacked Bar Graph for ' + folder)
            plt.legend()
            plt.show()


    #method to find Average bbox area for each class
    def average_bboxarea_for_each_class(self,folder_path):
        # Create a dictionary to hold the total bbox area and count for each class label
        bbox_area_counts = {}
        for root, dirs, files in os.walk(folder_path):
            for file in files:                
                if file.endswith(".txt"):                   
                    label_file_path = os.path.join(root, file)
                    with open(label_file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:                        
                            class_label,x,y,w,h = line.strip().split()                          
                            if class_label not in bbox_area_counts:
                                bbox_area_counts[class_label] = {'total_area': 0, 'count': 0}                           
                            bbox_area = (float(w) - float(x)) * (float(h) - float(y))
                            # Add the bbox area and count to the dictionary
                            bbox_area_counts[class_label]['total_area'] += bbox_area
                            bbox_area_counts[class_label]['count'] += 1        
        for class_label in bbox_area_counts:
            avg_area = bbox_area_counts[class_label]['total_area'] / bbox_area_counts[class_label]['count']            
            print('Average bbox area for class', class_label, ':', avg_area)
    
    
    #A method to find outliers (based on bbox sizes)
    def outliers(self,folder_path):
        areas=[]
        outliers = []
        outlier_labels = {} 
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    label_file_path = os.path.join(root, file)
                    with open(label_file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:                           
                            class_label, x, y, w, h = map(float, line.strip().split())
                            class_label=int(class_label)                            
                            # Calculate the bbox area using the x, y, w, h values
                            bbox_area = (float(w) - float(x)) * (float(h) - float(y))
                            areas.append(bbox_area)
                            q1 = np.percentile(areas, 25)
                            q3 = np.percentile(areas, 75)
                            iqr = q3 - q1   
                            lower_bound = q1 - (1.5 * iqr)
                            upper_bound = q3 + (1.5 * iqr)        
                            if bbox_area < lower_bound or bbox_area > upper_bound:
                                outliers.append((file, bbox_area,int(class_label)))
                                if class_label not in outlier_labels:
                                    outlier_labels[class_label] = 1
                                else:
                                    outlier_labels[class_label] += 1
        
        print(f"Total number of bounding boxes: {len(areas)}")
        print(f"Number of outliers: {len(outliers)}")                               
        for outlier in outliers:
            print(f"File name: {outlier[0]}, outlier value: {outlier[1]},class label:{outlier[2]}")                      
        for label, count in outlier_labels.items():
            print(f"Class label: {label}, outlier count: {count}")
    
    
    def speed_test(self,model_version,test_path,num_images=None):
        image_files = os.listdir(test_path)
        if num_images is not None:
            image_files = image_files[:num_images]          
       
        model = yolov5.load(F'../Trained models/v{model_version}/best.pt')       
        start_time = datetime.datetime.now()
        
        for image_file in tqdm(image_files):
            image_path = os.path.join(test_path, image_file)
            # image = cv2.imread(image_path,) 
            results = model(image_path)  
            
        end_time = datetime.datetime.now()
        time_taken = (end_time - start_time).total_seconds()
        ips = len(image_files) / time_taken
        return ips
    
    def find_parent_folder(self,parent_dir):
        class_labels = {0: 'globe_life', 1: 'clover', 2: 'chime_1', 3: 'chime_2', 4: 'chime_3', 5: 'fiserv_1', 6: 'fiserv_2', 7: 'fiserv_3', 8: 'fiserv_4', 9: 'fiserv_5', 10: 'motorola_razr_1', 11: 'motorola_1', 12: 'motorola_2', 13: 'moto_g_1', 14: 'motorola_edge_1', 15: 'motorola_edge_2', 16: 'motorola_razr_2', 17: 'opendoor_1', 18: 'clover_2', 19: 'nexen_tyre_1', 20: 'nexen_tyre_2', 21: 'skechers_1', 22: 'nexen_tyre_3', 23: 'skechers_2', 24: 'skechers_3', 25: 'moto_microphone_1', 26: 'progressive_1', 27: 'progressive_2', 28: 'Jersey_mikes_subs_1', 29: 'jersey_mikes_short_1', 30: 'new_jersey_devils_1', 31: 'tax_act_1'}
        class_folders = {class_name: [] for class_name in class_labels.values()}
        # Loop over the folders in the parent directory
        for folder_name in os.listdir(parent_dir):
            folder_path = os.path.join(parent_dir, folder_name)
            if os.path.isdir(folder_path):
                images_path = os.path.join(folder_path, "images")
                for image_file in os.listdir(images_path):
                    image_name = os.path.splitext(image_file)[0]
                    label_path = os.path.join(folder_path, "labels", image_name + ".txt")
                    if os.path.exists(label_path):
                        with open(label_path, "r") as f:
                            lines = f.readlines()
                            for line in lines:
                                class_label, x, y, w, h = map(float, line.strip().split())
                                class_label=int(class_label)
                                # Map the class label to its corresponding name
                                class_name = class_labels.get(class_label)
                                # Append the folder name to the list associated with the class name
                                if class_name and folder_name not in class_folders[class_name]:
                                    class_folders[class_name].append(folder_name)                   
        return class_folders
                             
# #a) Total class counts
# location=r'\path\to\New Brands'
# yolo = YOLOtextfiles()
# class_counts = yolo.get_class_counts(location)
# print(class_counts)

# #b)Class instances per image
# #which returns a dictionary where each key represents a class label and the value represents the number of instances of that class 
# class_instance = yolo.count_instances(location)


# #c) Class groupped occurances
# #returns the number of instances of objects in the image that have been grouped together
# grouped_instances = yolo.count_grouped_instances(location)


# #d) Generate a graph to depict labelled classes
#         #1) Class counts
# #display the count plot showing total class count with respect to corresponding labels
# plot_count = yolo.plot_class_counts(location)

#         #2) Train/test/val counts
# #display the count plot for total count of train test validation data
# ttv_count = yolo.train_test_val_count(location)

#         #3) EDA
# #plotting scatter plot for eda
# scatter=yolo.scatter_plot(location)

# #plotting stacked bar graph to show each foldera and their count of images and labels
# plot_folder_count=yolo.plot_folder_count(location)

# #plotting stacked bar graph to find the brands and its variants
# yolo.stacked_graph_brand_variants(location)

# #e) Average bbox area for each class
# #returns the average bbox area for each class label
# average_area = yolo.average_bboxarea_for_each_class(location)


# #f) A method to find outliers (based on bbox sizes)
# #displays the name of the file, the label, and the outliers
# outliers = yolo.outliers(location)


# #finding images per second
# model_version=r'D:\ayruz\duk-2023\Trained models\exp84-5s\best.pt'
# test_path=r"D:\ayruz\duk-2023\fp test images"
# num_images = 10000
# model=yolo.speed_test(model_version,test_path,num_images)
# print("Iterations per second:",model)

#method to get the parent folders of different classes
parent_dir = r"/path/to/your/folder"
yolo=YOLOtextfiles()
yolo.find_parent_folder(parent_dir)
