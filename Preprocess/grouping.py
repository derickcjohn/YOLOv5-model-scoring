import shutil
import os
from tqdm import tqdm
#path to the directory where the project is located, change the path here
project_dir = r'..\project'
#path to the directory where the output needs to be saved, change the path here
output_dir = '..\output'
#Path to the class names txt file
original_file = '..\classes.txt'

original_lines = []

with open(original_file, 'r') as f:
    original_lines = f.readlines()
    
class_map={}
    
for i,item in enumerate(original_lines):
    class_map[i]=item.rstrip('\n')
    

if not os.path.exists(os.path.join(project_dir,output_dir)):
    os.makedirs(os.path.join(project_dir,output_dir))

for filename in tqdm(os.listdir(os.path.join(project_dir,'images'))):
    label_file = os.path.join(project_dir,'labels',os.path.splitext(filename)[0]+'.txt')
    if not os.path.exists(label_file):
        print(f'Skipping {filename} because label file not found')
        continue
    with open(label_file) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()] #strip white spaces and ignore empty lines
        if len(lines) == 0: # skip empty label text files
            print(f'Skipping {filename} because label file is empty')
            continue
        class_names = list(set([class_map[int(line.split()[0])] for line in lines]))
        if len(class_names)>1:
            if not os.path.exists(os.path.join(project_dir,output_dir,'miscellaneous')):
                os.makedirs(os.path.join(project_dir,output_dir,'miscellaneous'))
                os.makedirs(os.path.join(project_dir,output_dir,'miscellaneous','images'))
                os.makedirs(os.path.join(project_dir,output_dir,'miscellaneous','labels'))
            shutil.copy2(os.path.join(project_dir,'images',filename),os.path.join(project_dir,output_dir,'miscellaneous','images',filename))
            shutil.copy2(os.path.join(project_dir,'labels',os.path.splitext(filename)[0]+'.txt'),os.path.join(project_dir,output_dir,'miscellaneous','labels',os.path.splitext(filename)[0]+'.txt'))
    
        else:
            class_path = os.path.join(project_dir,output_dir,class_names[0])
            if not os.path.exists(class_path):
                os.makedirs(class_path)
                os.makedirs(os.path.join(class_path,'images'))
                os.makedirs(os.path.join(class_path,'labels'))
            if os.path.exists(os.path.join(project_dir,'images',filename)):
                shutil.copy2(os.path.join(project_dir,'images',filename),os.path.join(class_path,'images'))
                shutil.copy2(os.path.join(project_dir,'labels',os.path.splitext(filename)[0]+'.txt'),os.path.join(class_path,'labels'))
            else:
                print(f"Image file {filename} not found")