import pandas as pd
import numpy as np
import math
from subprocess import call

user_input = input("Does csv file already exist for model?[y/n] ")

if user_input.lower() == "n":
    call(["python", "Comparison metrics.py"])

class Model:
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)

    def calculate_average_confidence_score(self):
        scores = []
        for score_list in self.df['Confidence Score']:
            if pd.isna(score_list):
                continue
            try:
                scores.append(float(score_list))
            except ValueError:
                scores += [float(score) for score in score_list.split(',')]
        scores=[x for x in scores if not math.isnan(x)]
        if len(scores) == 0:
            return 0.0
        else:
            return sum(scores) / len(scores)*100
        
    def calculate_average_detection_count(self):
        detection_sum = 0.0
        gt_count = 0
        for index, row in self.df.iterrows():
            det_metric = row['Det Counts Metric']
            if pd.isna(det_metric):
                continue
            detection_sum += float(det_metric)
            ground_truth = row['Ground Truth']
            if not pd.isna(det_metric):
                gt_count += 1
        det=(detection_sum/gt_count)*100
        return det
    
    def evaluate(self):
        self.df['Ground Truth Classes'] = self.df['Ground Truth'].apply(lambda x: x.split(',') if isinstance(x, str) else [])   
        self.df['Ground Truth Areas'] = self.df['Ground Truth BBox Area'].apply(lambda x: [float(i) for i in x.split(',')] if isinstance(x, str) else [])
        self.df['Predicted Classes'] = self.df['Identified Class'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
        self.df['Predicted Areas'] = self.df['BBox Area Metric'].apply(lambda x: [float(i) for i in x.split(',')] if isinstance(x, str) else [])
        total_objects = 0
        accurate_objects = 0
        for i,row in self.df.iterrows():
            matched_pred_boxes = []
            gt_classes = row['Ground Truth Classes']
            gt_areas = row['Ground Truth Areas']
            pred_classes = row['Predicted Classes']
            pred_areas = row['Predicted Areas']
            total_objects += len(pred_classes)
            count_class = []
            for j,pred_class in enumerate(pred_classes):
                if pred_class not in gt_classes:
                    continue
                pred_area = pred_areas[j] if j < len(pred_areas) else 0.0
                gt_indexes = [idx for idx, gt_class in enumerate(gt_classes) if gt_class == pred_class and idx not in count_class]
                for gt_index in gt_indexes:
                    gt_area = gt_areas[gt_index]
                    if gt_area in matched_pred_boxes:
                        continue
                    ratio = pred_area / gt_area
                    if ratio>= 0.9 and ratio <= 1.1:
                        accurate_objects += 1
                        matched_pred_boxes.append(gt_area)
                        count_class.append(gt_index)
        accuracy_percentage = (accurate_objects / total_objects) * 100 if total_objects > 0 else 0.0
        return accuracy_percentage
    
    def f1_score(self):
        self.df['Ground Truth Classes'] = self.df['Ground Truth'].apply(lambda x: x.split(',') if isinstance(x, str) else [])   
        self.df['Predicted Classes'] = self.df['Identified Class'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
        tp = 0  # true positives
        fp = 0  # false positives
        fn = 0  # false negatives
        for i, row in self.df.iterrows():
            gt_classes = row['Ground Truth Classes']
            pred_classes = row['Predicted Classes']
            fp_found=[]
            tp_found=[]
            fn_found=[]
            for gt_class in gt_classes:
                if gt_class not in pred_classes:
                    fn=fn+1
                else:
                    x = gt_classes.count(gt_class)
                    y = pred_classes.count(gt_class)
                    if x==y:
                        tp=tp+1
                    elif x>y and gt_class not in tp_found and gt_class not in fn_found:
                        tp=tp+y
                        fn=fn+(x-y)
                        tp_found.append(gt_class)
                        fn_found.append(gt_class)
                    elif y>x:
                        if gt_class not in tp_found and gt_class not in fp_found:
                            tp=tp+x
                            fp += (y-x)
                            tp_found.append(gt_class)
                            fp_found.append(gt_class)
            for pred_class in pred_classes:
                if pred_class not in gt_classes and pred_class not in fp_found:
                    fp += 1
                    fp_found.append(pred_class)
        #print(tp,fp,fn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        #print(precision,recall)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return f1*100


model_name = input("Enter the model version number: ")
path = f'..\Tested models result csv\Test Result x{model_name} (v5).csv'
model = Model(path)

conf_score = model.calculate_average_confidence_score()
print(f"The average Confidence Score is: {conf_score:.2f}")

average_detection = model.calculate_average_detection_count()
print(f"The average of Count of detection is: {average_detection:.2f}")

bbox_area_metric = model.evaluate()
print(f"The bbox area metric is: {bbox_area_metric:.2f}")

#model scoring 
f1_score = model.f1_score()
print(f"The F1 score is: {f1_score:.2f}")
weights = [0.7, 0.1, 0, 0.2]
weighted_score = (conf_score * weights[3] +
                  average_detection * weights[2] +
                  bbox_area_metric * weights[1] +
                  f1_score * weights[0])
print("Total score of the model:",weighted_score)