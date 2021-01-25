import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import cv2
import math

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

train_dir = "D:\Outsourcing/Chest-Xray/NIH"
dataframe = pd.read_csv(r"D:\Outsourcing/Chest-Xray/NIH/Data_Entry_2017.csv")
dataframe_bbox = pd.read_csv(r"D:\Outsourcing/Chest-Xray/NIH/BBox_List_2017.csv")

#print(dataframe.head())
class_names = dataframe["Finding Labels"].head(20).unique().tolist()
#print(class_names) #.unique())

dataset_list = {
	0: "BDI",
	1: "NIH",
	1: "CHX"
}
NIH_dict = {
	0: "Atelectasis",
	1: "Cardiomegaly",
	2: "Effusion",
	3: "Infiltration",
	4: "Mass",
	5: "Nodule",
	6: "Pneumonia",
	7: "Pneumothorax",
	8: "Consolidation",
	9: "Edema",
	10: "Emphysema",
	11: "Fibrosis",
	12: "Pleural_Thickening",
	13: "Hernia"
	}

NIH_bbox_dict = {
	0: "Atelectasis",
	1: "Cardiomegaly",
	2: "Effusion",
	3: "Infiltrate",
	4: "Mass",
	5: "Nodule",
	6: "Pneumonia",
	7: "Pneumothorax",
	8: "Consolidation",
	9: "Edema",
	10: "Emphysema",
	11: "Fibrosis",
	12: "Pleural_Thickening",
	13: "Hernia"	
}
BDI_dict = {
    0: "Aortic enlargement",
    1: "Atelectasis",
    2: "Calcification",
    3: "Cardiomegaly",
    4: "Consolidation",
    5: "ILD",
    6: "Infiltration",
    7: "Lung Opacity",
    8: "Nodule/Mass",
    9: "Other lesion",
    10: "Pleural effusion",
    11: "Pleural thickening",
    12: "Pneumothorax",
    13: "Pulmonary fibrosis"
	}

CHX_dict = {
	0: "Enlarged Cardiomediastinum",
	1: "Cardiomegaly",
	2: "Lung Opacity",
	3: "Lung Lesion",
	4: "Edema",
	5: "Consolidation",
	6: "Pneumonia",
	7: "Atelectasis",
	8: "Pneumothorax",
	9: "Pleural Effusion",
	10: "Pleural Other",
	11: "Fracture",
	12: "Support Devices",
	}

# initialize matrix of NIH abnormabilities:
sub_df = pd.DataFrame(np.zeros((dataframe["Finding Labels"].count(), 14)),columns=["Atelectasis",
																				"Cardiomegaly",
																				"Effusion",
																				"Infiltration",
																				"Mass",
																				"Nodule",
																				"Pneumonia",
																				"Pneumothorax",
																				"Consolidation",
																				"Edema",
																				"Emphysema",
																				"Fibrosis",
																				"Pleural_Thickening",
																				"Hernia"
																				])


"""
dataframe = pd.concat([dataframe,sub_df],axis=1)


for index, row in dataframe.iterrows():	
	k = 0
	if (index % 2000 == 0): print(index)
	for label_index in NIH_dict:		
		if (row["Finding Labels"].find(NIH_dict[label_index]) != -1):
			dataframe.loc[index,"{}".format(NIH_dict[label_index])] = 1
			k = 1
		else:
			dataframe.loc[index,"{}".format(NIH_dict[label_index])] = 0
	if k == 0: 
		dataframe.loc[index,"No finding"] = 1
	else: 
		dataframe.loc[index,"No finding"] = 0
#	if (index == 200): break

print(dataframe)
dataframe.to_csv("D:\Outsourcing/Chest-Xray/NIH/dataset-{}.csv".format(dataset_list[1]))
"""

"""
Process the BBox_List_2017.csv:
	+ 
"""
for index, row in dataframe_bbox.iterrows():	
	k = 0
	for label_index in NIH_dict:
		if (row["Finding Label"].find(NIH_bbox_dict[label_index]) != -1):
			dataframe_bbox.loc[index,"class_id"] = label_index
			k = 1
	if k == 0: 
		dataframe_bbox.loc[index,"class_id"] = 14

dataframe_bbox.to_csv("D:\Outsourcing/Chest-Xray/NIH/dataset-bbox-{}.csv".format(dataset_list[1]))