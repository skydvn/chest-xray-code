# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 20:00:31 2021

@author: mduon
"""

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

grid_size = (3, 3)
def show_images(images, title=None):
    fig = plt.figure(figsize=(15., 15.))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=grid_size,
                    axes_pad=0.1,
                    )
    for ax, image in zip(grid, images):
        show_image(image, ax)
    if title is not None:
        plt.suptitle(title, fontweight="bold", fontsize=16)
    plt.axis(False)
    plt.show()
    

def show_image(image, ax=None):
    if ax is None:
        plt.figure(figsize=(15., 15.))        
        ax = plt
    ax.axis(False)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
    else:
        ax.imshow(image, cmap='gray')
        

colors = np.array([
                    [1, 0, 1],
                    [0, 0, 1],
                    [0, 1, 1],
                    [0, 1, 0],
                    [1, 1, 0],
                    [1, 0, 0] 
                   ])

def get_color(c, offset, max_num):
    ratio = offset/max_num*5
    i = math.floor(ratio)
    j = math.ceil(ratio)
    ratio -= i
    r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
    return r
    
def draw_bbox_and_label(image, x_min, y_min, x_max, y_max, class_id, label, number_of_class):
    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)
    height, width = image.shape[:2]
    offset = class_id * 123457 % number_of_class;    
    color = (get_color(2, offset, number_of_class)*255, get_color(1, offset, number_of_class)*255, get_color(0, offset, number_of_class)*255)
    font_size = height/1000
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size, int(2*font_size))
    box_width = x_max - x_min
    if box_width < text_size[0][0]:
        text_bg_x_max = text_size[0][0] + x_min
    else:
        text_bg_x_max = box_width + x_min
        
    cv2.rectangle(image, (x_min, y_min - int(3 + 18 * font_size)), (text_bg_x_max, y_min), color, -1)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color)
    image = cv2.putText(image, label, (x_min, y_min - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,  font_size, (0, 0, 0), int(2*font_size), cv2.LINE_AA)
    return image

def percent_distribution(data):
    # Get the count for each label
    label_counts = data.class_name.value_counts()

    # Get total number of samples
    total_samples = len(data)

    # Count the number of items in each class
    for i in range(len(label_counts)):
        label = label_counts.index[i]
        count = label_counts.values[i]
        percent = int((count / total_samples) * 10000) / 100
        print("{:<30s}:   {} or {}%".format(label, count, percent))

def plot_histogram(dataframe, class_name, title, legend_title, xaxis_title, yaxis_title):
    fig = px.histogram(dataframe, x=class_name, color=class_name,
                       
                       ).update_xaxes(categoryorder="total descending")
    fig.update_layout(title=title,
                      legend_title=legend_title,
                      xaxis_title=xaxis_title,
                      yaxis_title=yaxis_title)
    fig.show()

def class_column_concat(dataframe, dict):
    mydata = pd.DataFrame([None], columns=['class_total'])
    for label_index in dict:
            df = pd.DataFrame("{}".format(dict[label_index]), index=range(int(dataframe["{}".format(dict[label_index])].sum())), columns = ["class_total"])
            print(df)
            mydata=pd.concat([mydata,df],axis=0)

    #del mydata["class_total"]
    mydata=mydata.dropna()
    return mydata

def nih_imagefolder_switch(argument):
    switcher = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }
    print switcher.get(argument, "Invalid month")

#train_dir = "./train"
#dataframe = pd.read_csv("./dataset-standard.csv")
train_dir = "D:\Outsourcing/Chest-Xray/NIH"
dataframe = pd.read_csv("D:\Outsourcing/Chest-Xray/NIH/dataset-CHX.csv")
dataframe_bbox = pd.read_csv("D:\Outsourcing/Chest-Xray/NIH/dataset-bbox-CHX.csv")
#image = cv2.imread(r"D:\Outsourcing\Chest-Xray\VinBigdata\train\000ae00eb3942d27e0b97903dd563a6e.jpg")
#print(image)

class_names = dataframe["Finding Labels"].unique().tolist()

class_dict = {
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
    13: "Pulmonary fibrosis",    
    14: "No finding"
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
	13: "Hernia",
    14: "No finding"
	}

image_map = {
    (0, 1335): 0.97, 
    (1336, 3923): 0.15,
    (3923, 6585): 0.97, 
    (6585, 9232): 0.15, #  #004
    (0, 1335): 0.97, 
    (1336, 3923): 0.15,        }
    (0, 1335): 0.97, 
    (1336, 3923): 0.15,
    (0, 1335): 0.97, 
    (1336, 3923): 0.15,    
}

# for index, row in dataframe.iterrows():	
# 	for label_index in NIH_dict:
# 		if (row["Finding Labels"].find(NIH_dict[label_index]) != -1):
# 			dataframe.loc[index,"{}".format(NIH_dict[label_index])] = 1
# 		else:
# 			dataframe.loc[index,"{}".format(NIH_dict[label_index])] = 0
# 	if index == 200: 
# 		break

# mydata = class_column_concat(dataframe, NIH_dict)
# print(mydata)
# # percent_distribution(dataframe)

# plot_histogram(mydata, 
#                class_name="class_total", 
#                title="<b>ABNORMALITIES COUNT PLOT</b>",
#                legend_title="<b>Class name</b>",
#                xaxis_title="<b>Abnormality Class Name</b>",
#                yaxis_title="<b>Number of samples</b>")


# plot_histogram(dataframe, 
#                class_name="rad_id", 
#                title="<b>ANNOTATIONS PER PATIENT</b>",
#                legend_title="<b>RADIOLOGIST ID</b>",
#                xaxis_title="<b>Radiologist ID</b>",
#                yaxis_title="<b>Number of Annotations Made</b>")


# fig = go.Figure()


# for i in range(15):
#     fig.add_trace(go.Histogram(
#                                 x=dataframe[dataframe["class_id"]==i]["rad_id"],
#                                 name=f"<b>{class_dict[i]}</b>")
#                  )

# fig.update_xaxes(categoryorder="total descending")
# fig.update_layout(title="<b>DISTRIBUTION OF CLASS LABEL ANNOTATIONS BY RADIOLOGIST</b>",
#                   barmode='stack',
#                   xaxis_title="<b>Radiologist ID</b>",
#                   yaxis_title="<b>Number of Annotations Made</b>")

# fig.show()


heatmap_size = 1000
heatmap = np.zeros((heatmap_size, heatmap_size, 14))
resized_dataframe = dataframe_bbox.copy()


for index, row in dataframe_bbox.iterrows():
    image_index = int(row["image_id"][:row["image_id"].find("_")])
    if 
    # image_path = os.path.join(train_dir, ''.join(row.image_id + '.jpg'))    
    # image = cv2.imread(image_path)
    # image_height, image_width = image.shape[:2]
    # labels = dataframe.loc[dataframe['image_id'] == row.image_id]
    # for _, label in labels.iterrows():
    #     if label.class_id == 14:
    #         continue
    #     x_min = int(label.x*heatmap_size/image_width)
    #     y_min = int(label.y*heatmap_size/image_height)
    #     width = int(label.w*heatmap_size/image_width)
    #     height = int(label.h*heatmap_size/image_height)
    #     heatmap[y_min:(y_min+height), x_min:(x_min+width), label.class_id] += 1



# heatmap_list = [heatmap[:, :, i] for i in range(heatmap.shape[2])]
# fig = plt.figure(figsize=(15., 15.))
# grid = ImageGrid(fig, 111,
#                 nrows_ncols=(4, 4),
#                 axes_pad=0.5,
#                 )

# for i, ax in enumerate(grid):
#     if i < len(heatmap_list):
#         ax.imshow(heatmap_list[i], cmap='gray')
#         ax.set_title(class_dict[i], fontweight="bold", fontsize=16)
#     ax.axis(False)
# plt.suptitle("Heatmaps Showing Bounding Box Placement", fontweight="bold", fontsize=24)
# plt.axis(False)
# plt.show()



# random_samples = dataframe.sample(n=grid_size[0]*grid_size[1]).iterrows()
# print(random_samples)
# images = []
# for index, row in random_samples:
#     image_path = os.path.join(train_dir, ''.join(row.image_id + '.jpg'))    
#     #print(image_path)
#     #print(os.path.isfile(image_path))
#     image = cv2.imread(r"{}".format(image_path))
#     print(image)
#     labels = dataframe.loc[dataframe['image_id'] == row.image_id]
#     for _, label in labels.iterrows():
#         if label.class_id == 14:
#             continue
#         draw_bbox_and_label(image, label.x_min, label.y_min, label.x_max, label.y_max, label.class_id, label.class_name + ' - ' + label.rad_id, len(class_names))
#     #image = cv2.resize(image, (1000, 1000))
#     images.append(image)
# show_images(images, f"{grid_size[0]*grid_size[1]} random sample")
# # show_images(images)



