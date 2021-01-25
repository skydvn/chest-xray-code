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
    label_counts = data.class_name.value_counts() #label_counts is a list of all labels with its count
    # Get total number of samples
    total_samples = len(data)
    # Count the number of items in each class
    for i in range(len(label_counts)):
        label = label_counts.index[i]
        count = label_counts.values[i]
        percent = int((count / total_samples) * 10000) / 100
        nofinding_buffer = 0
        print("{:<30s}:   {} or {}%".format(label, count, percent))
        if label == "No finding": 
        	nofinding_buffer = percent

    return nofinding_buffer

def plot_histogram(dataframe, class_name, title, legend_title, xaxis_title, yaxis_title):
    fig = px.histogram(dataframe, x=class_name, color=class_name,
                       
                       ).update_xaxes(categoryorder="total descending")
    fig.update_layout(title=title,
                      legend_title=legend_title,
                      xaxis_title=xaxis_title,
                      yaxis_title=yaxis_title)
    fig.show()
    
train_dir = "D:\Outsourcing\Chest-Xray\VinBigdata/train"
dataframe = pd.read_csv(r"D:\Outsourcing\Chest-Xray\VinBigdata\train_vinbdid.csv")
dataframe.head()

#image = cv2.imread(r"D:\Outsourcing\Chest-Xray\VinBigdata\train\000ae00eb3942d27e0b97903dd563a6e.jpg")
#print(image)

class_names = dataframe["class_name"].unique().tolist()

print(class_names)
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


percent_distribution(dataframe)

#####################PLOT DISTRIBUTION OF OVERALL ABNORMALITIES################################
# plot_histogram(dataframe, 
#                class_name="class_name", 
#                title="<b>ABNORMALITIES COUNT PLOT</b>",
#                legend_title="<b>Class name</b>",
#                xaxis_title="<b>Abnormality Class Name</b>",
#                yaxis_title="<b>Number of samples</b>")

# plot_histogram(dataframe, 
#                class_name="rad_id", 
#                title="<b>ANNOTATIONS PER RADIOLOGIST</b>",
#                legend_title="<b>RADIOLOGIST ID</b>",
#                xaxis_title="<b>Radiologist ID</b>",
#                yaxis_title="<b>Number of Annotations Made</b>")
#########################################################################################################

fig = go.Figure()

#########################################################################################################
#########################################################################################################
#####################DEMONSTATE ALL RADIOLOGISTS 'S DIAGNOSE DISTRIBUTION################################
# init struct of radiologist can't diagnose anything 
rad_nofinding = []
for i in range(16):
	print("\nABNO DISTRIBUTION OF RADIOLOGIST {}".format(i+1))
	nofinding_buffer = percent_distribution(dataframe[dataframe["rad_id"]=="R{}".format(i+1)])
	print(nofinding_buffer)
	if nofinding_buffer == 100:		# if nofinding_buffer -> append value of radiologist to 
		rad_nofinding.append("R{}".format(i+1))
	print("\n")

print(rad_nofinding)

###################### SKIP ALL RADIOLOGISTS THAT CANT DIAGNOSE ANYTHING ################################
# rad_nofinding: the list of radiologists that cant diagnose anything


############################## Plot Abnormabilities Distribution ########################################
for i in range(15):
	if (all(x != "R{}".format(i+1) for x in rad_nofinding)):
	    fig = px.histogram(dataframe[dataframe["rad_id"]=="R{}".format(i+1)], 
	                       x="class_name",
	                       color="class_name",                
	                       ).update_xaxes(categoryorder="total descending")

	    # fig.update_xaxes(categoryorder="total descending")
	    fig.update_layout(title=f"<b>DISTRIBUTION OF RADIOLOGIST {i+1} </b>",
	                      barmode='stack',
	                      xaxis_title="<b>Radiologist ID</b>",
	                      yaxis_title="<b>Number of Annotations Made</b>")
	    fig.show()

#########################################################################################################
#########################################################################################################
#########################################################################################################

#########################################################################################################
################################# BBox Probabilitic Analysis ############################################

############################# X_min Y_min Probabilitic Analysis #########################################
# Loops over each Radiologist + All 3 Radiologists
	# Loops over each of abnormabilities
		# Expectation 
		# Variance
		# 3-D Distribution Graph || Heatmap
#########################################################################################################

############################# X_max Y_max Probabilitic Analysis #########################################
# Loops over each Radiologist + All 3 Radiologists
	# Loops over each of abnormabilities	
		# Expectation 
		# Variance
		# 3-D Distribution Graph || Heatmap
#########################################################################################################

#########################################################################################################

# heatmap_size = 1000
# heatmap = np.zeros((heatmap_size, heatmap_size, 14))
# resized_dataframe = dataframe.copy()

# for index, row in dataframe.iterrows():
#     image_path = os.path.join(train_dir, ''.join(row.image_id + '.jpg'))    
#     image = cv2.imread(image_path)
#     image_height, image_width = image.shape[:2]
#     labels = dataframe.loc[dataframe['image_id'] == row.image_id]
#     for _, label in labels.iterrows():
#         if label.class_id == 14:
#             continue
#         x_min = int(label.x_min*heatmap_size/image_width)
#         y_min = int(label.y_min*heatmap_size/image_height)
#         x_max = int(label.x_max*heatmap_size/image_width)
#         y_max = int(label.y_max*heatmap_size/image_height)
#         heatmap[y_min:y_max, x_min:x_max, label.class_id] += 1



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

############################ Show boundary box of the random image ####################################

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



