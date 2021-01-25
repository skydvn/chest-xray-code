import os
import numpy as np
import pandas as pd
import random
import cv2
import math
import glob


# Create list of radiologist with their stats / statistic (all the abnormabilities counter) #
############ Create list + structure of data ############ 
# init radio list:
radio_list = []				
for i in range(17):
	radio_list.append("R{}".format(i+1))
column_dict = {
	"Rad_id": radio_list,
    "Aortic enlargement": [0]*len(radio_list),
    "Atelectasis": [0]*len(radio_list),
    "Calcification": [0]*len(radio_list),
    "Cardiomegaly": [0]*len(radio_list),
    "Consolidation": [0]*len(radio_list),
    "ILD": [0]*len(radio_list),
    "Infiltration": [0]*len(radio_list),
    "Lung Opacity": [0]*len(radio_list),
    "Nodule/Mass": [0]*len(radio_list),
    "Other lesion": [0]*len(radio_list),
    "Pleural effusion": [0]*len(radio_list),
    "Pleural thickening": [0]*len(radio_list),
    "Pneumothorax": [0]*len(radio_list),
    "Pulmonary fibrosis": [0]*len(radio_list),    
    "No finding": [0]*len(radio_list),
    "Total": [0]*len(radio_list)
}
print(column_dict)
df_radio_abno = pd.DataFrame.from_dict(column_dict)
print(df_radio_abno)

def radiologist_statistic(data, j):
	# Get the count for each label
	temp_df = pd.DataFrame(columns = column_dict)
	label_counts = data.class_name.value_counts() 
	# Get total number of samples
	total_samples = len(data)
	dist_sum = 0
	# Count the number of items in each class
	for i in range(len(label_counts)):
		label = label_counts.index[i]
		count = label_counts.values[i]
		percent = int((count / total_samples) * 10000) / 100
		# sum 
		dist_sum += count
		# change data in dataframe
		df_radio_abno.loc[j,label] = percent
	df_radio_abno.loc[j,"Total"] = dist_sum

# Read dataset csv 
train_dir = "D:\Outsourcing\Chest-Xray\VinBigdata/train"
test_dir  = "D:\Outsourcing\Chest-Xray\VinBigdata/test"
dataframe = pd.read_csv(r"D:\Outsourcing\Chest-Xray\VinBigdata\train_vinbdid.csv")
fix_dataframe = dataframe

# for j in range(17):
# 	print(j)
# 	radiologist_statistic(dataframe[dataframe["rad_id"] == "R{}".format(j+1)], j)
# 	# Append temp_list into dataframe
# print(df_radio_abno)
# df_radio_abno.to_csv(r"../VinBigdata/df_radio_abno.csv")
df_radio_abno = pd.read_csv(r"../VinBigdata/df_radio_abno.csv")
print(df_radio_abno)

# print(dataframe[dataframe["image_id"] == "000ae00eb3942d27e0b97903dd563a6e"])
# Each image -> Get data of that image (all statistics) -> compare with radiologist_statistic 
# Suppress all the boundary box in 1 image / save boundary box of the highest <Major voting>

# Loops image in folder train // test
for root, dirs, files in os.walk(train_dir, topdown=False):
	for file_name in files:
		# Check image name in dataframe
		img_name = file_name[0 : file_name.find(".jpg")]
		img_dataframe = dataframe[dataframe["image_id"] == img_name]
		# count class_name
		img_class_count = img_dataframe.class_name.value_counts() 
		# loops over class_name (each abnormability)
		for img_class_name in img_class_count.index:
			# get radiologist name
			img_class_dataframe  = img_dataframe[img_dataframe["class_name"] == img_class_name]
			img_class_rad_counts = img_dataframe[img_dataframe["class_name"] == img_class_name].rad_id.value_counts() 
			if len(img_class_rad_counts.index) != 1: 
				# Loops over radiologist 
				temp_dict = {}
				# push all needed data into list
				for img_class_rad_id in img_class_rad_counts.index:
					# Get that radiologist 's statistic of that abnormability from <<df_radio_abno>>
					# Got <img_class_rad_i: [Rx Ry ... Rz]> of <img_class_name>
					# 				 Row							Column
					table_index = int(img_class_rad_id[1:])-1
					temp_value = df_radio_abno.loc[table_index,"Total"]*df_radio_abno.loc[table_index,img_class_name]/100
					temp_dict.update({img_class_rad_id: temp_value})
				max_value = max(temp_dict.values())  # maximum value
				# Get Major Voting Key
				max_keys = [k for k, v in temp_dict.items() if v == max_value]
				# Save the radiologist with highest statistic -> dropout all other radiologist 
				print(max_keys[0])
				for img_class_rad_id in img_class_rad_counts.index:
					if  ((img_class_rad_id != max_keys[0]) & (img_class_name != "No finding")):
						#fix_dataframe = fix_dataframe.drop(img_class_rad_id, axis = 0)
						a = fix_dataframe.loc[(fix_dataframe["class_name"] == img_class_name) &
									    	  (fix_dataframe["image_id"] == img_name) &
										      (fix_dataframe["rad_id"] == img_class_rad_id)]
						print(a.index[0])
						print(len(fix_dataframe))
						print(a)
						print("before")						
						print(fix_dataframe[fix_dataframe["image_id"] == img_name])
						fix_dataframe = fix_dataframe.drop(index = a.index[0])	
						fix_dataframe.reset_index(drop= True)


						print("after")
						print(fix_dataframe[fix_dataframe["image_id"] == img_name])
				print("\n")			
fix_dataframe.to_csv("../VinBigdata/train_b.csv")


# 		# count class_name
# 		img_class_count = img_dataframe.class_name.value_counts() 
# 		# loops over class_name (each abnormability)
# 		for img_class_i in img_class_count.index:
# 			# get radiologist name
# 			img_class_rad_counts = img_dataframe[img_dataframe["class_id"] == img_class_i].rad_id.value_counts() 
# 			# If number of radiologists != 1 --> Loops // else just pass to check another abnormability 's class
# 			if len(img_class_rad_counts.index) != 1: 
# 				# Loops over radiologist 
# 				for img_class_rad_i in img_class_rad_counts.index:
# 					# Get that radiologist 's statistic of that abnormability 
# 					print(img_class_rad_i)
# 					# Save the radiologist with highest statistic -> dropout all other radiologist 




# Suppress all the boundary box in 1 image with percentage of abnormabilities < 1%

# Images with many high profile radiologists -> split into copies + separated boundary box. 



# Check image name in dataframe

# img_dataframe = dataframe[dataframe["image_id"] == "b325d5dcf507c8cce2c5de3e9afb2847"]
# print(fix_dataframe[fix_dataframe["image_id"] == "b325d5dcf507c8cce2c5de3e9afb2847"])
# # count class_name
# img_class_count = img_dataframe.class_name.value_counts() 
# # loops over class_name (each abnormability)
# for img_class_name in img_class_count.index:
# 	# get radiologist name
# 	img_class_dataframe  = img_dataframe[img_dataframe["class_name"] == img_class_name]
# 	img_class_rad_counts = img_dataframe[img_dataframe["class_name"] == img_class_name].rad_id.value_counts() 
# 	if len(img_class_rad_counts.index) != 1: 
# 		# Loops over radiologist 
# 		temp_dict = {}
# 		# push all needed data into list
# 		for img_class_rad_id in img_class_rad_counts.index:
# 			# Get that radiologist 's statistic of that abnormability from <<df_radio_abno>>
# 			# Got <img_class_rad_i: [Rx Ry ... Rz]> of <img_class_name>
# 			# 				 Row							Column
# 			table_index = int(img_class_rad_id[1:])-1
# 			temp_value = df_radio_abno.loc[table_index,"Total"]*df_radio_abno.loc[table_index,img_class_name]/100
# 			temp_dict.update({img_class_rad_id: temp_value})
# 		max_value = max(temp_dict.values())  # maximum value
# 		# Get Major Voting Key
# 		max_keys = [k for k, v in temp_dict.items() if v == max_value]
# 		# Save the radiologist with highest statistic -> dropout all other radiologist 

# 		for img_class_rad_id in img_class_rad_counts.index:
# 			if img_class_rad_id != max_keys[0]:
# 				print(img_class_rad_id)
# 				print(max_keys[0])
# 				#fix_dataframe = fix_dataframe.drop(img_class_rad_id, axis = 0)
# 				a = fix_dataframe.loc[(fix_dataframe["class_name"] == img_class_name) &
# 							    	  (fix_dataframe["image_id"] == "b325d5dcf507c8cce2c5de3e9afb2847") &
# 								      (fix_dataframe["rad_id"] == img_class_rad_id)]
# 				fix_dataframe = fix_dataframe.drop(index = a.index)				      

