# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 07:52:04 2021

@author: mduon
"""

"""
IMPORT
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os 
import cv2

def dataset_analysis():
    """
    Define
    """
    abnormalities_vinbdi = ["Aortic enlargement","Atelectasis","Calcification","Cardiomegaly",\
                            "Consolidation","ILD","Infiltration","Lung Opacity","Nodule/Mass",\
                            "Other lesion","Pleural effusion","Pleural thickening","Pneumothorax",\
                            "Pulmonary fibrosis","No finding"]

    """
    Defines IO path
    """
    folder_vinbdi = "D:\Outsourcing\Chest-Xray\VinBigdata"
    print(folder_vinbdi)
    folder_chex   = "D:\Outsourcing\Chest-Xray\CheXpert-v1.0-small"

    """
    Read CSV file
    """
    path = os.path.join(folder_vinbdi, "train_{}.csv".format("vinbdid"))
    df = pd.read_csv(path) #("D:\Outsourcing\Chest-Xray\VinBigdata/train_{}.csv".format("vinbdi"))
    df.reset_index(inplace = True)
    #df.set
    """
    Data Analyze
    """
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    abno_vinbdi_distribution =  np.zeros(15)
    # Counts label for each data
    for abno_cnt in range(len(abnormalities_vinbdi)):
        seriesObj = df.apply(lambda x: True if x["class_name"] == abnormalities_vinbdi[abno_cnt] else False , axis=1)
        abno_vinbdi_distribution[abno_cnt] = len(df[seriesObj == True].index)
    print(abno_vinbdi_distribution)
    # Plot distribution graph of labels 
    x = np.arange(len(abno_vinbdi_distribution))
    width = 0.8 

    fig, ax = plt.subplots()


    for i in range(len(abno_vinbdi_distribution)): 
        rects = ax.bar(1 + width/2 + 2*i*width/2, abno_vinbdi_distribution[i], width, label=abnormalities_vinbdi[i])
        autolabel(rects)
        
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Samples")
    ax.set_title("Dataset 's distribution")
    ax.set_xticks(x)
    #ax.set_xticklabels(abnormalities_vinbdi)
    ax.legend()
       
    fig.tight_layout()
    plt.show()
    
    # Get image size
    im = cv2.imread("D:\Outsourcing/Chest-Xray/VinBigdata/train/000ae00eb3942d27e0b97903dd563a6e.jpg")
    h, w, c = im.shape
    plot = [""] * 15
    # Loops for tickers in Dataframe  
    for abno_cnt in range(len(abnormalities_vinbdi)):
        rslt_df = df[df["class_name"] == abnormalities_vinbdi[abno_cnt]]
        plot[abno_cnt] = plt.figure(abno_cnt)   
        plt.plot(rslt_df["x_min"], rslt_df["y_min"], 'o', color='red');
        plt.plot(rslt_df["x_max"], rslt_df["y_max"], 'o', color='blue');
        plt.title("{}".format(abnormalities_vinbdi[abno_cnt]))
        plt.xlim([0, w])
        plt.ylim([0, h])

    # Fetch bbox coordinator of data
    plt.show()
    
    # Plot distribution of bbox for each label

dataset_analysis()    
