
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import os
import open3d as o3d
import torch
import csv

import time


if __name__ == "__main__":
    
    
    load_path='/media/ntnu410/1T/NTNU/dataset/DepthOnly/ICVL_Hand/'
    center = np.loadtxt(os.path.join(load_path,'icvl_center_train.txt'))
    # img_path = []
    # joint_world = []
    
    # save label for ICVL
    with open(os.path.join(load_path,'train', "icvl_train_list.txt"), "r") as f:
        lines = f.readlines()
    train_path  = [line.strip() for line in lines if not line == "\n"]
    
    with open(os.path.join(load_path, "train", "labels.txt"), 'r') as f:
        joint_line = f.readlines()
    labels = [jline.split() for jline in joint_line if not jline == "\n"]
    
    with open('train_labels.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in train_path:
            for j in labels:
                if j[0] ==  i:
                    writer.writerow(j)
                    break
                    
    