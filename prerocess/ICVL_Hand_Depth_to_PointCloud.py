
import numpy as np
import csv


import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import os
import open3d as o3d
import torch
import csv

import time

# def uvd2xyz(data,fx,fy):
#     """
#         Convert data from uvd to xyz
#     """
#     x = data.copy()
#     if len(x.shape) == 3:
#         x[:, :, 0] = (x[:, :, 0] - 160) / fx * x[:, :, 2]
#         x[:, :, 1] = (x[:, :, 1] - 120) / fy * x[:, :, 2]

#     return x

def pixel2world(x, y, z, img_width, img_height, fx, fy):
    
    w_x = (x - img_width / 2) / fx * z 
    w_y = (y - img_height / 2) / fy * z
    w_z = z
    return w_x, w_y, w_z


def depthmap2points(image, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, w, h, fx, fy)
    
    return points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # device = xyz.device
    N, C = xyz.shape
    centroids = torch.zeros(npoint, dtype=torch.long)
    distance = torch.ones( N) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long)

    for i in range(npoint):

        centroids[i] = farthest

        centroid = xyz[farthest, :].view(1, 3)
        
        dist = torch.sum((xyz - centroid) ** 2, -1)
        dist = dist.type(torch.FloatTensor)
        distance = np.squeeze(distance,0)
        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = torch.max(distance, -1)[1]
    return centroids


def Hand_croper(cent,img,fx,fy,cent_w,cent_h):
    cube_size = 125

    du = (cube_size - 30) / cent[2] * fx
    dv = (cube_size - 30) / cent[2] * fy
    left = int(cent[0] - du)
    right = int(cent[0] + du)
    top = int(cent[1] - dv)
    buttom = int(cent[1] + dv)
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, cent_w * 2)
    buttom = min(buttom, cent_h * 2)
    MM = np.zeros_like(img)
    MM[top:buttom, left:right] = 1
    img = img * MM
    
    MM = np.logical_and(img < cent[2] + cube_size, img > cent[2] - cube_size)
    img = img * MM
    
    return img

if __name__ == "__main__":
    
    
    load_path='/media/ntnu410/1T/NTNU/dataset/DepthOnly/ICVL_Hand/'
    save_dir = "/media/ntnu410/1T/NTNU/dataset/DepthOnly/ICVL_PointCloud/"
    train_test  = ['train']
    # train_test  = ['test']
    
    half_u = 160
    half_v= 120
    # test_list = ['test_seq_1','test_seq_2']
    w = 320 
    h = 240
    
    fx = 241.42
    fy = 241.42
    

    # img_path = []
    # joint_world = []
    
    
    # #Train data
    # for mode in train_test:
    #     if mode == 'train':
    #         center = np.loadtxt(os.path.join(load_path,'icvl_center_train.txt'))
    #         with open(os.path.join(load_path,mode,'train_labels.csv'), newline='') as csvfile: # open path and label
    #             # path_and_label = list(csv.reader(csvfile))[20530:]
    #             path_and_label = list(csv.reader(csvfile))
    #         for i,path_labels in enumerate(path_and_label):
    #             # i += 20530
    #             if i > 22060:
    #                 break
    #             img_path = os.path.join(load_path,mode,'Depth',path_labels[0])
    #             # img_path.append(joints[0])
    #             image = plt.imread(img_path) * 65535
    #             image = Hand_croper(center[i],image,fx,fy,half_u,half_v)
                
    #             Points = depthmap2points(image,fx,fy).reshape((320*240,3))
                
    #             x = np.where(Points[:,2] == 0)
    #             Points = np.delete(Points,x,axis = 0)
                
    #             # if Points.size == 0:
    #             #     print(img_path)

    #             # joints = np.asarray(path_labels[1:],dtype=float).reshape((16,3))
    #             # joints[:,0],joints[:,1],joints[:,2] = pixel2world(joints[:,0],joints[:,1],joints[:,2],w,h,fx,fy)
                    
    #             fps = farthest_point_sample(torch.from_numpy(Points),6144).numpy()
    #             Points = Points[fps,:]
            
    #             ##---Save To Point Cloud---###
    #             Save_Path = os.path.join(save_dir,mode)
                
    #             if not os.path.exists(Save_Path):
    #                 os.makedirs(Save_Path)
                    
    #             pcd = o3d.geometry.PointCloud()
    #             pcd.points = o3d.utility.Vector3dVector(Points)
    #             o3d.io.write_point_cloud(Save_Path+'/{:0>6d}'.format(i)+'_Points.ply', pcd)
                
    #             # # Try_Points = o3d.io.read_point_cloud(Save_Path+'/{:0>6d}'.format(i)+'_Points.ply')
    #             # # Try_Points = torch.tensor(Try_Points.points)
                
    #             # colors = [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]]
    #             # # colors = [[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]
    #             # # colors2 = [[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]
    #             # # colors = [[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0]]
    #             # test_pcd = o3d.geometry.PointCloud()
    #             # # test_pcd_Joint = o3d.geometry.PointCloud()
    #             # # test_Est_World_Joints = o3d.geometry.PointCloud()
    #             # vis = o3d.visualization.Visualizer()
                
    #             # # vis.add_geometry(axis_pcd)
    #             # vis.add_geometry(test_pcd)
    #             # # vis.add_geometry(test_pcd_Joint)
    #             # # vis.add_geometry(test_Est_World_Joints)
                
            
    #             # test_pcd.points = o3d.utility.Vector3dVector(Points)
                
    #             # # test_pcd_Joint.colors = o3d.utility.Vector3dVector(colors)
    #             # # test_pcd_Joint.points = o3d.utility.Vector3dVector(joints)
    #             #     # test_Est_World_Joints.points = o3d.utility.Vector3dVector(Org_joints)
                
                 
    #             #     # test_Est_World_Joints.colors = o3d.utility.Vector3dVector(colors2) 
                    
    #             # o3d.visualization.draw_geometries([test_pcd] , window_name="Open3D1")
            
    # #     else:
    # #Test data
    # mode = 'test'
    for test in test_list:
        # test = 'test_seq_1'
        center = np.loadtxt(os.path.join(load_path,'icvl_center_test.txt'))
        with open(os.path.join(load_path, mode, test+'.txt'), "r") as f:
            lines = f.readlines()
        test_path = [line.split() for line in lines if not line == "\n"]
        j = 702
        for i,path_labels in enumerate(test_path):
            if test == 'test_seq_2': i += j
            # else: j = i
            img_path = os.path.join(load_path,mode,'Depth',path_labels[0])
            image = plt.imread(img_path) * 65535
            image = Hand_croper(center[i],image,fx,fy,half_u,half_v)
            
            Points = depthmap2points(image,fx,fy).reshape((320*240,3))
            x = np.where(Points[:,2] == 0)
            Points = np.delete(Points,x,axis = 0)
            # y = np.where(Points == np.max(Points))
            # Points = np.delete(Points,y,axis = 0)
            
            joints = np.asarray(path_labels[1:],dtype=float).reshape((16,3))
            joints[:,0],joints[:,1],joints[:,2] = pixel2world(joints[:,0],joints[:,1],joints[:,2],w,h,fx,fy)
                
            fps = farthest_point_sample(torch.from_numpy(Points),6144).numpy()
            Points = Points[fps,:]
        
            ##---Save To Point Cloud---###
            Save_Path = os.path.join(save_dir,mode,test)
            
            if not os.path.exists(Save_Path):
                os.makedirs(Save_Path)
                
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(Points)
            if test == 'test_seq_2': 
                i -= j
                o3d.io.write_point_cloud(Save_Path+'/{:0>6d}'.format(i)+'_Points.ply', pcd)
                    else:
                        o3d.io.write_point_cloud(Save_Path+'/{:0>6d}'.format(i)+'_Points.ply', pcd)
            


