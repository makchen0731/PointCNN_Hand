import os
import numpy as np
import sys
import struct
import torch
from torch.utils.data import Dataset
import open3d as o3d
import scipy.io as sio

from tqdm import tqdm


Joints = np.array([0,1,3,5, 6,7,9,11, 12,13,15,17, 18,19,21,23, 24,25,27,28, 32,30,31])
Eval = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 21, 22, 20])# 14 Joints



                    
if __name__ == '__main__':
        data_dir = '/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/NYU_Hand_PointCloud'
        
        
        Points = o3d.io.read_point_cloud('/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/NYU_Hand_PointCloud/test/synthdepth_1_0006931_Points.ply')
        points = np.asarray(Points.points)
        
        joint_load = sio.loadmat('/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/NYU_Hand_PointCloud/test/joint_data.mat')
        
        joints = joint_load['joint_xyz'][0][6931, Joints, :][Eval, :]

        # points = torch.clone(Points[0]).cpu().detach().numpy()
        # joints = torch.clone(Joints[0]).cpu().detach().numpy()
        
        # data_dir = '/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/NYU_Hand_PointCloud'
        # train_set = NYU_Data_Loder(data_dir,'train')
        # train_loader = torch.utils.data.DataLoader(train_set, batch_size=1 , shuffle=False)
        
        # Val_set = NYU_Data_Loder(data_dir, 'test')
        # Val_loader = torch.utils.data.DataLoader(Val_set, batch_size=1, shuffle=False)
        # for i,data in enumerate(tqdm(Val_loader, 0)):
            
        # file = os.path.join()
        # points = o3d.io.read_point_cloud('/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/NYU_Hand_PointCloud/test/synthdepth_1_0002800_Points.ply')
        # points = np.asarray(points.points)
        
        # joint_path = os.path.join('/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/NYU_Hand_PointCloud/test/joint_data.mat')
        # joint_load = sio.loadmat(joint_path)
        # joints  = joint_load['joint_xyz'][0][2800, Joints, :][Eval, :]
            
    
        ##---open 3d plot---###
        import open3d as o3d
    
        # points = torch.clone(Points[0]).cpu().detach().numpy()
        # joints = torch.clone(Joints[0]).cpu().detach().numpy()
        # Org_joints = torch.clone(Stand_jos[0]).cpu().detach().numpy()
        # axis_pcd = o3d.create_mesh_coordinate_frame(size=50, origin=org)
        
        
        colors = [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]]
        # colors = [[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]
        # colors2 = [[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]
        # colors = [[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0]]
        test_pcd = o3d.geometry.PointCloud()
        test_pcd_Joint = o3d.geometry.PointCloud()
        test_Est_World_Joints = o3d.geometry.PointCloud()
        vis = o3d.visualization.Visualizer()
        
        # vis.add_geometry(axis_pcd)
        vis.add_geometry(test_pcd)
        vis.add_geometry(test_pcd_Joint)
        # vis.add_geometry(test_Est_World_Joints)
        
    
        test_pcd.points = o3d.utility.Vector3dVector(points)
        
        test_pcd_Joint.colors = o3d.utility.Vector3dVector(colors)
        test_pcd_Joint.points = o3d.utility.Vector3dVector(joints)
            # test_Est_World_Joints.points = o3d.utility.Vector3dVector(Org_joints)
        
         
            # test_Est_World_Joints.colors = o3d.utility.Vector3dVector(colors2) 
            
        o3d.visualization.draw_geometries([test_pcd] + [test_pcd_Joint] , window_name="Open3D1")
            