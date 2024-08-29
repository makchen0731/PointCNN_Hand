import os
import numpy as np
import sys
import struct
import torch
from torch.utils.data import Dataset
import open3d as o3d
import scipy.io as sio

from tqdm import tqdm
# dataset_dir = "/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/cvpr15_MSRAHandGestureDB"
# save_dir = "/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/MSRA_PointCloud"
# subject_list = ['P0','P1','P2','P3','P4','P5','P6','P7','P8']
# folder_list = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']


Joints = np.array([0,1,3,5, 6,7,9,11, 12,13,15,17, 18,19,21,23, 24,25,27,28, 32,30,31])
Eval = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 21, 22, 20])# 14 Joints


class NYU_Data_Loder(Dataset):
    def __init__(self, dataset_dir , mode ):
        
        
        
        self.dataset_dir = dataset_dir
        self.mode = mode
        
        if not self.mode in ['train', 'test']: raise ValueError('Invalid mode')
        
        self._load()
        
    
    def __getitem__(self, index):
        
        Points = o3d.io.read_point_cloud(self.file_path[index])
        Points = torch.tensor(Points.points)
        return Points,self.joints_world[index]  #self.file_path[index] self.Points_world[index]

    def __len__(self):
        return len(self.file_path)


    def _load(self):
        self.file_path = []
        
        if self.mode == 'train':
            joint_path = os.path.join(self.dataset_dir,self.mode,'joint_data.mat')
            joint_load = sio.loadmat(joint_path)
            # self.joints_world = joint_load['joint_xyz'][0][:, Joints, :][:, Eval, :] # 14 Joints 
            self.joints_world = joint_load['joint_xyz'][0][:, Joints, :][:, Eval, :] # 14 Joints  #Only 3000 Frames
            for idx in range(np.shape(self.joints_world)[0]):
                #Hand Point Cloud
                idx += 1 
                # file = os.path.join(self.dataset_dir, 'P'+str(mid) , fd,'{:0>6d}'.format(i-1)+'_Points.ply')
                file = os.path.join(self.dataset_dir, self.mode , 'synthdepth_1_{:0>7d}'.format(idx)+'_Points.ply')
                self.file_path.append(file)
        
        
        if self.mode == 'test':
            joint_path = os.path.join(self.dataset_dir,self.mode,'joint_data.mat')
            joint_load = sio.loadmat(joint_path)
            # self.joints_world = joint_load['joint_xyz'][0][1:, Joints, :][:, Eval, :] # 14 Joints  & image1 is broken
            self.joints_world = joint_load['joint_xyz'][0][1:, Joints, :][:, Eval, :] # 14 Joints  & image1 is broken
            for idx in range(np.shape(self.joints_world)[0]):
                #Hand Point Cloud
                idx += 1
                if idx == 1:
                    pass
                else:
                    # file = os.path.join(self.dataset_dir, 'P'+str(mid) , fd,'{:0>6d}'.format(i-1)+'_Points.ply')
                    file = os.path.join(self.dataset_dir, self.mode , 'synthdepth_1_{:0>7d}'.format(idx)+'_Points.ply')
                    self.file_path.append(file)
                    
                    
                    
if __name__ == '__main__':
    data_dir = '/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/NYU_Hand_PointCloud'
    train_set = NYU_Data_Loder(data_dir,'train')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1 , shuffle=True)
    for i,data in enumerate(tqdm(train_loader, 0)):
        Points , Joints  = data
        import open3d as o3d
    
        points = torch.clone(Points[0]).cpu().detach().numpy()
        joints = torch.clone(Joints[0]).cpu().detach().numpy()
        # Org_joints = torch.clone(Stand_jos[0]).cpu().detach().numpy()
        # axis_pcd = o3d.create_mesh_coordinate_frame(size=50, origin=org)
        
        
        colors = [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]]
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
            