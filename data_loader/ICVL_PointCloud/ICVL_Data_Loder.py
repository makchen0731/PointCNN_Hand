import os
import numpy as np
import sys
import struct
import torch
from torch.utils.data import Dataset
import open3d as o3d
import scipy.io as sio
import csv
from tqdm import tqdm


def pixel2world(x, y, z, img_width, img_height, fx, fy):
    
    w_x = (x - img_width / 2) / fx * z 
    w_y = (y - img_height / 2) / fy * z
    w_z = z
    return w_x, w_y, w_z



class ICVL_Data_Loder(Dataset):
    def __init__(self, dataset_dir , mode ):
        
        
        
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.joint_num = 16
        self.world_dim = 3


        self.w = 320
        self.h = 240
        self.fx = 241.42
        self.fy = 241.42
    
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
        
        
        
        # if self.mode == 'train':
        #     with open(os.path.join(self.dataset_dir,self.mode ,'train_labels.csv'), newline='') as csvfile: # open path and label
        #         path_and_label = list(csv.reader(csvfile))[9988:] #Path and joints
        #         # path_and_label = list(csv.reader(csvfile)) #Path and joints
                
        #     self.joints_world = np.delete(np.asarray(path_and_label),0,axis = 1)#.view((len(path_and_label), self.joint_num, self.world_dim))
        #     self.joints_world = self.joints_world.astype('float64').reshape((len(self.joints_world),self.joint_num, self.world_dim))
        #     self.joints_world[:,:,0],self.joints_world[:,:,1],self.joints_world[:,:,2] = pixel2world(self.joints_world[:,:,0],self.joints_world[:,:,1],self.joints_world[:,:,2],self.w,self.h,self.fx,self.fy)
        #     for idx,path_joint in enumerate(path_and_label):
        #         #Hand Point Cloud
        #         idx =idx + 9988
        #         # file = os.path.join(self.dataset_dir, 'P'+str(mid) , fd,'{:0>6d}'.format(i-1)+'_Points.ply')
        #         file = os.path.join(self.dataset_dir, self.mode , '{:0>6d}'.format(idx)+'_Points.ply')
        #         self.file_path.append(file)
        
        
        if self.mode == 'test':

            test_list = ['test_seq_1','test_seq_2']
            # test_list = ['test_seq_1']
            # test_list = ['test_seq_2']
            
            for test_path in test_list:
                # joints_world =  np.loadtxt(os.path.join(self.dataset_dir,self.mode,'{}.txt'.format(test_path)))
                
                

                    if test_path == 'test_seq_1':
                        with open(os.path.join(self.dataset_dir,self.mode,'{}.txt'.format(test_path)), "r") as f:
                            lines = f.readlines()
                            test_line = [line.split() for line in lines if not line == "\n"]
                            self.test_seq_1_joints = np.delete(np.asarray(test_line),0,axis = 1)
                            self.test_seq_1_joints = np.asarray(self.test_seq_1_joints, dtype = np.float64)
                            
                    
                        for idx in range(702):
                            #Hand Point Cloud
                            # idx += 1
                            # if idx == 1:
                            #     pass
                            # else:
                                # file = os.path.join(self.dataset_dir, 'P'+str(mid) , fd,'{:0>6d}'.format(i-1)+'_Points.ply')
                                file = os.path.join(self.dataset_dir, self.mode , test_path, '{:0>6d}'.format(idx)+'_Points.ply')
                                self.file_path.append(file)
                    else:
                        with open(os.path.join(self.dataset_dir,self.mode,'{}.txt'.format(test_path)), "r") as f:
                            lines = f.readlines()
                            test_line = [line.split() for line in lines if not line == "\n"]
                            self.test_seq_2_joints = np.delete(np.asarray(test_line),0,axis = 1)
                            self.test_seq_2_joints = np.asarray(self.test_seq_2_joints, dtype = np.float64)
                            
                    
                        for idx in range(894):
                            #Hand Point Cloud
                            # idx += 1
                            # if idx == 1:
                            #     pass
                            # else:
                                # file = os.path.join(self.dataset_dir, 'P'+str(mid) , fd,'{:0>6d}'.format(i-1)+'_Points.ply')
                                file = os.path.join(self.dataset_dir, self.mode , test_path, '{:0>6d}'.format(idx)+'_Points.ply')
                                self.file_path.append(file)


            self.joints_world = np.row_stack((self.test_seq_1_joints,self.test_seq_2_joints)).reshape((-1,16,3))
            self.joints_world[:,:,0],self.joints_world[:,:,1],self.joints_world[:,:,2] = pixel2world(self.joints_world[:,:,0],self.joints_world[:,:,1],self.joints_world[:,:,2],self.w,self.h,self.fx,self.fy)
if __name__ == '__main__':
    # data_dir = '/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/ICVL_PointCloud'
    data_dir = '/media/ntnu410/1T/NTNU/dataset/DepthOnly/ICVL_PointCloud'
    train_set = ICVL_Data_Loder(data_dir,'test')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1 , shuffle=True)
    # Val_set = ICVL_Data_Loder(data_dir, 'test')
    # Val_loader = torch.utils.data.DataLoader(Val_set, batch_size=1, shuffle=False)

    
    for i,data in enumerate(tqdm(train_loader, 0)):
        Points , Joints  = data
        
        points = torch.clone(Points[0]).cpu().detach().numpy()
        joints = torch.clone(Joints[0]).cpu().detach().numpy()
        
        
        # y = np.where(points[:,2] == 0)
        # points = np.delete(points,y,axis = 0)
        # print(np.where(points))
        
        
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
        
        
        colors = [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]]
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
            