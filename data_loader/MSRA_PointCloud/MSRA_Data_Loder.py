import os
import numpy as np
import sys
import struct
import torch
from torch.utils.data import Dataset
import open3d as o3d
import struct
from tqdm import tqdm
# dataset_dir = "/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/cvpr15_MSRAHandGestureDB"
# save_dir = "/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/MSRA_PointCloud"
# subject_list = ['P0','P1','P2','P3','P4','P5','P6','P7','P8']
# folder_list = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']


class MSRA_Data_Loder(Dataset):
    def __init__(self, dataset_dir , mode, test_subject_id ):
        # self.Points_num = 6114
        self.joint_num = 21
        self.world_dim = 3
        
        # self.subject_list = ['P0','P1','P2']
        # self.subject_list = ['P5','P6','P7']
        # self.subject_list = ['P5','P6']
        self.subject_list = ['P0','P1','P2','P3','P4','P5','P6','P7','P8']
        
        
        self.folder_list = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']
        # self.folder_list = ['9','I','IP','L','MP','RP']
        # self.folder_list = ['9','I','IP']
        # self.folder_list = ['9']
        # self.folder_list = ['1']
        self.subject_num = len(self.subject_list)
        
        self.dataset_dir = dataset_dir

        self.mode = mode
        self.test_subject_id = test_subject_id


        if not self.mode in ['train', 'test']: raise ValueError('Invalid mode')
        # assert self.test_subject_id >= 0 and self.test_subject_id < self.subject_num

        
        self._load()
        
    
    def __getitem__(self, index):
        
        Points = o3d.io.read_point_cloud(self.file_path[index])
        Points = torch.tensor(Points.points)
        return Points,self.joints_world[index]  #self.file_path[index] self.Points_world[index]

    def __len__(self):
        return self.num_samples


    def _load(self):
        self._compute_dataset_size()

        self.num_samples = self.train_size if self.mode == 'train' else self.test_size
        self.joints_world = np.zeros((self.num_samples, self.joint_num, self.world_dim))
        # self.Points_world  = np.zeros((self.num_samples, self.Points_num, self.world_dim))
        self.file_path = []


        #
        frame_id = 0
        
        # for mid in range(self.subject_num):
        for k,mid in enumerate(self.subject_list):
            if self.mode == 'train': model_chk = (mid != self.test_subject_id)
            elif self.mode == 'test': model_chk = (mid == self.test_subject_id)
            # if self.mode == 'train': model_chk = (len(mid) != self.test_subject_id)
            # elif self.mode == 'test': model_chk = (len(mid) == self.test_subject_id)
            
            else: raise RuntimeError('unsupported mode {}'.format(self.mode))

            if model_chk:
                for fd in self.folder_list:
                    # annot_file = os.path.join(self.dataset_dir, 'P'+str(mid), fd, 'joint.txt')
                    annot_file = os.path.join(self.dataset_dir, mid, fd, 'joint.txt')
                    # print(annot_file)
                    
                    
                    lines = []
                    with open(annot_file) as f:
                        lines = [line.rstrip() for line in f]

                    # skip first line
                    for i in range(1, len(lines)):
                        
                        # joint point
                        splitted = lines[i].split()
                        
                        for jid in range(self.joint_num):
                            self.joints_world[frame_id, jid, 0] = float(splitted[jid * self.world_dim])
                            self.joints_world[frame_id, jid, 1] = float(splitted[jid * self.world_dim + 1])
                            self.joints_world[frame_id, jid, 2] = -float(splitted[jid * self.world_dim + 2])
                        #Hand Point Cloud
                        # file = os.path.join(self.dataset_dir, 'P'+str(mid) , fd,'{:0>6d}'.format(i-1)+'_Points.ply')
                        file = os.path.join(self.dataset_dir, mid , fd,'{:0>6d}'.format(i-1)+'_Points.ply')
                        self.file_path.append(file)
                        
                        
                        # Pcl_file = os.path.join(self.dataset_dir, mid , fd,'{:0>6d}'.format(i-1)+'_Points.ply')
                        # print(Pcl_file)
                        # Points = o3d.io.read_point_cloud(Pcl_file)
                        # self.Points_world[frame_id,:,:] = np.expand_dims(np.array(Points.points),axis = 0)
                        # self.Points_world[frame_id,:,:] = np.array(Points.points)
                        frame_id += 1


    def _compute_dataset_size(self):
        self.train_size, self.test_size = 0, 0

        # for mid in range(self.subject_num):
        for mid in self.subject_list:
            num = 0
            for fd in self.folder_list:
                # annot_file = os.path.join(self.dataset_dir, 'P'+str(mid), fd, 'joint.txt')
                annot_file = os.path.join(self.dataset_dir, mid, fd, 'joint.txt')
                with open(annot_file) as f:
                    num = int(f.readline().rstrip())
                if mid == self.test_subject_id: self.test_size += num
                else: self.train_size += num
                
                
        # print(self.train_size)
    # def _check_exists(self):
    #     # Check basic data
    #     for mid in range(self.subject_list):
    #         for fd in self.folder_list:
    #             annot_file = os.path.join(self.dataset_dir, 'P'+str(mid), fd, 'joint.txt')
    #             if not os.path.exists(annot_file):
    #                 print('Error: annotation file {} does not exist'.format(annot_file))
    #                 return False

        # return True
if __name__ == '__main__':
    data_dir = '/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/MSRA_PointCloud'
    # train_set = MSRA_Data_Loder(data_dir,'train', 'P8')
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

    Val_set = MSRA_Data_Loder(data_dir, 'test','P0')
    Val_loader = torch.utils.data.DataLoader(Val_set, batch_size=1, shuffle=False)

    
    for i,data in enumerate(tqdm(Val_loader, 0)):
        Points , Joints  = data
        
        points = torch.clone(Points[0]).cpu().detach().numpy()
        joints = torch.clone(Joints[0]).cpu().detach().numpy()
        
        
        y = np.where(points == np.max(points))
        points = np.delete(points,y,axis = 0)
        print(np.where(points))
        
        
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
            