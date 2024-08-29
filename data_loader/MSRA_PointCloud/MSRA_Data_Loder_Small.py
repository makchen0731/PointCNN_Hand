import os
import numpy as np
import sys
import struct
from torch.utils.data import Dataset
import open3d as o3d

# dataset_dir = "/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/cvpr15_MSRAHandGestureDB"
# save_dir = "/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/MSRA_PointCloud"
# subject_list = ['P0','P1','P2','P3','P4','P5','P6','P7','P8']
# folder_list = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']


class MSRA_Data_Loder(Dataset):
    def __init__(self, dataset_dir , mode, test_subject_id ):

        self.joint_num = 21
        self.world_dim = 3
        
        # self.subject_list = ['P0','P1','P2']
        self.subject_list = ['P0','P1','P2']
        # self.subject_list = ['P0','P1','P2','P3','P4','P5','P6','P7','P8']
        # self.folder_list = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']
        
        self.folder_list =  ['1','2','3','4','5','6','7','8','9','I']
        self.subject_num = len(self.subject_list)
        
        self.dataset_dir = dataset_dir

        self.mode = mode
        self.test_subject_id = test_subject_id


        if not self.mode in ['train', 'test']: raise ValueError('Invalid mode')
        assert self.test_subject_id >= 0 and self.test_subject_id < self.subject_num

        
        self._load()
        
    
    def __getitem__(self, index):
        
        Points = o3d.io.read_point_cloud(self.file_path[index])
        Points = np.array(Points.points)
        return self.file_path[index],Points,self.joints_world[index]

    def __len__(self):
        return self.num_samples


    def _load(self):
        self._compute_dataset_size()

        self.num_samples = self.train_size if self.mode == 'train' else self.test_size
        self.joints_world = np.zeros((self.num_samples, self.joint_num, self.world_dim))
        self.file_path = []


        #
        frame_id = 0
        
        for mid in range(self.subject_num):
            if self.mode == 'train': model_chk = (mid != self.test_subject_id)
            elif self.mode == 'test': model_chk = (mid == self.test_subject_id)
            else: raise RuntimeError('unsupported mode {}'.format(self.mode))

            if model_chk:
                for fd in self.folder_list:
                    annot_file = os.path.join(self.dataset_dir, 'P'+str(mid), fd, 'joint.txt')

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
                        file = os.path.join(self.dataset_dir, 'P'+str(mid) , fd,'{:0>6d}'.format(i-1)+'_Points.ply')
                        
                        self.file_path.append(file)
                        
                        frame_id += 1


    def _compute_dataset_size(self):
        self.train_size, self.test_size = 0, 0

        for mid in range(self.subject_num):
            num = 0
            for fd in self.folder_list:
                annot_file = os.path.join(self.dataset_dir, 'P'+str(mid), fd, 'joint.txt')
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
