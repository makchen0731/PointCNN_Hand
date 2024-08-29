import os
import numpy as np
import sys
import struct
import torch
import open3d as o3d
        
def pixel2world(x, y, z, img_width, img_height, fx, fy):
    
    w_x = (x - img_width / 2) * z / fx
    w_y = (img_height / 2 - y) * z / fy
    w_z = z
    return w_x, w_y, w_z


def depthmap2points(image, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, w, h, fx, fy)
    
    return points




def load_depthmap(filename, img_width, img_height, max_depth): 
    with open(filename, mode='rb') as f:
        data = f.read()
        _, _, left, top, right, bottom = struct.unpack('I'*6, data[:6*4])
        num_pixel = (right - left) * (bottom - top)
        cropped_image = struct.unpack('f'*num_pixel, data[6*4:])

        cropped_image = np.asarray(cropped_image).reshape(bottom-top, -1)
        depth_image = np.zeros((img_height, img_width), dtype=np.float32)
        depth_image[top:bottom, left:right] = cropped_image
        depth_image[depth_image == 0] = max_depth
        
    return depth_image
    
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



class MSRA_To_PointCloud():
    def __init__(self,dataset_dir,subject_list,folder_list,idx):
        self.Joint_num = 21
        self.Hand_Cloud_num = 3074
        self.img_width = 320
        self.img_height = 240
        self.min_depth = 100
        self.max_depth = 700
        self.fx = 241.42
        self.fy = 241.42
        
        
        self.dataset_dir = dataset_dir
        self.subject_list = subject_list
        self.folder_list = folder_list
        self.idx = idx
        
        
        self.Bin_File = os.path.join( self.dataset_dir,self.subject_list,self.folder_list ,'{:0>6d}'.format(idx-1) + '_depth.bin')
        
        
        
    def Hand_Point_Cloud(self):
        depthmap =load_depthmap(self.Bin_File,self.img_width,self.img_height,self.max_depth)
        points = depthmap2points(depthmap, self.fx, self.fy)
        points = points.reshape((-1, 3))
        
        return points
        
    
if __name__ == "__main__":
    dataset_dir = "/media/ntnu410/1T/NTNU/dataset/DepthOnly/cvpr15_MSRAHandGestureDB"
    save_dir = "/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/MSRA_PointCloud"
    subject_list = ['P0','P1','P2','P3','P4','P5','P6','P7','P8']
    folder_list = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']
    
    for sub_list in subject_list:
        for fold_list in folder_list:
            File_Num = os.path.join(dataset_dir,sub_list,fold_list,'joint.txt')
            lines = []
            with open(File_Num) as f:
                lines = [line.rstrip() for line in f]
            for idx in range(1,len(lines)):
                Depth_to_Points = MSRA_To_PointCloud(dataset_dir,sub_list,fold_list,idx )
                Points = Depth_to_Points.Hand_Point_Cloud()
                
                ###--- Hand Point Only(delete depth > 700) & Get FPS =3074 ---###
                x = np.where(Points[:,2] == 700)
                Points = np.delete(Points,x,axis = 0)
                fps = farthest_point_sample(torch.from_numpy(Points),6114).numpy()
                Points = Points[fps,:]
                
                ##---Save To Point Cloud---###
                Save_Path = os.path.join(save_dir,sub_list,fold_list)
                
                if not os.path.exists(Save_Path):
                    os.makedirs(Save_Path)
                    
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(Points)
                o3d.io.write_point_cloud(Save_Path+'/{:0>6d}'.format(idx-1)+'_Points.ply', pcd)