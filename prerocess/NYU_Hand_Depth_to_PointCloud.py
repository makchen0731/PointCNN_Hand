import cv2
import numpy as np
import scipy.io as sio
import os
import numpy as np
import sys
import struct
import open3d as o3d
import torch

def pixel2world(x, y, z, img_width, img_height, fx, fy):
    

    normalizedX = x / 640 - 0.5
    normalizedY = 0.5 - y / 480

    
    w_x = normalizedX * z * fx
    w_y = normalizedY * z * fy
    
    # w_x = (x - img_width / 2) * z / fx
    # w_y = (img_height / 2 - y) * z / fy
    w_z = z
    return w_x, w_y, w_z


def depthmap2points(image, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float64)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, w, h, fx, fy)
    
    return points


def load_depthmap(img_path):
    img = cv2.imread(img_path)
    depth = np.asarray(img[:,:,0] + img[:, :, 1] * 256, dtype=np.float64)
    return depth

def hand_only(Pcl , Hand_center):
    Hand_Pcl = np.zeros((40,40,40))
    Hand_Pcl = Pcl[Hand_center[0]-20:Hand_center[0]+20,Hand_center[1]-20:Hand_center[1]+20,Hand_center[2]-20:Hand_center[2]+20]
    return Hand_Pcl



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



class NYU_To_PointCloud():
    def __init__(self,img_path):
        self.xRes = 640
        self.yRes = 480
        self.xzFactor = 1.08836710
        self.yzFactor = 0.817612648
        self.img_path = img_path
        
        
        
    def Hand_Point_Cloud(self):
        depthmap =load_depthmap(self.img_path)
        points = depthmap2points(depthmap,self.xzFactor,self.yzFactor)
        points = points.reshape((-1, 3))
        
        return points



if __name__ == "__main__":
    
    dataset_dir ='/media/ntnu410/1T/NTNU/dataset/DepthOnly/NYU_HAND'
    save_dir = '/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/NYU_HAND/NYU_Hand_PointCloud'
    folder_list = ['test']
    
    
    ###---Depth Load And change To Point Cloud---###
    
    for folder in folder_list:
        mat_file_path = os.path.join(dataset_dir,folder,'joint_data.mat')
        labels = sio.loadmat(mat_file_path)
        joint_xyz = labels['joint_xyz'][0]
        
        for i in range(np.shape(joint_xyz)[0]):
            if i == 0:
                pass
            else:
                Img_path = os.path.join(dataset_dir,folder,'synthdepth_1_{:0>7d}.png'.format(i+1))
                Depth_to_Points = NYU_To_PointCloud(Img_path)
                Points = Depth_to_Points.Hand_Point_Cloud()
                
                x = np.where(Points[:,2] == 0)
                Points = np.delete(Points,x,axis = 0)
                y = np.where(Points == np.max(Points))
                Points = np.delete(Points,y,axis = 0)
                
                fps = farthest_point_sample(torch.from_numpy(Points),6144).numpy()
                Points = Points[fps,:]
                    
                
                
                ###---Save To Point Cloud---###
                # Save_Path = os.path.join(save_dir,folder)
                
                # if not os.path.exists(Save_Path):
                #     os.makedirs(Save_Path)
                    
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(Points)
                # o3d.io.write_point_cloud(Save_Path+'/synthdepth_1_{:0>7d}'.format(i+1)+'_Points.ply', pcd)
            
            
    # # xj_min = np.min(joint_xyz[:,0])-20
    # # xj_max = np.max(joint_xyz[:,0])+20
    # # yj_min = np.min(joint_xyz[:,1])-20
    # # yj_max = np.max(joint_xyz[:,1])+20
    # zj_min = np.min(joint_xyz[:,2])-20
    # zj_max = np.max(joint_xyz[:,2])+20
    
    # # joint_xyz[:,0][joint_xyz[:,0] < xj_min ] = 700
    # # joint_xyz[:,0][joint_xyz[:,0] > xj_max ] = 700
    # # joint_xyz[:,1][joint_xyz[:,1] < yj_min ] = 700
    # # joint_xyz[:,1][joint_xyz[:,1] > yj_max ] = 700
    # joint_xyz[:,2][joint_xyz[:,2] < zj_min ] = -1500
    # joint_xyz[:,2][joint_xyz[:,2] > zj_max ] = -1500
    
    
    # center_xyz = np.loadtxt(center_path)[5000]
    # Hand_PointCloud = hand_only(Point_Cloud,center_xyz)
    
    
    
    ###---open 3d plot---###
    # import open3d as o3d

    # points = np.copy(Point_Cloud)
    # colors = [[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]
    
    
    # PCL = o3d.geometry.PointCloud()
    # Joint = o3d.geometry.PointCloud()
    
    # vis = o3d.visualization.Visualizer()
    # vis.add_geometry(PCL)
    # vis.add_geometry(Joint)
    
    # PCL.points = o3d.utility.Vector3dVector(points)
    # Joint.points = o3d.utility.Vector3dVector(joint_xyz)
    # Joint.colors = o3d.utility.Vector3dVector(colors) 
    
    # o3d.visualization.draw_geometries([PCL]+[Joint], window_name="Open3D1")
    
