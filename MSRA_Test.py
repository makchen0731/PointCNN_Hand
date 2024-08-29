# In[parameters]

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import os
import time
from tqdm import tqdm

import sys
sys.path.append('/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand')

from data.MSRA_PointCloud.MSRA_Data_Loder import MSRA_Data_Loder

from utils.model import FPS_PointCNN 
from utils.util_layers import Dense 

import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler

# from Yolo_OPENCV.util.vis import draw_3d_skeleton
import matplotlib.pyplot as plt
#######################################################################################
## Some helpers

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)


#Loss and opt
parser.add_argument('--SAMPLE_NUM', type=int, default = 1024,  help='number of FPS sample points')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 6,  help='number of input point features')
parser.add_argument('--step_size', type=int, default = 10,  help='step size of StepLR')
parser.add_argument('--gamma', type=int, default = 0.8,  help='gamma of StepLR')
parser.add_argument('--lambda1', type=int, default = 0.5,  help='Loss of lambda1')
parser.add_argument('--lambda2', type=int, default = 1,  help='Loss of lambda2')

#Resume
parser.add_argument('--FPmodel', type=str, default = '',  help='FPmodel training resume')
parser.add_argument('--SPmodel', type=str, default = '',  help='SPmodel for training resume')
parser.add_argument('--FPoptimizer', type=str, default = '',  help='Poptimizer for training resume')
parser.add_argument('--SPoptimizer', type=str, default = '',  help='Voptimizer for training resume')
parser.add_argument('--Train_Error_Memory', type=str, default = '',  help='Train Error for training resume')
parser.add_argument('--Test_Error_Memory', type=str, default = '',  help='Test Error for training resume')



parser.add_argument('--JOINT_NUM', type=int, default = 21,  help='number of joints')
parser.add_argument('--save_root_dir', type=str, default='results',  help='output folder')

parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') 
# parser.add_argument('--num_classes', type=int, default=42, help='classes')  # had MLP features 


parser.add_argument('--est_points', type=int, default=63, help='classes')  #[x,y,z,x1,y1,z1... ...]
parser.add_argument('--est_Vector', type=int, default=63, help='classes')  #[O,V1,V2.........]

#Dataloder Parameter
parser.add_argument('--num_workers', type=int, default=8, help='dataloder num_workers default can be 16 was best with batch size=16')
parser.add_argument('--pin_memory', type=bool, default=True, help='dataloder pin_memory')



opt = parser.parse_args()

torch.cuda.set_device(opt.main_gpu)



# In[Units]
def select_region(pts ,  # (N, x, dims)
                  pts_idx      # (N, P, K)
                  ) :          # (N ,P, K, dims)

    regions = torch.stack([
        pts[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))
    ], dim = 0).view(np.shape(pts)[0],-1,3)
    return regions


def MaxAbs(points,joints): #(-1,1)
    
    OrgPoints = torch.sum(points,dim = 1,keepdim = True)/np.shape(points)[1] # caculator amd regist O for return to wold coordinate MaxAbs standard
    Points_at_Ocoord = points - OrgPoints # move point cloud to Origion O
    Joints_at_Ocoord = joints - OrgPoints # move joint cloud to Origion O
    
    AbsPos = torch.abs(torch.clone(Points_at_Ocoord)) #torch.abs(clone_points) = |points.clone| did't change points's value
    Posmax = torch.max(torch.max(AbsPos,dim = 1,keepdim=True)[0] , dim=2 , keepdim = True)[0]# [0] = values , [1] = indices position
    
    MaxAbs_Pos = Points_at_Ocoord/Posmax
    MaxAbs_jos = Joints_at_Ocoord/Posmax
    
    return MaxAbs_Pos,MaxAbs_jos,OrgPoints,Posmax


def MaxAbs_To_world(out,OrgPoints,Posmax):
    
    Est_World_Joints = out*Posmax+OrgPoints
    
    return Est_World_Joints



def MaxMin(points,joints): #(0,1)
    
    OrgPoints = torch.sum(points,dim = 1,keepdim = True)/np.shape(points)[1] # caculator amd regist O for return to wold coordinate MaxAbs standard
    Points_at_Ocoord = points - OrgPoints # move point cloud to Origion O
    Joints_at_Ocoord = joints - OrgPoints # move joint cloud to Origion O

    Posmax = torch.max(torch.max(Points_at_Ocoord,dim = 1,keepdim=True)[0] , dim=2 , keepdim = True)[0]# [0] = values , [1] = indices position
    Posmin = torch.min(torch.min(Points_at_Ocoord,dim = 1,keepdim=True)[0] , dim=2 , keepdim = True)[0]
    
    MaxAbs_Pos = (Points_at_Ocoord - Posmin)/(Posmax - Posmin)
    MaxAbs_jos = (Joints_at_Ocoord - Posmin)/(Posmax - Posmin)
    
    return MaxAbs_Pos,MaxAbs_jos,OrgPoints,Posmin,Posmax
    
    
def MaxMin_To_world(out,OrgPoints,Posmin,Posmax):

    Est_World_Joints = out*(Posmax - Posmin)+Posmin+OrgPoints
    
    return Est_World_Joints



def farthest_point_sample(xyz, npoint):

    device = xyz.device
    B, N, C = xyz.shape  # B：Batch_size, N:num_points, C:channel
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # n
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
       
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = distance.double()
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    point = torch.empty((1,npoint,np.shape(xyz)[2])).to(device)
    for i,idx in enumerate(centroids):
        xyz1 = xyz[i,idx,:].unsqueeze(dim=0)
        point = torch.cat([point,xyz1], 0)
    point = point[1:,:,:]
    return point



def Average_Error(joints,est_joints):
    arg_error = torch.sqrt(torch.sum((joints - est_joints)**2,dim = 2)) # ((x-x1)^2+(y-y1)^2+(z-z1)^2)^(1/2) (B,21)
    arg_error = torch.sum(arg_error,dim = 0)/np.shape(arg_error)[0]
    
    return arg_error



# In[Draw joints]
def fig2data(fig):

    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def draw_3d_skeleton(pose_cam_xyz,org_xyz,hand_clod ,image_size,est=True):
    """
    pose_cam_xyz,org_xyz,hand_clod ,image_size
    """


    fig = plt.figure(1)
    ax = plt.subplot(111, projection='3d')
    marker_sz = 15
    line_wd = 2
    
    #drow est joints
    color_hand_joints = [[1.0, 0.0, 0.0],
                      [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                      [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                      [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                      [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                      [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little
    
    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
                pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], pose_cam_xyz[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                    pose_cam_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)
        
    
    
    ax.scatter(hand_clod[:,0], hand_clod[:,1], hand_clod[:,2], c='c', marker='^',s=1) #100 is shift hand point cloud in x axis to that we can see clear
    
    ax.axis('auto')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ret = fig2data(fig)  # H x W x 4
    return ret


# In[init]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

save_dir = os.path.join(opt.save_root_dir, "Estimation")
error_dir = os.path.join(opt.save_root_dir, "Error")


#######################################################################################
## Data, transform, dataset and loader
# Data
print('==> Preparing data ..')
data_dir = '/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/data/MSRA_PointCloud'
keypoints_num = 21
test_subject_id = 'P0'


# In[Net]

AbbPointCNN = lambda C_in, C_out, K, D, P: FPS_PointCNN(C_in, C_out,3, K, D, P)


class FPointPCNN(nn.Module):


    def __init__(self,drop_rate):
        
        super(FPointPCNN, self).__init__()
        
        self.est_points = opt.est_points
        self.drop_rate = drop_rate 
        
        self.pcnn1 = AbbPointCNN(3, 48, 8, 1,1024)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(48, 96, 8, 1, 1024),
            AbbPointCNN(96, 192, 12, 2, 384),
            AbbPointCNN(192, 384, 16, 2, 128)
            # AbbPointCNN(192, 384, 16, 3, 128)
        )
        
        self.fcn = nn.Sequential(
            Dense(384, 128),
            Dense(128, 64, drop_rate),
            # Dense(64, self.est_points, with_bn=False, activation=None)
            Dense(64, self.est_points, with_bn=False)
            
        )

    def forward(self, x):
        x = self.pcnn1(x)
        x = self.pcnn2(x)[1]  # grab features
        
        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        logits_view  = logits_mean.view(np.shape(logits_mean)[0],opt.JOINT_NUM,3) # (B,21,3)
        return logits_view


    
class SPointPCNN(nn.Module):


    def __init__(self,drop_rate):
        super(SPointPCNN, self).__init__()
        
        self.est_Vector = opt.est_Vector
        self.drop_rate = drop_rate 
        
        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, 512)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 1, 256),
            AbbPointCNN(64, 128, 12, 2, 128),
            AbbPointCNN(128, 256, 16, 2, 128),
            AbbPointCNN(256, 256, 16, 3, 128)
        )
        
        
        self.fcn = nn.Sequential(
            Dense(256, 128),
            Dense(128, 64, drop_rate),
            # Dense(64, self.est_Vector, with_bn=False, activation=None)
            Dense(64, self.est_Vector, with_bn=False)
        )
        
    def forward(self, x):
        x = self.pcnn1(x)
        x = self.pcnn2(x)[1]  # grab features
        
        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        logits_view  = logits_mean.view(np.shape(logits_mean)[0],opt.JOINT_NUM,3) # (B,21,3)
        return logits_view

    
        
class MyMseLoss(nn.Module):
    

    def __init__(self):
        super(MyMseLoss,self).__init__()
        
        
    def forward(self,Est_P,Tag_P):
        
        Vloss = torch.mean(torch.pow((Est_P - Tag_P),2))
        
        return Vloss

class MyEUCLoss(nn.Module):
    
    def __init__(self):
        super(MyEUCLoss, self).__init__()

    def forward(self, Est_V, Tag_V):
        EUCloss = torch.mean(torch.sqrt(torch.sum((Est_V-Tag_V)**2,dim = 2,keepdim=True)))
        # Vloss = torch.mean(torch.sqrt(torch.sum((Est_V-Tag_V)**2,dim = 2)))
        return EUCloss
    
    
    
# In[dataloder]
# Dataset and loader

# No separate validation dataset, just use test dataset instead
Val_set = MSRA_Data_Loder(data_dir, 'test', test_subject_id)
Val_loader = torch.utils.data.DataLoader(Val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,pin_memory=opt.pin_memory)

#---validation start---#

FPmodel = FPointPCNN(drop_rate=0).double()
SPmodel = SPointPCNN(drop_rate=0).double()


FPmodel.load_state_dict(torch.load('../results/best_result/FMSRA_EUCLoss.pth'))
SPmodel.load_state_dict(torch.load('../results/best_result/SMSRA_EUCLoss.pth'))
    

FPmodel.cuda()
SPmodel.cuda()

FPmodel.eval()
SPmodel.eval()


Test_Error_Epoch_Memory = []


Arg_error = 0
Arg_Tol_error = 0
Final_Arg_error = 0
Final_Arg_Tol_error = 0

with torch.no_grad():
    for i, data in enumerate(tqdm(Val_loader, 0)):

        
        Points , Joints  = data  # data = 'name',"points"(hand only),'joints',points + joints
        Points , Joints =  Points.cuda() , Joints.cuda()
         

        #MaxAbs
        Stand_Pos,Stand_jos,OrgPoints,Posmax = MaxAbs(Points,Joints) # For each (Batch,"Points",Dim) remanber OrgPoints,Posmax this can be unMaxAbs to wold coordinate
        Stand_Pos,Stand_jos,OrgPoints,Posmax = Stand_Pos.cuda(),Stand_jos.cuda(),OrgPoints.cuda(),Posmax.cuda()
        
        torch.cuda.synchronize()
        F_start = time.time()

        ###---Points Estimation---###
        FP_out = FPmodel((Stand_Pos,Stand_Pos)) # [O,P1,P2....]
        
        torch.cuda.synchronize()
        F_end = time.time()
        
        
        #Estimat To World Coordination Error
        Est_World_Joints = MaxAbs_To_world(FP_out,OrgPoints,Posmax) #MaxAbs
        
        Average_World_error = Average_Error(Joints,Est_World_Joints)
        Average_Total_Error = torch.sum(Average_World_error)/np.shape(Average_World_error)[0]
        Arg_error += Average_World_error
        Arg_Tol_error  += Average_Total_Error
        
        
        
        ###---Cat OrgPoints and EstPoints---###
        Est_And_Org = farthest_point_sample(Stand_Pos,491)
        Est_And_Org = torch.cat((Est_And_Org,FP_out),dim=1)

        SP_out = SPmodel((Est_And_Org,Est_And_Org)) # B,(O,V1,V2,........)<-(B,21,3)

        Final_Est_World_Joints = MaxAbs_To_world(SP_out,OrgPoints,Posmax) #MaxAbs

        
        Final_Average_World_error = Average_Error(Joints,Final_Est_World_Joints)
        Final_Average_Total_Error = torch.sum(Final_Average_World_error)/np.shape(Final_Average_World_error)[0]
        
        print(Final_Average_Total_Error)
        
        # if Final_Average_Total_Error <=20:
        Final_Arg_error += Final_Average_World_error
        Final_Arg_Tol_error  += Final_Average_Total_Error
        
    
        Test_Error_Epoch_Memory = np.append(Test_Error_Epoch_Memory,torch.clone(Final_Average_Total_Error).detach().cpu().numpy())
        np.save("%s/3Net_Cat_ESTjoints_each_Test_Error"%(error_dir),Test_Error_Epoch_Memory)
        
        
        # ##---open 3d plot---###
        # import open3d as o3d
    
        points = torch.clone(Points[0]).cpu().detach().numpy()
        org_joints = torch.clone(Joints[0]).cpu().detach().numpy()
        est_joints = torch.clone(Final_Est_World_Joints[0]).cpu().detach().numpy()
        hand_point_clouds = torch.clone(Points[0]).cpu().detach().numpy()
        
        
        #draw joint 
        image = np.zeros((480,480))
        skeletonshow = draw_3d_skeleton(est_joints,org_joints, hand_point_clouds, image.shape[:2],est = False)
        # est_joints = draw_3d_skeleton(est_joints, image.shape[:2])
        # plt.show()
        plt.pause(0.5)
        # input()
        skeletonshow = skeletonshow[:,:,:3]
        

idx = i+1

Arg_error = Arg_error / idx
Arg_Tol_error = Arg_Tol_error / idx

Final_Arg_error = Final_Arg_error / idx
Final_Arg_Tol_error = Final_Arg_Tol_error / idx


print("Test Totel Average error  = %f(mm).\n"%(Arg_Tol_error))
print("Final Test World error=21 Joints(mm).\n",Final_Arg_error)
print("Final Test Totel Average error  = %f(mm).\n"%(Final_Arg_Tol_error))
