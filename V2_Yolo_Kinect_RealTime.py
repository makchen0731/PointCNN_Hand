


# In[parameters]

import numpy as np
import torch
import torch.nn as nn

import argparse
import os
import time

import sys
sys.path.append('/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand')


from utils.model import RandPointCNN # Orig PointCNN Model
from utils.util_layers import Dense # Orig PointCNN Model

#Yolo kinect

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame, FrameMap
from pylibfreenect2 import (Logger,
                            createConsoleLogger,
                            createConsoleLoggerWithDefaultLevel,
                            getGlobalLogger,
                            setGlobalLogger,
                            LoggerLevel)

from pylibfreenect2 import OpenGLPacketPipeline

from Yolo_OPENCV.yolo import YOLO

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import cv2
import open3d as o3d



#######################################################################################
## Some helpers
parser = argparse.ArgumentParser()

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


parser.add_argument('--est_points', type=int, default=63, help='classes')  
parser.add_argument('--est_Vector', type=int, default=63, help='classes')  


#Yolo parameters
parser.add_argument('-c', '--confidence', default=0.7, help='Confidence for yolo')
parser.add_argument('-nh', '--hands', default=-1, help='Total number of hands to be detected per frame (-1 for all)')
parser.add_argument('-s', '--size', default=416, help='Size for yolo')



opt = parser.parse_args()


yolo = YOLO("Yolo_OPENCV/cfg/cross-hands.cfg", "Yolo_OPENCV/cfg/cross-hands.weights", ["hand"])
yolo.size = int(opt.size)
yolo.confidence = float(opt.confidence)




# In[Units]

def MaxAbs(points): #(-1,1)
    
    OrgPoints = torch.sum(points,dim = 1,keepdim = True)/np.shape(points)[1] # caculator amd regist O for return to wold coordinate MaxAbs standard
    Points_at_Ocoord = points - OrgPoints # move point cloud to Origion O
    # Joints_at_Ocoord = joints - OrgPoints # move joint cloud to Origion O
    
    AbsPos = torch.abs(torch.clone(Points_at_Ocoord)) #torch.abs(clone_points) = |points.clone| did't change points's value
    Posmax = torch.max(torch.max(AbsPos,dim = 1,keepdim=True)[0] , dim=2 , keepdim = True)[0]# [0] = values , [1] = indices position
    
    MaxAbs_Pos = Points_at_Ocoord/Posmax
    # MaxAbs_jos = Joints_at_Ocoord/Posmax
    
    return MaxAbs_Pos,OrgPoints,Posmax


def MaxAbs_To_world(out,OrgPoints,Posmax):
    
    # World_Joints = OrgPoints.view(np.shape(out)[0],21,3)
    # Est_Value = out.view(np.shape(out)[0],21,3)
    Est_World_Joints = out*Posmax+OrgPoints
    
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


def Hand_cut(img,dx,dy):
    hand_ony = np.zeros(((dx[0]-20 + dx[1]),(dy[0]-20 + dy[1]),3))
    hand_ony = img[(dy[0]-20):(dy[0] + dy[1]),(dx[0]-20): (dx[0]+ dx[1]),:]
    
    # hand_ony = np.zeros(((dx[0]-20 + dx[1]),(dy[0]-20 + dy[1]),3))
    # hand_ony = img[(dy[0]- dy[1]):(dy[0] + dy[1]),(dx[0]-dx[1]): (dx[0]+ dx[1]),:]
    
    return hand_ony


#Drow XYZ angle in Matplot
def PlotXYZ(x,y,z,frame):
    plt.figure()
    plt.plot(frame,x,"r",label = "X")
    plt.plot(frame,y,"g",label = "Y")
    plt.plot(frame,z,"b",label = "Z")
    
    
    plt.ylabel("angle")
    plt.xlabel("frame")
    
    plt.legend(loc = 'upper right')


def pixel2world(x, y, z, cx, cy, fx, fy):
    
    # w_x = (cx - x) * z / fx
    w_x = (x - cx) * z / fx
    w_y = (y - cy) * z / fy
    # w_y = (cy - y) * z / fy
    w_z = z
    return w_x, w_y, w_z


def depthmap2points(image, cx, cy, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, cx, cy, fx, fy) # points shape = (H,W,(x,y,z))
    
    return points


# In[Draw joints]
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf



#Org
# def draw_3d_skeleton(pose_cam_xyz, image_size):
#     """
#     :param pose_cam_xyz: 21 x 3
#     :param image_size: H, W
#     :return:
#     """
#     assert pose_cam_xyz.shape[0] == 21
    
#     # Plt Show
#     color_hand_joints = [[1.0, 0.0, 0.0],
#                      [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
#                      [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
#                      [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
#                      [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
#                      [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little
    
#     fig = plt.figure(1)
#     fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)
    
#     ax = plt.subplot(111, projection='3d')
#     marker_sz = 15
#     line_wd = 2
    
#     for joint_ind in range(pose_cam_xyz.shape[0]):
#         ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
#                 pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
#         if joint_ind == 0:
#             continue
#         elif joint_ind % 4 == 1:
#             ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], pose_cam_xyz[[0, joint_ind], 2],
#                     color=color_hand_joints[joint_ind], lineWidth=line_wd)
#         else:
#             ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
#                     pose_cam_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
#                     linewidth=line_wd)
    
    
#     ax.scatter(hand_clod[:,0], hand_clod[:,1], hand_clod[:,2], c='c', marker='^',s=1) #100 is shift hand point cloud in x axis to that we can see clear
    
    
#     ax.axis('auto')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.view_init(elev=225, azim=90)
    
    
#     ret = fig2data(fig)  # H x W x 4
#     # plt.close(fig)
#     return ret






def draw_3d_skeleton(pose_cam_xyz,org_xyz,hand_clod ,image_size,est=True):
    """
    pose_cam_xyz,org_xyz,hand_clod ,image_size
    """
    # assert pose_cam_xyz.shape[0] == 21
    
    # Plt Show
    

    fig = plt.figure(1)
    # fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)
    
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
        
            
    # #drow org joints
    # color_hand_target = [[1.0, 0.0, 0.0],
    #               [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # thumb
    #               [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # index
    #               [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # middle
    #               [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # ring
    #               [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]  # little
    
    # # fig = plt.figure(1)
    # # fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)
    
    # # ax = plt.subplot(111, projection='3d')
    # # marker_sz = 15
    # # line_wd = 2
    
    # for joint_ind in range(org_xyz.shape[0]):
    #     ax.plot(org_xyz[joint_ind:joint_ind + 1, 0], org_xyz[joint_ind:joint_ind + 1, 1],
    #             org_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_target[joint_ind], markersize=marker_sz)
    #     if joint_ind == 0:
    #         continue
    #     elif joint_ind % 4 == 1:
    #         ax.plot(org_xyz[[0, joint_ind], 0], org_xyz[[0, joint_ind], 1], org_xyz[[0, joint_ind], 2],
    #                 color=color_hand_target[joint_ind], lineWidth=line_wd)
    #     else:
    #         ax.plot(org_xyz[[joint_ind - 1, joint_ind], 0], org_xyz[[joint_ind - 1, joint_ind], 1],
    #                 org_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_target[joint_ind],
    #                 linewidth=line_wd)
    
    
    
    ax.scatter(hand_clod[:,0], hand_clod[:,1], hand_clod[:,2], c='c', marker='^',s=1) #100 is shift hand point cloud in x axis to that we can see clear
    
    ax.axis('auto')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=-85, azim=-75)
    plt.pause(0.05)
    
    ret = fig2data(fig)  # H x W x 4
    # plt.close(fig)
    return ret
# In[Net]

AbbPointCNN = lambda C_in, C_out, K, D, P: RandPointCNN(C_in, C_out,3, K, D, P)


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
    
if __name__=='__main__':
    
    # In[Kinect init]
    try:
        from pylibfreenect2 import OpenGLPacketPipeline
        pipeline = OpenGLPacketPipeline()
    except:
        try:
            from pylibfreenect2 import OpenCLPacketPipeline
            pipeline = OpenCLPacketPipeline()
        except:
            from pylibfreenect2 import CpuPacketPipeline
            pipeline = CpuPacketPipeline()
    print("Packet pipeline:", type(pipeline).__name__)
    
    # Create and set logger
    logger = createConsoleLogger(LoggerLevel.Debug)
    setGlobalLogger(logger)
    
    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("No device connected!")
        sys.exit(1)
    
    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial, pipeline=pipeline)
    
    listener = SyncMultiFrameListener(
        FrameType.Color | FrameType.Ir | FrameType.Depth)
    
    # Register listeners
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)
    
    device.start()
    
    # NOTE: must be called after device.start()
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())
    
    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)
        
    # In[Load Madel]
    
    torch.cuda.set_device(opt.main_gpu)
    device = torch.device("cuda:0")

    FPmodel = FPointPCNN(drop_rate=0).double()
    SPmodel = SPointPCNN(drop_rate=0).double()
    
    # FPmodel.load_state_dict(torch.load(os.path.join(save_dir, '3FPmadel_2Net_Cat_ESTjoints_%d.pth'% (epoch))))
    # SPmodel.load_state_dict(torch.load(os.path.join(save_dir, '3SPmadel_2Net_Cat_ESTjoints_%d.pth'% (epoch))))
        
    # FPmodel.load_state_dict(torch.load('/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/results/best_result/FNYU_EUL_ESTjoints_264.pth'))
    # SPmodel.load_state_dict(torch.load('/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/results/best_result/SNYU_EUL_ESTjoints_264.pth'))
    
    FPmodel.load_state_dict(torch.load('/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/results/best_result/FMSRA_EUCLoss_183.pth'))
    SPmodel.load_state_dict(torch.load('/home/ntnu410/NTNU/virtualenv/PointCnn/PointCNN_Hand/results/best_result/SMSRA_EUCLoss_183.pth'))
        
    
    FPmodel.to(device)
    SPmodel.to(device)
    
    FPmodel.eval()
    SPmodel.eval()
    
    
    
    Arg_error = 0
    Arg_Tol_error = 0
    Final_Arg_error = 0
    Final_Arg_Tol_error = 0
    # idx = 1
    
    
    
    with torch.no_grad():
        while True:
            # In[Realsense seting]
            #Get frameset of color and depth
            cx = 254.878
            cy = 205.395
            fx = 365.456
            fy = 365.456
            
            
            KT_frames = listener.waitForNewFrame()
            
            # ir = KT_frames["ir"]
            color = KT_frames["color"]
            depth = KT_frames["depth"]
        
            # with optinal parameters
            registration.apply(color, depth, undistorted, registered)
        
            # cv2.imshow("ir", ir.asarray() / 65535.)  #
            Depth_Meter = depth.asarray() / 45.
            cv2.imshow("depth", cv2.flip(Depth_Meter,0)) # real value is mm   /4500 = Max distance 
            Color_Depth = registered.asarray(np.uint8)[:,:,:3]
                
            #Get Pointcloud By focal lenght
            xyz_vtx = depthmap2points(Depth_Meter,cx,cy,fx,fy)#.reshape(-1, 3)
            
            
            listener.release(KT_frames)
            
            # In[Yolo Hand]
            Color_Depth = np.flip(Color_Depth,0)
            width, height, inference_time, results = yolo.inference(Color_Depth)
        
            # display fps
            frames = np.copy(Color_Depth)
            cv2.putText(frames, 'FPS = {}'.format(round(1/inference_time,2)), (15,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 2)
        
            # sort by confidence
            results.sort(key=lambda x: x[2])
        
            # how many hands should be shown
            hand_count = len(results)
            if opt.hands != -1:
                hand_count = int(opt.hands)
            
            # display hands
            for detection in results[:hand_count]:
                idx, name, confidence, x, y, w, h = detection
        
                # draw a bounding box rectangle and label on the image
                color = (0, 255, 255)
                if w >= h:
                    h = w
                elif h >= w:
                    w = h
                    
                cx = int(x + (w / 2)) #X,Y is left upper
                cy = int(y + (h / 2))
        
                cv2.rectangle(frames, (x-20, y-20), (x + w + 10, y + h +10), color, 2)
                # cv2.rectangle(frame, ((cx-180, cy-120), (cx+180, cy + 120), color, 2)
                text = "%s (%s)" % (name, round(confidence, 2))
                cv2.putText(frames, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                
                cv2.circle(frames,(cx,cy),radius = 1, color = (0, 0, 255) ,thickness = 4 )
                
                cut_rangex = [x , w]
                cut_rangey = [y , h+20]
                # print(' x = {},y = {}'.format((x-20 + w),(y-20 + h)))
        
                if (x-20 + w <= 640) and (y-20 + h <= 480):
                    hand_only = Hand_cut(Color_Depth,cut_rangex,cut_rangey)
                    # draw_3d_skeleton(None,None,np.copy(xyz_vtx).reshape((-1,3)),None)
                    
                    xyz_cut = Hand_cut(xyz_vtx,cut_rangex,cut_rangey)
                    
                    
                    
                if 'hand_only' in  globals():
                    cv2.imshow("Hand Only", hand_only)
                    # print(np.shape(frames))
                    
                    
                    pointsss = xyz_cut.reshape((-1,3)) #*100
                    
                    x = np.where(pointsss[:,2] == 0)
                    pointsss = np.delete(pointsss,x,axis = 0)
                    
                    handbox = np.where(pointsss[:,2] >= np.min(pointsss[:,2] + 4)) #+4 is 4cm
                    handonly_points = np.delete(pointsss,handbox,axis = 0)

                    
                    ###---joint Estimate--###
                    
                    if np.shape(handonly_points)[0] >= 1024:
                        # Points  = handonly_points#.to(device)
                        # Points  =  Points.cuda() 
                        Points  =  torch.tensor(handonly_points).cuda().double()
                        
                        
                        #MaxAbs
                        Stand_Pos = torch.clone(Points).reshape((-1,3)).unsqueeze(0)
                        Stand_Pos,OrgPoints,Posmax = MaxAbs(Stand_Pos) # For each (Batch,"Points",Dim) remanber OrgPoints,Posmax this can be unMaxAbs to wold coordinate
                        Stand_Pos,OrgPoints,Posmax = Stand_Pos.cuda(),OrgPoints.cuda(),Posmax.cuda()
                        
                        Stand_Pos = farthest_point_sample(Stand_Pos,6144)
                        
                        ###---Points Estimation---###
                        FP_out = FPmodel((Stand_Pos,Stand_Pos)) # [O,P1,P2....]
                        
                        #Estimat To World Coordination Error
                        Est_World_Joints = MaxAbs_To_world(FP_out,OrgPoints,Posmax) #MaxAbs
                        # Est_World_Joints = MaxMin_To_world(FP_out,OrgPoints,Posmin,Posmax) #MaxMin
                        
                        ###---Cat OrgPoints and EstPoints---###
                        Est_And_Org = farthest_point_sample(Stand_Pos,491)
                        Est_And_Org = torch.cat((Est_And_Org,FP_out),dim=1)
                        
                        SP_out = SPmodel((Est_And_Org,Est_And_Org)) # B,(O,V1,V2,........)<-(B,21,3)
                        
                        
                        Final_Est_World_Joints = MaxAbs_To_world(SP_out,OrgPoints,Posmax) #MaxAbs
                        
                        
                        # # O3D point cloud
                        # joints = Final_Est_World_Joints.cuda().cpu().numpy().squeeze(0)
                        # colors = [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]]
                        # # colors = [[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]
                        # # colors2 = [[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]
                        # # colors = [[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0]]
                        # test_pcd = o3d.geometry.PointCloud()
                        # test_pcd_Joint = o3d.geometry.PointCloud()
                        # # test_Est_World_Joints = o3d.geometry.PointCloud()
                        # vis = o3d.visualization.Visualizer()
                        
                        # # vis.add_geometry(axis_pcd)
                        # vis.add_geometry(test_pcd)
                        # vis.add_geometry(test_pcd_Joint)
                        # # vis.add_geometry(test_Est_World_Joints)
                        
                    
                        # test_pcd.points = o3d.utility.Vector3dVector(handonly_points)
                        
                        # test_pcd_Joint.colors = o3d.utility.Vector3dVector(colors)
                        # test_pcd_Joint.points = o3d.utility.Vector3dVector(joints)
                        #     # test_Est_World_Joints.points = o3d.utility.Vector3dVector(Org_joints)
                        
                         
                        #     # test_Est_World_Joints.colors = o3d.utility.Vector3dVector(colors2) 
                            
                        # o3d.visualization.draw_geometries([test_pcd] + [test_pcd_Joint] , window_name="Open3D1")
                    
                    
                    
                        #Draw skeleton
                        pose_xyz = Final_Est_World_Joints.cuda().cpu().numpy().squeeze(0).reshape((21,3))
                        
                        
                        image = np.zeros((480,480))
                        draw_3d_skeleton(pose_xyz,None,handonly_points ,image_size=image.shape[:2],est=True)
                        # skeleton_3d = draw_3d_skeleton(pose_xyz, hand_only.shape[:2])
                        # skeletonshow = skeleton_3d[:,:,:3]
                        # cv2.imshow("skeletonshow", skeletonshow)
                        
                        # print(Final_Est_World_Joints)
                    

            cv2.imshow("preview", frames)

            # rval, frame = vc.read()
        
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
    
    cv2.destroyWindow("preview")
    
    device.stop()
    device.close()
    
    sys.exit(0)
    # vc.release()
    
