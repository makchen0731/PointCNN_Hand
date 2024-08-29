"""
Author: Austin J. Garrett

PyTorch implementation of the PointCNN paper, as specified in:
  https://arxiv.org/pdf/1801.07791.pdf
Original paper by: Yangyan Li, Rui Bu, Mingchao Sun, Baoquan Chen
"""
import open3d as o3d

# External Modules
import torch
import torch.nn as nn

import numpy as np
import time

from knn_cuda import KNN

from torch import cuda

import matplotlib.pyplot as plt

def draw_3d_skeleton(hand_clod ,image_size,est=True):
    
    fig = plt.figure(1)
    fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)
    
    ax = plt.subplot(111, projection='3d')

    ax.scatter(hand_clod[:,0], hand_clod[:,1], hand_clod[:,2]-100, c='r', marker='^',s=1)
    
    ax.axis('auto')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=-85, azim=-75)
    
    
    # ret = fig2data(fig)  # H x W x 4
    # plt.close(fig)
    return fig
# Internal Modules




from utils.util_layers import Conv, SepConv, Dense, EndChannels, Depthwise_conv

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




class XConv(nn.Module):
    """ Convolution over a single point and its neighbors.  """
    
    
    def __init__(self, C_in, C_out, dims, K,
                 P, C_mid, depth_multiplier ):
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param P: Number of representative points.
        :param C_mid: Dimensionality of lifted point features.
        :param depth_multiplier: Depth multiplier for internal depthwise separable convolution.
        """
        super(XConv, self).__init__()


        self.C_in = C_in
        self.C_mid = C_mid
        self.dims = dims
        self.K = K
        

        self.P = P

        # Additional processing layers
        # self.pts_layernorm = LayerNorm(2, momentum = 0.9)

        # Main dense linear layers
        # self.dense1 = Dense(dims, C_mid,with_bn=False)
        # self.dense2 = Dense(C_mid, C_mid,with_bn=False)
        
        
        self.dense1 = nn.Sequential(torch.nn.Linear(dims, self.C_mid),nn.ELU()) #nn.ReLU())
        # self.bn1 = nn.BatchNorm1d(self.C_mid, momentum=0.99)
        self.dense2 = nn.Sequential(torch.nn.Linear(self.C_mid, self.C_mid),nn.ELU()) #nn.ReLU())
        # self.bn2 = nn.BatchNorm1d(self.C_mid, momentum=0.99)
        
        
        # # Layers to generate X
        # self.x_trans = nn.Sequential(
        #     EndChannels(Conv(
        #         in_channels = dims,
        #         out_channels = K*K,
        #         kernel_size = (1, K),
        #         with_bn = False
        #     )),
        #     Dense(K*K, K*K, with_bn = False),
        #     Dense(K*K, K*K, with_bn = False, activation = None)
        # )
        
        # Layers to generate X 
        self.conv2d = EndChannels(Conv(in_channels = dims,out_channels = K*K,kernel_size = (1, K),with_bn = False))
        self.dep_conv1 = EndChannels(Depthwise_conv(in_channels = K ,depth_multiplier = K ,kernel_size=(1,K),activation = True))
        self.dep_conv2 = EndChannels(Depthwise_conv(in_channels = K ,depth_multiplier = K ,kernel_size=(1,K),activation = None))
        
        self.end_conv = EndChannels(SepConv(
            in_channels = C_mid + C_in,
            out_channels = C_out,
            kernel_size = (1, K),
            depth_multiplier = depth_multiplier
        )).cuda()
        
    def forward(self, x ):
        """
        Applies XConv to the input data.
        :param x: (rep_pt, pts, fts) where
          - rep_pt: Representative point.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the feature
          associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated into point rep_pt.
        """
        rep_pt, pts, fts = x

        # if fts is not None:
        #     assert(rep_pt.size()[0] == pts.size()[0] == fts.size()[0])  # Check N is equal.
        #     assert(rep_pt.size()[1] == pts.size()[1] == fts.size()[1])  # Check P is equal.
        #     assert(pts.size()[2] == fts.size()[2] == self.K)            # Check K is equal.
        #     assert(fts.size()[3] == self.C_in)                          # Check C_in is equal.
        # else:
        #     assert(rep_pt.size()[0] == pts.size()[0])                   # Check N is equal.
        #     assert(rep_pt.size()[1] == pts.size()[1])                   # Check P is equal.
        #     assert(pts.size()[2] == self.K)                             # Check K is equal.
        # assert(rep_pt.size()[2] == pts.size()[3] == self.dims)          # Check dims is equal.

        N = len(pts)
        P = rep_pt.size()[1]  # (N, P, K, dims)
        p_center = torch.unsqueeze(rep_pt, dim = 2)  # (N, P, 1, dims)

        # Move pts to local coordinate system of rep_pt.
        pts_local = pts - p_center  # (N, P, K, dims)
        # pts_local = self.pts_layernorm(pts - p_center)
        
        # Individually lift each point into C_mid space.
        fts_lifted0 = self.dense1(pts_local)
        fts_lifted  = self.dense2(fts_lifted0)  # (N, P, K, C_mid)
        
        # fts_lifted0 = self.dense1(pts_local.view(-1,3))
        # fts_lifted0 = self.bn1(fts_lifted0).view(N, self.P, self.K, self.C_mid)
        # fts_lifted  = self.dense2(fts_lifted0.view(-1,self.C_mid))
        # fts_lifted = self.bn1(fts_lifted).view(N, self.P, self.K, self.C_mid)
        
        if fts is None:
            fts_cat = fts_lifted
        else:
            fts_cat = torch.cat((fts_lifted, fts), -1)  # (N, P, K, C_mid + C_in)

        # # Learn the (N, K, K) X-transformation matrix.
        # X_shape = (N, P, self.K, self.K)
        # X = self.x_trans(pts_local)
        # X = X.view(*X_shape)
        
        ###--- Learn the (N, K, K) X-transformation matrix. ---###
        
        X_shape = (N, P, self.K, self.K)
        X = self.conv2d(pts_local)
        X = X.view(*X_shape)
        X = self.dep_conv1(X)
        X = X.view(*X_shape)
        X = self.dep_conv2(X)
        X = X.view(*X_shape)
        
        
        
        
        # Weight and permute fts_cat with the learned X.
        fts_X = torch.matmul(X, fts_cat)
        fts_p = self.end_conv(fts_X).squeeze(dim = 2)
        return fts_p

class PointCNN(nn.Module):
    """ Pointwise convolutional model. """

    def __init__(self, C_in , C_out , dims , K, D, P,
                 # r_indices_func : Callable[[UFloatTensor,  # (N, P, dims)
                 #                            UFloatTensor,  # (N, x, dims)
                 #                            int, int],
                 #                           ULongTensor]    # (N, P, K)
                ) :
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param D: "Spread" of neighboring points.
        :param P: Number of representative points.
        """
        super(PointCNN, self).__init__()

        C_mid = C_out // 2 if C_in == 0 else C_out // 4

        if C_in == 0:
            depth_multiplier = 1
        else:
            depth_multiplier = min(int(np.ceil(C_out / C_in)), 4)


        ###---Orig KNN---###
        # self.r_indices_func = lambda rep_pts, pts: r_indices_func(rep_pts, pts, K, D)
        ###---Import KNN---###
        self.knn = KNN(k=K, transpose_mode=True)
        
        
        self.dense = Dense(C_in, C_out // 2,with_bn =False) if C_in != 0 else None
        self.x_conv = XConv(C_out // 2 if C_in != 0 else C_in, C_out, dims, K, P, C_mid, depth_multiplier)
        self.D = D

    def select_region(self, pts ,  # (N, x, dims)
                      pts_idx      # (N, P, K)
                     ) :          # (P, K, dims)
        """
        Selects neighborhood points based on output of r_indices_func.
        :param pts: Point cloud to select regional points from.
        :param pts_idx: Indices of points in region to be selected.
        :return: Local neighborhoods around each representative point.
        """
        regions = torch.stack([
            pts[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))
        ], dim = 0)
        return regions

    def forward(self, x   # (N, P, dims)
                        # (N, x, dims)
                           # (N, x, C_in)
               ) :              # (N, P, C_out)
        """
        Given a set of representative points, a point cloud, and its
        corresponding features, return a new set of representative points with
        features projected from the point cloud.
        :param x: (rep_pts, pts, fts) where
          - rep_pts: Representative points.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the
          feature associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated to rep_pts.
        """
        rep_pts, pts, fts = x
        
        fts = self.dense(fts) if fts is not None else fts #fts = (B,P,C)

        # # This step takes ~97% of the time. Prime target for optimization: KNN on GPU.
        # pts_idx = self.r_indices_func(rep_pts.cpu(), pts.cpu()).cuda()
        # -------------------------------------------------------------------------- #

        ###---import KNN---###
        _, pts_idx = self.knn(pts.cuda(),rep_pts.cuda())
                

        pts_regional = self.select_region(pts, pts_idx)
        fts_regional = self.select_region(fts, pts_idx) if fts is not None else fts
        fts_p = self.x_conv((rep_pts, pts_regional, fts_regional))

        return fts_p

class FPS_PointCNN(nn.Module):
    """ PointCNN with randomly subsampled representative points. """

    def __init__(self, C_in, C_out, dims , K , D, P,
                 # r_indices_func : Callable[[UFloatTensor,  # (N, P, dims)
                 #                            UFloatTensor,  # (N, x, dims)
                 #                            int, int],
                 #                           ULongTensor]    # (N, P, K)
                ) -> None:
        """ See documentation for PointCNN. """
        super(FPS_PointCNN, self).__init__()
        # self.pointcnn = PointCNN(C_in, C_out, dims, K, D, P, r_indices_func)
        self.pointcnn = PointCNN(C_in, C_out, dims, K, D, P)
        self.P = P

    def forward(self, x # (N, x, dims)
                          # (N, x, dims)
               ):       # (N, P, dims)
                              # (N, P, C_out)
        """
        Given a point cloud, and its corresponding features, return a new set
        of randomly-sampled representative points with features projected from
        the point cloud.
        :param x: (pts, fts) where
         - pts: Regional point cloud such that fts[:,p_idx,:] is the
        feature associated with pts[:,p_idx,:].
         - fts: Regional features such that pts[:,p_idx,:] is the feature
        associated with fts[:,p_idx,:]
        """
        
        pts, fts = x
        
        
        
        #draw joint 

        # axis_pcd = o3d.create_mesh_coordinate_frame(size=50, origin=org)
        
        # points = torch.clone(pts[0]).cpu().detach().numpy()
        # test_pcd = o3d.geometry.PointCloud()
        # vis = o3d.visualization.Visualizer()
        # vis.add_geometry(test_pcd)
        # test_pcd.points = o3d.utility.Vector3dVector(points)

        # o3d.visualization.draw_geometries([test_pcd] , window_name="Open3D1")
        
        if 0 < self.P < pts.size()[1]:
            # Select random set of indices of subsampled points.
            # idx = np.random.choice(pts.size()[1], self.P, replace = False).tolist()
            
            # FPS select 

            rep_pts = farthest_point_sample(pts, self.P)
            

            # rep_pts = pts[:,fps[0],:]
        else:
            # All input points are representative points.
            rep_pts = pts
            
            
        # torch.cuda.synchronize()
        # start = time.time()
        
        rep_pts_fts = self.pointcnn((rep_pts, pts, fts)) # rep_pts_fts is Fp from Xconv 
        
        # torch.cuda.synchronize()
        # end = time.time()
            
        # print(end - start)
        
        
        return rep_pts, rep_pts_fts
