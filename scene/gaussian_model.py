#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
# from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity


class GaussianModel(nn.Module):

    def build_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def setup_functions(self):
        
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, semantic_feature_size = 512):
        super().__init__()
        
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        
        self.keyframe_idx = torch.empty(0)
        self.trackable_mask = torch.empty(0)
        self.edge_mask = torch.empty(0)
        
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.semantic_feature_size = semantic_feature_size  
        self.setup_functions()
        self._semantic_feature = torch.empty(0) 

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._semantic_feature, 
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._semantic_feature) = model_args 
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    @property
    def get_semantic_feature(self):
        return self._semantic_feature 
    
    def rewrite_semantic_feature(self, x):
        self._semantic_feature = x
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def create_from_pcd2_tensor(self, points, colors, rots_, scales_, z_vals_, trackable_idxs, semantic_feature, edge_mask, speedup = True):
        # Create initial gaussian map
        # Initialize with rotations/scales from gicp
        fused_point_cloud = points
        fused_color = RGB2SH(colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        

        if semantic_feature == None:
            if speedup:
                # Use zero initialization for point features in speedup mode
                semantic_feature = torch.zeros(fused_point_cloud.shape[0], self.semantic_feature_size).float().cuda()
            else:
                semantic_feature = torch.empty(fused_point_cloud.shape[0], self.semantic_feature_size).float().cuda()
                torch.nn.init.xavier_uniform_(semantic_feature)
        # self._semantic_feature = torch.zeros(fused_point_cloud.shape[0], semantic_feature_size, 1).float().cuda() 
        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        z_vals = torch.clamp_min((z_vals_**1.5)*2., 1.).unsqueeze(-1).repeat(1,3)
        scales_withz = scales_ / z_vals
        scales = torch.log(scales_withz)
        rots = rots_
        
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
    
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        self.trackable_mask = torch.zeros((self.get_xyz.shape[0]), dtype=torch.bool, device="cuda")
        # print(trackable_idxs)
        self.trackable_mask[(trackable_idxs)] = 1
        
        self.edge_mask = torch.tensor(edge_mask, dtype=torch.bool, device="cuda")
        
        self.keyframe_idx = torch.zeros((self.get_xyz.shape[0],1), dtype=torch.int32, device="cuda")
        self._semantic_feature = nn.Parameter(semantic_feature.unsqueeze(1).contiguous().requires_grad_(True))
        
        torch.cuda.empty_cache()
    
    def add_from_pcd2_tensor(self, points, colors, rots_, scales_, z_vals_, trackable_idxs, semantic_feature, keyframe_idx, edge_mask, speedup = True):
        # Add new gaussians to the whole gaussian map
        # Initialize with rotations/scales from gicp
        fused_point_cloud = points
        fused_color = RGB2SH(colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        if semantic_feature == None:
            if speedup:
                semantic_feature = torch.zeros(fused_point_cloud.shape[0], self.semantic_feature_size).float().cuda()
            else:
                semantic_feature = torch.empty(fused_point_cloud.shape[0], self.semantic_feature_size).float().cuda()
                torch.nn.init.xavier_uniform_(semantic_feature)
        
        # Ours(z_value**1.5*2)
        z_vals = torch.clamp_min((z_vals_**1.5)*2., 1.).unsqueeze(-1).repeat(1,3)
        scales_withz = scales_ / z_vals
        scales = torch.log(scales_withz)
        rots = rots_

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self.new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.new_features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.new_features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.new_scaling = nn.Parameter(scales.requires_grad_(True))
        self.new_rotation = nn.Parameter(rots.requires_grad_(True))
        self.new_opacities = nn.Parameter(opacities.requires_grad_(True))
        self.new_semantic_feature = nn.Parameter(semantic_feature.unsqueeze(1).contiguous().requires_grad_(True))

        # Update trackable table #
        self.new_trackable_mask = torch.zeros((self.new_xyz.shape[0]), dtype=torch.bool, device="cuda")
        if len(trackable_idxs) != 0:
            self.new_trackable_mask[(trackable_idxs)] = 1
        
        self.new_edge_mask = torch.tensor(edge_mask, dtype=torch.bool, device="cuda")
        
        # self.trackable_mask = torch.concat([self.trackable_mask, self.new_trackable_mask], dim=0)
        self.densification_postfix(self.new_xyz, self.new_features_dc, 
                                   self.new_features_rest, self.new_opacities,
                                   self.new_scaling, self.new_rotation, self.new_trackable_mask, self.new_semantic_feature, self.new_edge_mask)
        new_keyframe_idx = torch.ones((self.new_xyz.shape[0], self.keyframe_idx.shape[1]), device="cuda", dtype=torch.int32) * keyframe_idx
        # Expanding keyframe_idx table
        # Add new gaussians
        self.keyframe_idx = torch.concat([  self.keyframe_idx,
                                            new_keyframe_idx], dim=0)
        
        torch.cuda.empty_cache()


    def add_densified_gaussians(self, new_points, new_colors, new_semantic_feature,
                                new_z_values, keyframe_idx, pixel_scale, speedup=True):
        """Insert new Gaussians produced by language-boundary / error-guided
        densification (B1).

        New Gaussians are back-projected image points: opacity is initialised low
        (0.1), rotation is identity, and the scale is isotropic and proportional
        to the metric pixel footprint at the point depth (z * pixel_scale, with
        pixel_scale = 1/fx). The semantic feature is expected to already be in the
        per-Gaussian dimensionality (i.e. encoded to 16-D by the caller in speedup
        mode). ``keyframe_idx`` is propagated so loop closure transforms these
        points together with their source keyframe.

        Returns the number of Gaussians actually added.
        """
        M = new_points.shape[0]
        if M == 0:
            return 0

        fused_color = RGB2SH(new_colors)
        features = torch.zeros((M, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        raw_scale = (new_z_values.float() * pixel_scale).clamp_min(1e-6).unsqueeze(-1).repeat(1, 3)
        scales = torch.log(raw_scale)

        rots = torch.zeros((M, 4), dtype=torch.float, device="cuda")
        rots[:, 0] = 1.0  # identity quaternion (w, x, y, z)

        opacities = inverse_sigmoid(0.1 * torch.ones((M, 1), dtype=torch.float, device="cuda"))

        new_xyz = nn.Parameter(new_points.float().contiguous().requires_grad_(True))
        new_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacities = nn.Parameter(opacities.requires_grad_(True))
        new_semantic_feature = nn.Parameter(
            new_semantic_feature.float().unsqueeze(1).contiguous().requires_grad_(True))

        new_trackable_mask = torch.zeros((M,), dtype=torch.bool, device="cuda")
        new_edge_mask = torch.zeros((M,), dtype=torch.bool, device="cuda")

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacities, new_scaling, new_rotation,
                                   new_trackable_mask, new_semantic_feature, new_edge_mask)

        # keep keyframe_idx in sync (densification_postfix does not touch it)
        new_keyframe_idx = torch.ones((M, self.keyframe_idx.shape[1]),
                                      device="cuda", dtype=torch.int32) * keyframe_idx
        self.keyframe_idx = torch.concat([self.keyframe_idx, new_keyframe_idx], dim=0)

        torch.cuda.empty_cache()
        return M

    def get_trackable_gaussians_tensor(self, opacity_th, trackable_kf_idx):
        with torch.no_grad():
            opacity_filter = self.get_opacity > opacity_th
            # target_idxs = torch.logical_and(opacity_filter.squeeze(-1), self.trackable_mask)
            
            idx_filter = self.keyframe_idx > trackable_kf_idx
            target_idxs = torch.logical_and(torch.logical_and(self.trackable_mask, idx_filter.squeeze(-1)), opacity_filter.squeeze(-1))
            
            target_points = self.get_xyz[target_idxs]
            target_rots = self.get_rotation[target_idxs]
            target_scales = self.get_scaling[target_idxs]
            target_kf_idxs = self.keyframe_idx[target_idxs]
            
            edge_filter = self.edge_mask[target_idxs] == 1
            
            return target_points.cpu(), target_rots.cpu(), target_scales.cpu(), target_kf_idxs.cpu(), edge_filter.cpu()

    def apply_rotation_to_quaternions(self, selected_quat, transform_R):
        """
        selected_quat: [N,4] (w,x,y,z)
        transform_R: [3,3]
        """
        # R -> quaternion conversion
        def rot_to_quat(R):
            tr = R.trace()
            if tr > 0:
                S = torch.sqrt(tr + 1.0) * 2
                qw = 0.25 * S
                qx = (R[2,1] - R[1,2]) / S
                qy = (R[0,2] - R[2,0]) / S
                qz = (R[1,0] - R[0,1]) / S
            else:
                if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
                    S = torch.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                    qw = (R[2,1] - R[1,2]) / S
                    qx = 0.25 * S
                    qy = (R[0,1] + R[1,0]) / S
                    qz = (R[0,2] + R[2,0]) / S
                elif R[1,1] > R[2,2]:
                    S = torch.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                    qw = (R[0,2] - R[2,0]) / S
                    qx = (R[0,1] + R[1,0]) / S
                    qy = 0.25 * S
                    qz = (R[1,2] + R[2,1]) / S
                else:
                    S = torch.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                    qw = (R[1,0] - R[0,1]) / S
                    qx = (R[0,2] + R[2,0]) / S
                    qy = (R[1,2] + R[2,1]) / S
                    qz = 0.25 * S
            return torch.tensor([qw,qx,qy,qz], device=R.device, dtype=R.dtype)

        qR = rot_to_quat(transform_R)  # [4]

        # quaternion product (qR ⊗ Q)
        # qR: [4], selected_quat: [N,4]
        w1, x1, y1, z1 = qR
        w2, x2, y2, z2 = selected_quat.T

        new_q = torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=1)

        # normalize to unit quaternion
        new_q = new_q / new_q.norm(dim=1, keepdim=True)

        return new_q

    def transform_gaussians(self, kf_idx, transform):
        select_filter = self.keyframe_idx == kf_idx
        select_filter = select_filter.squeeze(-1)
        
        all_xyz = self._xyz.clone()  # [N,3]
        selected_xyz = all_xyz[select_filter]
        
        all_quats = self.get_rotation.clone()
        selected_quats = all_quats[select_filter]

        transform = torch.tensor(transform, dtype=selected_xyz.dtype).cuda()
        transform_R = transform[:3,:3]
        transform_t = transform[:3,3]

        transformed_xyz = (transform_R @ selected_xyz.T).T + transform_t.unsqueeze(0)

        transformed_quats = self.apply_rotation_to_quaternions(selected_quats, transform_R)

        all_xyz[select_filter] = transformed_xyz
        optimizable_tensors_xyz = self.replace_tensor_to_optimizer(all_xyz, "xyz")

        all_quats[select_filter] = transformed_quats
        optimizable_tensors_quats = self.replace_tensor_to_optimizer(all_quats, "rotation")

        self._xyz = optimizable_tensors_xyz["xyz"]
        self._rotation = optimizable_tensors_quats["rotation"]

    def get_gaussians_tensor(self):
        '''
        get all gaussians from GS map
        return points, rots, scales, trackable_mask, kf_idxs
        '''
        with torch.no_grad():
            target_idxs = self.trackable_mask
            
            target_points = self.get_xyz
            target_rots = self.get_rotation
            target_scales = self.get_scaling
            
            return target_points.cpu(), target_rots.cpu(), target_scales.cpu(), target_idxs.cpu(), self.keyframe_idx.cpu()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._semantic_feature], 'lr':training_args.semantic_feature_lr, "name": "semantic_feature"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    

    def training_update(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._semantic_feature], 'lr':training_args.semantic_feature_lr, "name": "semantic_feature"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._semantic_feature.shape[1]*self._semantic_feature.shape[2]):  
            l.append('semantic_{}'.format(i))
        return l

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_unreliable_opacity(self, filter):
        opacities_new = self._opacity.clone()
        opacities_new[filter] = inverse_sigmoid(torch.min(self.get_opacity[filter], torch.ones_like(self.get_opacity[filter])*0.01))

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_visible_opacity(self, visibility_filter):
        def func(x):
            mean = 0.5
            result = 2*mean * 1/(1 + torch.exp(-10*(x-(mean))))
            return torch.min(x, result)
            # return torch.clip(result, min=None, max=0.99)
        
        def func2(x, mean):
            mean = 0.7
            return 1.2 * 1/(1 + torch.exp(-5*(x-(mean))))
        
        def func3(x, mean):
            return 2. * 1/(1 + torch.exp(-2.*(x))) - 1
        
        def func4(x):
            # return 0.9 * x
            # return torch.relu(1.1*x - 0.1) + 0.01
            return torch.log(x+1.)
        
        opacities_new = self._opacity
        # visible_opacity = self.get_opacity[visibility_filter].detach().cpu().numpy()
        large_gaussians = self.get_scaling.max(dim=1).values > 0.03
        very_large_gaussians = self.get_scaling.max(dim=1).values > 0.07
        mask = torch.logical_and(visibility_filter, large_gaussians)
        # mask = visibility_filter
        # plt.hist(visible_opacity, bins=np.arange(0.,1.0,0.005))
        # plt.show()
        
        # opacities_new[visibility_filter] = inverse_sigmoid(torch.min(self.get_opacity[visibility_filter], torch.ones_like(self.get_opacity[visibility_filter])*0.01))
        # opacities_new[visibility_filter] = inverse_sigmoid(torch.min(self.get_opacity[visibility_filter], func(self.get_opacity[visibility_filter])))
        opacities_new[mask] = inverse_sigmoid(torch.min(self.get_opacity[mask], func4(self.get_opacity[mask])))
        # opacities_new[mask] = inverse_sigmoid(torch.min(self.get_opacity[mask], torch.ones_like(self.get_opacity[mask])*0.01))
        # opacities_new[large_gaussians] = inverse_sigmoid(torch.min(self.get_opacity[large_gaussians], func4(self.get_opacity[large_gaussians])))
        # opacities_new[very_large_gaussians] = inverse_sigmoid(torch.min(self.get_opacity[very_large_gaussians], func4(self.get_opacity[very_large_gaussians])))


        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def reset_visible_opacity2(self, visibility_filter):
        # like dropout? 
        # or decay opacities of large gaussians
        
        opacities_new = self._opacity
        visible_opacity = self.get_opacity[visibility_filter].detach().cpu().numpy()
        # plt.hist(visible_opacity, bins=np.arange(0.,1.0,0.005))
        # plt.show()
        # print(f"Opacity mean : {np.mean(visible_opacity)}")
        
        # opacities_new[visibility_filter] = inverse_sigmoid(torch.min(self.get_opacity[visibility_filter], torch.ones_like(self.get_opacity[visibility_filter])*0.01))
        # opacities_new[visibility_filter] = inverse_sigmoid(torch.min(self.get_opacity[visibility_filter], func4(self.get_opacity[visibility_filter], np.mean(visible_opacity))))
        
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        
        count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("semantic_"))
        semantic_feature = np.stack([np.asarray(plydata.elements[0][f"semantic_{i}"]) for i in range(count)], axis=1) 
        semantic_feature = np.expand_dims(semantic_feature, axis=-1) 

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic_feature = nn.Parameter(torch.tensor(semantic_feature, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.trackable_mask = self.trackable_mask[valid_points_mask]
        self.edge_mask = self.edge_mask[valid_points_mask]
        
        try:
            self.keyframe_idx = self.keyframe_idx[valid_points_mask]
        except:
            pass


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_trackable_mask, new_semantic_feature, new_edge_mask):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "semantic_feature": new_semantic_feature}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic_feature = optimizable_tensors["semantic_feature"] 

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        self.trackable_mask = torch.concat([self.trackable_mask, new_trackable_mask], dim=0)
        self.edge_mask = torch.concat([self.edge_mask, new_edge_mask], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        #torch.cuda.empty_cache()
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        if scene_extent != None:
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        #torch.cuda.empty_cache()
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_trackable_mask = self.trackable_mask[selected_pts_mask].repeat(N)
        new_semantic_feature = self._semantic_feature[selected_pts_mask].repeat(N,1,1)
        new_edge_mask = self.edge_mask[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_trackable_mask, new_semantic_feature, new_edge_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        #torch.cuda.empty_cache()
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        #torch.cuda.empty_cache()
        if scene_extent != None:
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                    torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
                                              
        #torch.cuda.empty_cache()
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_trackable_mask = self.trackable_mask[selected_pts_mask]
        new_semantic_feature = self._semantic_feature[selected_pts_mask] 
        new_edge_mask = self.edge_mask[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_trackable_mask, new_semantic_feature, new_edge_mask)
        #torch.cuda.empty_cache()

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        #torch.cuda.empty_cache()
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        #torch.cuda.empty_cache()
        self.densify_and_clone(grads, max_grad, extent)
        #torch.cuda.empty_cache()
        self.densify_and_split(grads, max_grad, extent)
        #torch.cuda.empty_cache()

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            if extent != None:
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            else:
                prune_mask = torch.logical_or(prune_mask, big_points_vs)
        self.prune_points(prune_mask)

    def densify_only(self, max_grad, extent):
        #torch.cuda.empty_cache()
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # if SLAM mode, extent = None

        #torch.cuda.empty_cache()
        self.densify_and_clone(grads, max_grad, extent)
        #torch.cuda.empty_cache()
        self.densify_and_split(grads, max_grad, extent)
        #torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    def prune_large_and_transparent(self, min_opacity, extent):
        
        #torch.cuda.empty_cache()
        # grads = self.xyz_gradient_accum / self.denom
        # grads[grads.isnan()] = 0.0
        # plt.hist(self.get_scaling.max(dim=1).values.detach().cpu().numpy()) # , bins=np.arange(0.,1.0,0.005)
        # plt.show()
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        if extent != None:
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)


    def prune_large_transparent_and_lang(self, min_opacity, extent, dist_threshold, sim_threshold, max_sample_size=18000, sample_ratio=0.60, k_nearest=180):
        """
        GPU-only pruning:
        1) Low opacity
        2) Abnormally large scaling
        3) Redundant meaning (nearby + high semantic similarity), vectorized
        """
        import torch
        import torch.nn.functional as F

        # ------------ Hyperparams -------------
        eps = 1e-8
        # --------------------------------------

        # ------------ Basic tensors (stay on GPU) -----------
        xyz      = self.get_xyz.detach()               # (N, 3)
        opacity  = self.get_opacity.detach().squeeze() # (N,)
        scaling  = self.get_scaling.detach()           # (N, d_scale)
        device   = xyz.device
        N        = xyz.shape[0]

        # 1) transparent / large masks  (O(N))
        transparent_mask = (opacity < min_opacity)
        if extent is not None:
            large_mask = scaling.max(dim=1).values > (0.1 * extent)
        else:
            large_mask = torch.zeros_like(transparent_mask, dtype=torch.bool)

        # ------------ Sampling ----------------
        if N <= max_sample_size:
            sample_size = N
        else:
            sample_size = min(max_sample_size, int(N * sample_ratio))

        sample_indices = torch.randperm(N, device=device)[:sample_size]
        sample_xyz     = xyz[sample_indices]  # (S, 3)

        # ------------ Semantic features -------
        sem = self._semantic_feature.detach()  # (N, D) or (N,1,D)/(N,D,1)
        if sem.ndim == 3:
            if sem.shape[1] == 1:
                sem = sem.squeeze(1)
            elif sem.shape[2] == 1:
                sem = sem.squeeze(2)
            else:
                sem = sem.reshape(sem.shape[0], -1)
        sample_sem = sem[sample_indices]                     # (S, D)
        sample_sem = F.normalize(sample_sem, dim=1, eps=eps) # cosine norm

        # ------------ KNN (torch.cdist + topk) ----------------
        dist_mat = torch.cdist(sample_xyz, sample_xyz, p=2)          # (S, S)
        dists, nn_idx = torch.topk(dist_mat, k=k_nearest + 1, dim=1, largest=False)
        dists  = dists[:, 1:]     # (S, k), remove self
        nn_idx = nn_idx[:, 1:]    # (S, k)

        # ------------ Cosine similarities (vectorized) --------
        sims_full = sample_sem @ sample_sem.t()                       # (S, S)
        row_ids   = torch.arange(sample_size, device=device).unsqueeze(1).expand_as(nn_idx)
        sims      = sims_full[row_ids, nn_idx]                        # (S, k)

        redundant_mask_local = (dists < dist_threshold) & (sims > sim_threshold)  # (S, k)

        if redundant_mask_local.any():
            redundant_local  = nn_idx[redundant_mask_local]
            redundant_global = sample_indices[redundant_local].unique()
        else:
            redundant_global = torch.empty(0, dtype=torch.long, device=device)

        redundant_meaning_mask = torch.zeros(N, dtype=torch.bool, device=device)
        if redundant_global.numel() > 0:
            redundant_meaning_mask[redundant_global] = True

        # ------------ Combine & prune ----------
        prune_mask = transparent_mask | large_mask | redundant_meaning_mask
        self.prune_points(prune_mask)


    def lang_feature_smoothness_loss(self, sample_size=2000, k_nearest=8,
                                     sigma_x=0.1, sigma_c=0.2, eps=1e-8):
        """Bilateral kNN language-feature smoothness regularizer (A3).

        Encourages Gaussians that are close in 3D space AND similar in color to
        share similar language features, denoising the 3D language field.
        Geometry and color are detached so this term only shapes the per-Gaussian
        semantic features (which carry gradients here).

        Returns a scalar loss (zero tensor if there are too few Gaussians).
        """
        import torch
        import torch.nn.functional as F

        N = self._xyz.shape[0]
        if N < (k_nearest + 2):
            return self._xyz.new_zeros(())

        device = self._xyz.device
        if N > sample_size:
            sample_indices = torch.randperm(N, device=device)[:sample_size]
        else:
            sample_indices = torch.arange(N, device=device)
        S = sample_indices.shape[0]

        # geometry / appearance used only to build affinity weights -> detached
        xyz = self.get_xyz[sample_indices].detach()                       # (S, 3)
        color = self._features_dc[sample_indices].detach().reshape(S, -1)  # (S, 3)

        # per-Gaussian language feature (keep gradient)
        sem = self._semantic_feature[sample_indices]
        if sem.ndim == 3:
            sem = sem.reshape(S, -1)
        sem_n = F.normalize(sem, dim=1, eps=eps)                          # (S, D)

        # kNN in 3D space
        dist_mat = torch.cdist(xyz, xyz, p=2)                             # (S, S)
        dists, nn_idx = torch.topk(dist_mat, k=k_nearest + 1, dim=1, largest=False)
        dists = dists[:, 1:]                                              # (S, k)
        nn_idx = nn_idx[:, 1:]                                            # (S, k)

        # bilateral affinity: close in space AND similar in color
        color_nn = color[nn_idx]                                         # (S, k, 3)
        color_diff = (color.unsqueeze(1) - color_nn).pow(2).sum(dim=2)    # (S, k)
        w_spatial = torch.exp(-(dists ** 2) / (sigma_x ** 2 + eps))
        w_color = torch.exp(-color_diff / (sigma_c ** 2 + eps))
        w = (w_spatial * w_color).detach()                               # (S, k)

        # cosine dissimilarity between each sample and its neighbours
        sem_nn = sem_n[nn_idx]                                           # (S, k, D)
        cos_sim = (sem_n.unsqueeze(1) * sem_nn).sum(dim=2)                # (S, k)
        diss = 1.0 - cos_sim                                             # (S, k)

        loss = (w * diss).sum() / w.sum().clamp_min(eps)
        return loss


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        semantic_feature = self._semantic_feature.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() 

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantic_feature), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
