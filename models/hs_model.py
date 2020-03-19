"""
This file provides a wrapper around resnet and vote_fc and is useful for inference since it fuses both forward passes in one.
"""
import torch
import torch.nn as nn

from models import get_pose_net, SMPL
# from models.resnet import resnet50
# from models.sc_layers_share_global6d import SCFC_Share
from models import HMR_HR
# from models.geometric_layers import orthographic_projection, rodrigues, quat2mat
import numpy as np
class HSModel(nn.Module):

    def __init__(self, cfg, is_train, smpl_mean_params, pretrained_checkpoint=None):
        super(HSModel, self).__init__()
        self.hrnet = get_pose_net(cfg, is_train)   

        # hidden_neuron_list = [4096,4096]

        self.hmr_hr = HMR_HR(cfg, smpl_mean_params)
 
        self.smpl = SMPL()

        if pretrained_checkpoint is not None:
            checkpoint = torch.load(pretrained_checkpoint)
            try:
                self.hrnet.load_state_dict(checkpoint['hrnet'])
            except KeyError:
                print('Warning: hrnet was not found in checkpoint')
            try:
                self.hmr_hr.load_state_dict(checkpoint['hmr_hr'])
            except KeyError:
                print('Warning: hmr_hr was not found in checkpoint')

    def forward(self, image):
        """Fused forward pass for the 2 networks
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed SMPL shape: size = (B, 6890, 3)
            Weak-perspective camera: size = (B, 3)
            SMPL pose parameters (as rotation matrices): size = (B, 24, 3, 3)
            SMPL shape parameters: size = (B, 10)
        """
        batch_size = image.shape[0]
        with torch.no_grad():
            outputs = self.hrnet(image)

            pred_rotmat, pred_shape, pred_cam  = self.hmr_hr(outputs)
            # pred_camera = pred_camera_with_global_rot[:,:3] #(B,3)
            # pred_global_rot = pred_camera_with_global_rot[:,3:][:,None,:] #(B,1,4)
            # pose_cube = pred_theta.view(-1, 4) # (batch_size * 24,  4)
            # R = quat2mat(pose_cube).view(batch_size, 23, 3, 3)
            # pred_rotmat = R.view(batch_size, 23, 3, 3)
            # pred_global_rot = pred_global_rot.view(batch_size, 1, 3, 3)
            # pred_rotmat = torch.cat((pred_global_rot,pred_rotmat),dim=1) #(B,24,3,3)
            # pred_vertices = self.smpl(pred_rotmat, pred_beta)

        return outputs, pred_rotmat, pred_shape, pred_cam
