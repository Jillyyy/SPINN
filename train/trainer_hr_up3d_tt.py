import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2

from datasets import UP3DDataset
from models import HSModel, SMPL
from smplify import SMPLify
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation
from utils.renderer import Renderer
from utils import BaseTrainer
from utils.loss import JointsMSELoss
from utils.evaluate import accuracy
from utils.vis import save_debug_images
from utils.pose_utils import reconstruction_error
from datasets.base_JointsDataset import BaseJointsDataset

import config
import constants
from .fits_dict import FitsDict

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class HSTrainer(BaseTrainer):
    
    def init_fn(self):
        self.train_ds = UP3DDataset(self.options, self.cfg, ignore_3d=self.options.ignore_3d, is_train=True)
        # create test dataset
        self.test_ds_name = 'up-3d'
        self.test_ds = BaseJointsDataset(self.options, self.cfg, self.test_ds_name, is_train=False)
        self.test_num_workers = 8
        self.test_batch_size = 32
        self.test_data_loader = DataLoader(self.test_ds, batch_size=self.test_batch_size\
            , shuffle=False, num_workers=self.test_num_workers)


        self.model = HSModel(self.cfg, is_train = True, smpl_mean_params=config.SMPL_MEAN_PARAMS).to(self.device)
        # print(self.model)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=0)
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_hm_keypoints = JointsMSELoss(use_target_weight=self.cfg.LOSS.USE_TARGET_WEIGHT).to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH

        # Initialize SMPLify fitting module
        self.smplify = SMPLify(step_size=1e-2, batch_size=self.options.batch_size, num_iters=self.options.num_smplify_iters, focal_length=self.focal_length)
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        # Load dictionary of fits
        self.fits_dict = FitsDict(self.options, self.train_ds)

        # Create renderer
        self.renderer = Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)

    def finalize(self):
        self.fits_dict.save()

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def train_step(self, input_batch):
        self.model.train()

        # Get data from the batch
        images = input_batch['img'] # input image
        gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints
        gt_pose = input_batch['pose'] # SMPL pose parameters
        gt_betas = input_batch['betas'] # SMPL beta parameters
        gt_joints = input_batch['pose_3d'] # 3D pose
        has_smpl = input_batch['has_smpl'].bool() # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].byte() # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'] # index of example inside its dataset
        target = input_batch['target']
        target_weight = input_batch['target_weight']
        batch_size = images.shape[0]

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices
        print(gt_vertices.shape)

        # Get current best fits from the dictionary
        opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)
        opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:,3:], global_orient=opt_pose[:,:3])
        opt_vertices = opt_output.vertices
        opt_joints = opt_output.joints


        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)

        opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)


        opt_joint_loss = self.smplify.get_fitting_loss(opt_pose, opt_betas, opt_cam_t,
                                                       0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device),
                                                       gt_keypoints_2d_orig).mean(dim=-1)

        # Feed images in the network to predict camera and SMPL parameters
        outputs, pred_rotmat, pred_betas, pred_camera = self.model(images)
        
        # add jointsMSELoss
        if isinstance(outputs, list):
            loss_jointsMSE = self.criterion_hm_keypoints(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss_jointsMSE += self.criterion_hm_keypoints(output, target, target_weight)
        else:
            output = outputs
            loss_jointsMSE = self.criterion_hm_keypoints(output, target, target_weight)

        # loss_jointsMSE = self.criterion_hm_keypoints()
        _, _, _, pred = accuracy(output.detach().cpu().numpy(),
                                 target.detach().cpu().numpy())
        # print(pred.shape)
        if self.step_count % 100 == 0:
            # prefix = 'test_%d' % self.step_count
            prefix = 'test_0'
            save_debug_images(self.cfg, images, input_batch, target, pred*4, output,
                              prefix)

        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)


        camera_center = torch.zeros(batch_size, 2, device=self.device)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length,
                                                   camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

        if self.options.run_smplify:

            # Convert predicted rotation matrices to axis-angle
            pred_rotmat_hom = torch.cat([pred_rotmat.detach().view(-1, 3, 3).detach(), torch.tensor([0,0,1], dtype=torch.float32,
                device=self.device).view(1, 3, 1).expand(batch_size * 24, -1, -1)], dim=-1)
            pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch_size, -1)
            # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
            pred_pose[torch.isnan(pred_pose)] = 0.0

            # Run SMPLify optimization starting from the network prediction
            new_opt_vertices, new_opt_joints,\
            new_opt_pose, new_opt_betas,\
            new_opt_cam_t, new_opt_joint_loss = self.smplify(
                                        pred_pose.detach(), pred_betas.detach(),
                                        pred_cam_t.detach(),
                                        0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device),
                                        gt_keypoints_2d_orig)
            new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)

            # Will update the dictionary for the examples where the new loss is less than the current one
            update = (new_opt_joint_loss < opt_joint_loss)
            

            opt_joint_loss[update] = new_opt_joint_loss[update]
            opt_vertices[update, :] = new_opt_vertices[update, :]
            opt_joints[update, :] = new_opt_joints[update, :]
            opt_pose[update, :] = new_opt_pose[update, :]
            opt_betas[update, :] = new_opt_betas[update, :]
            opt_cam_t[update, :] = new_opt_cam_t[update, :]


            self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu(), update.cpu())] = (opt_pose.cpu(), opt_betas.cpu())

        else:
            update = torch.zeros(batch_size, device=self.device).byte()

        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.

        # print(has_smpl)
        # Replace the optimized parameters with the ground truth parameters, if available
        opt_vertices[has_smpl, :, :] = gt_vertices[has_smpl, :, :]
        opt_cam_t[has_smpl, :] = gt_cam_t[has_smpl, :]
        opt_joints[has_smpl, :, :] = gt_model_joints[has_smpl, :, :]
        opt_pose[has_smpl, :] = gt_pose[has_smpl, :]
        opt_betas[has_smpl, :] = gt_betas[has_smpl, :]


        # Assert whether a fit is valid by comparing the joint loss with the threshold
        valid_fit = (opt_joint_loss < self.options.smplify_threshold).to(self.device)
        # Add the examples with GT parameters to the list of valid fits
        valid_fit = valid_fit | has_smpl

        opt_keypoints_2d = perspective_projection(opt_joints,
                                                  rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                  translation=opt_cam_t,
                                                  focal_length=self.focal_length,
                                                  camera_center=camera_center)


        opt_keypoints_2d = opt_keypoints_2d / (self.options.img_res / 2.)


        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight)

        # Compute 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d)

        # Per-vertex loss for the shape
        loss_shape = self.shape_loss(pred_vertices, opt_vertices, valid_fit)

        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = loss_jointsMSE +\
               self.options.shape_loss_weight * loss_shape +\
               self.options.keypoint_loss_weight * loss_keypoints +\
               self.options.keypoint_loss_weight * loss_keypoints_3d +\
               loss_regr_pose + self.options.beta_loss_weight * loss_regr_betas +\
               ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()
        # loss = loss_jointsMSE
        loss *= 60
        loss.requires_grad = True


        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': pred_vertices.detach(),
                  'opt_vertices': opt_vertices,
                  'pred_cam_t': pred_cam_t.detach(),
                  'opt_cam_t': opt_cam_t}
        losses = {'loss': loss.detach().item(),
                  'loss_jointsMSE': loss_jointsMSE.detach().item(),
                  'loss_keypoints': loss_keypoints.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                  'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas.detach().item(),
                  'loss_shape': loss_shape.detach().item()}

        return output, losses

    def train_summaries(self, input_batch, output, losses):
        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        pred_vertices = output['pred_vertices']
        opt_vertices = output['opt_vertices']
        pred_cam_t = output['pred_cam_t']
        opt_cam_t = output['opt_cam_t']
        images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images)
        images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, images)
        self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
        self.summary_writer.add_image('opt_shape', images_opt, self.step_count)
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)

    def test_summaries(self, mpjpe, recon_err, shape_err):
        print()
        print('*** Test Results on %s***' %self.test_ds_name)
        print('MPJPE (NonParam): ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error (NonParam): ' + str(1000 * recon_err.mean()))
        print('Shape Error (NonParam): ' + str(1000 * shape_err.mean()))
        print()
        # self.summary_writer.add_image('min_mpjpe_imgs', all_min_rend_imgs, self.step_count)
        # self.summary_writer.add_image('max_mpjpe_imgs', all_max_rend_imgs, self.step_count)
        self.summary_writer.add_scalars('test_keypoints_3d', {'mpjpe': 1000 * mpjpe.mean()}, self.step_count)
        self.summary_writer.add_scalars('test_keypoints_3d', {'recon': 1000 * recon_err.mean()}, self.step_count)
        self.summary_writer.add_scalars('test_keypoints_3d', {'shape_err': 1000 * shape_err.mean()}, self.step_count)

    def test(self):
        self.model.eval()
        batch_size = self.test_batch_size
        device = self.device

        # Pose metrics
        # MPJPE and Reconstruction error for the non-parametric and parametric shapes
        mpjpe = np.zeros(len(self.test_ds))
        recon_err = np.zeros(len(self.test_ds))
        shape_err = np.zeros(len(self.test_ds))

        min_mpjpes = []
        max_mpjpes = []
        min_rend_imgs = []
        max_rend_imgs = []
        # tt = 0
        for step, batch in enumerate(tqdm(self.test_data_loader, desc='Eval', total=len(self.test_data_loader))):

            # tt += 1
            # if tt > 20:
            #     break
            # Get ground truth annotations from the batch
            gt_pose = batch['pose'].to(device)
            gt_betas = batch['betas'].to(device)
            gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
            gt_model_joints = gt_out.joints[:,:24,:]
            gt_vertices = gt_out.vertices
            # gt_vertices = self.smpl(gt_pose, gt_betas)
            images = batch['img'].to(device)
            curr_batch_size = images.shape[0]

            with torch.no_grad():
                outputs, pred_rotmat, pred_betas, pred_camera = self.model(images)
                pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
                pred_vertices = pred_output.vertices
                pred_joints = pred_output.joints[:,:24,:]
                # feat = self.resnet(images)
                # pred_theta, pred_beta, pred_camera_with_global_rot, nnz_beta, recover_theta_norm = self.sc_fc(feat)
                # pred_camera = pred_camera_with_global_rot[:,:3] #(B,3)
                # pred_global_rot = pred_camera_with_global_rot[:,3:][:,None,:] #(B,1,9)
                # # pred_theta = torch.cat((pred_global_rot,pred_theta),dim=1) #(B,24,4)
                # pose_cube = pred_theta.view(-1, 4) # (batch_size * 23,  4)
                # R = quat2mat(pose_cube).view(curr_batch_size, 23, 3, 3)
                # pred_rotmat = R.view(curr_batch_size, 23, 3, 3)
                # pred_global_rot = pred_global_rot.view(curr_batch_size, 1, 3, 3)
                # pred_rotmat = torch.cat((pred_global_rot,pred_rotmat),dim=1) #(B,24,3,3)
                # pred_vertices = self.smpl(pred_rotmat, pred_beta) 


            # # Regressor broadcasting
            # J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)

            # # Get 14 ground truth joints
            # gt_keypoints_3d = batch['pose_3d'].cuda()
            # gt_keypoints_3d = gt_keypoints_3d[:, cfg.J24_TO_J14, :-1]

            # # Get 14 predicted joints 
            # pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            # pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            # pred_keypoints_3d = pred_keypoints_3d[:, cfg.H36M_TO_J14, :]
            # pred_keypoints_3d = pred_keypoints_3d - pred_pelvis 
            

            # Compute error metrics

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_joints - gt_model_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_joints.cpu().numpy(), gt_model_joints.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # Shape error
            se = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            shape_err[step * batch_size:step * batch_size + curr_batch_size] = se

            # # visualize min & max
            # min_mpjpe = np.min(error)
            # max_mpjpe = np.max(error)
            # # print(type(min_mpjpe))
            # min_mpjpes.append(min_mpjpe)
            # max_mpjpes.append(max_mpjpe)
            # # print(np.where(error==np.min(error)))
            # min_idx = int(np.where(error==np.min(error))[0][0])
            # max_idx = int(np.where(error==np.max(error))[0][0])

            # gt_keypoints_2d = batch['keypoints'].cpu().numpy()
            # gt_endpoint_2d = batch['endpoint_2d'].cpu().numpy()
            # gt_allpoints_2d = np.concatenate((gt_keypoints_2d,gt_endpoint_2d),axis=1)
            # pred_keypoints_3d_1 = self.smpl.get_joints(pred_vertices)
            # pred_keypoints_2d = orthographic_projection(pred_keypoints_3d_1, pred_camera)[:, :, :2]
            

            # # visualize min
            # img = batch['img_orig'][min_idx].cpu().numpy().transpose(1,2,0)
            # # Get LSP keypoints from the full list of keypoints
            # gt_keypoints_2d_ = gt_keypoints_2d[min_idx, self.to_lsp]
            # gt_allpoints_2d_ = gt_allpoints_2d[min_idx, self.to_all]
            # pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[min_idx, self.to_lsp]
            # # Get GraphCNN and SMPL vertices for the particular example
            # vertices = pred_vertices[min_idx].cpu().numpy()
            # cam = pred_camera[min_idx].cpu().numpy()
            # # Visualize reconstruction and detected pose
            # min_rend_img = visualize_reconstruction_allkp(img, self.options.img_res, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, self.renderer, gt_allpoints_2d_)
            # min_rend_img = min_rend_img.transpose(2,0,1)
            # min_rend_imgs.append(torch.from_numpy(min_rend_img))

            # # visualize max
            # img = batch['img_orig'][max_idx].cpu().numpy().transpose(1,2,0)
            # # Get LSP keypoints from the full list of keypoints
            # gt_keypoints_2d_ = gt_keypoints_2d[max_idx, self.to_lsp]
            # gt_allpoints_2d_ = gt_allpoints_2d[max_idx, self.to_all]
            # pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[max_idx, self.to_lsp]
            # # Get GraphCNN and SMPL vertices for the particular example
            # vertices = pred_vertices[max_idx].cpu().numpy()
            # cam = pred_camera[max_idx].cpu().numpy()
            # # Visualize reconstruction and detected pose
            # max_rend_img = visualize_reconstruction_allkp(img, self.options.img_res, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, self.renderer, gt_allpoints_2d_)
            # max_rend_img = max_rend_img.transpose(2,0,1)
            # max_rend_imgs.append(torch.from_numpy(max_rend_img))

        # min_mpjpes = np.array(min_mpjpes) 
        # max_mpjpes = np.array(max_mpjpes)

        # min_k = 16
        # min_k_idx = min_mpjpes.argsort()[0:min_k]
        # print(min_k_idx)

        # max_k = 16
        # max_k_idx = max_mpjpes.argsort()[::-1][0:max_k]

        # all_min_rend_imgs = []
        # for dd in range(min_k):
        #     all_min_rend_imgs.append(min_rend_imgs[min_k_idx[dd]])
        # all_max_rend_imgs = []
        # for dd in range(max_k):
        #     all_max_rend_imgs.append(max_rend_imgs[max_k_idx[dd]])
        # # all_min_rend_imgs = min_rend_imgs[min_k_idx]
        # # all_max_rend_imgs = max_rend_imgs[max_k_idx]

        
        # all_min_rend_imgs = make_grid(all_min_rend_imgs, nrow=4)
        # all_max_rend_imgs = make_grid(all_max_rend_imgs, nrow=4)


        # out_args = [mpjpe, recon_err, all_min_rend_imgs, all_max_rend_imgs]
        out_args = [mpjpe, recon_err, shape_err]

        return out_args