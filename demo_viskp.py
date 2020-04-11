"""
Demo code

To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

Example with OpenPose detection .json
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --openpose=examples/im1010_openpose.json
```
Example with predefined Bounding Box
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --bbox=examples/im1010_bbox.json
```
Example with cropped and centered image
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png
```

Running the previous command will save the results in ```examples/im1010_{shape,shape_side}.png```. The file ```im1010_shape.png``` shows the overlayed reconstruction of human shape. We also render a side view, saved in ```im1010_shape_side.png```.
"""

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import os
from tqdm import tqdm

from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer import Renderer
import config
import constants
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
# parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')

def draw_full_skeleton(input_image, joints, all_joints=None, draw_edges=True, vis=None, all_vis=None, radius=None):
    """
    joints is 3 x 19. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    """

    
    joints = joints + 112
    print(joints)

    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([0, 0, 255]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]) #
    }

    image = input_image.copy()                                   
    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    if all_joints is not None: 
        if all_joints.shape[0] != 2:
            all_joints = all_joints.T
        all_joints = np.round(all_joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white'
    ]

    all_jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'purple', 'purple', 'red', 'green', 'green', 'green', 'white', 'white' ,'white', 'white'
    ]

    # draw all ketpoints
    if joints is not None: 
        print(joints.shape[1])
        for i in range(joints.shape[1]):
            point = joints[:, i]
            # If invisible skip
            # if all_vis is not None and all_vis[i] == 0:
            #     continue
            if draw_edges:
                # print(radius)
                # print(point)
                # cv2.circle(image, (100, 60), 3, (0, 0, 213), -1)
                cv2.circle(image, (point[0], point[1]), 2, colors['blue'].tolist(),
                        2)
                # cv2.circle(image, (point[0], point[1]), radius-1, colors['blue'].tolist(),
                #         -1)
                # cv2.circle(image, (point[0], point[1]), radius - 2,
                #         colors['blue'].tolist(), -1)
            else:
                # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
                cv2.circle(image, (point[0], point[1]), radius - 1,
                        colors['blue'].tolist(), 1)
                # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
    return image


def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    print(img.shape)
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

if __name__ == '__main__':
    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)

    # imgs_path = '/project/hpn_yyy/datasets/preprocess/min_imgs_up-3d'  
    imgs_path = 'crop_vis'  

    imgpath_list = []   
    for dirpath, dirnames, filenames in os.walk(imgs_path):  
        for f in filenames :  
            if os.path.splitext(f)[1] == '.png' or os.path.splitext(f)[1] == '.jpg':  
                imgpath_list.append(os.path.join(dirpath, f)) 

    for imgpath in tqdm(imgpath_list):
        # Preprocess input image and generate predictions
        img, norm_img = process_image(imgpath, args.bbox, args.openpose, input_res=constants.IMG_RES)
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
            pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints
            
        # Calculate camera parameters for rendering
        camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        batch_size = 1
        camera_center = torch.zeros(batch_size, 2, device=device)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                    rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                                                    translation=camera_translation,
                                                    focal_length=constants.FOCAL_LENGTH,
                                                    camera_center=camera_center)
        # print(pred_keypoints_2d.shape)
        kp_img = draw_full_skeleton(img.permute(1,2,0).cpu().numpy(), pred_keypoints_2d[0][25:,:].cpu().numpy())
        # cv2.imwrite('test_kp.jpg', kp_img[:,:,::-1])
        camera_translation = camera_translation[0].cpu().numpy()
        pred_vertices = pred_vertices[0].cpu().numpy()
        img = img.permute(1,2,0).cpu().numpy()



        
        # Render parametric shape
        img_shape = renderer(pred_vertices, camera_translation, img)
        
        # Render side views
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
        center = pred_vertices.mean(axis=0)
        rot_vertices = np.dot((pred_vertices - center), aroundy) + center
        
        # Render non-parametric shape
        img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img))

        outfile = imgpath.split('.')[0] if args.outfile is None else args.outfile

        # Save reconstructions
        cv2.imwrite(outfile + '_kp.jpg', kp_img[:,:,::-1])
        cv2.imwrite(outfile + '_shape.png', 255 * img_shape[:,:,::-1])
        cv2.imwrite(outfile + '_shape_side.png', 255 * img_shape_side[:,:,::-1])