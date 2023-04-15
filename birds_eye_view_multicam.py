import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from utils import *
import target

if __name__ == '__main__':

    # set parameters
    cams_nb = 2
    data_folder = 'data/'
    
    #1 get camera params
    K, kc, img_size = [], [], []
    K2, kc2, map1, map2 = [], [], [], []
    for cam_ind in range(cams_nb):
        calib_params_filename = f'{data_folder}/calib{cam_ind+1}/calib_params.json'
        calib_img_files = glob.glob(f'{data_folder}/calib{cam_ind+1}/cam1_color_*.png')
        if os.path.exists(calib_params_filename):
            K_, kc_, w_, h_ = load_camera_params(calib_params_filename)
        else:
            K_, kc_, w_, h_ = calibrate_camera(calib_img_files, target.PatternSize, target.SquareSize)

        K.append(K_)
        kc.append(kc_)
        img_size.append((w_,h_))

        K2_, kc2_, map1_, map2_ = get_undistorted_intrinsics(K_, kc_, w_, h_)
        K2.append(K2_)
        kc2.append(kc2_)
        map1.append(map1_)
        map2.append(map2_)
        
    # pattern points offset
    pattern_points = create_target_points(target.PatternSize, target.SquareSize)
    pattern_pts_offset = []
    # set world offset for each target
    pattern_pts_offset.append(np.array([0,0,0]))
    pattern_pts_offset.append(np.array([0, -(83+279.4+198), 0]))

    #2 find ground location with respect to each camera
    # load images
    imgs, rvec, tvec = [], [], []    
    for cam_ind in range(cams_nb):
        img_files = glob.glob(f'{data_folder}/imgs12/cam{cam_ind+1}_color_*.png')
        imgs_ = [cv2.imread(fname) for fname in img_files]
        # estimate ground pose
        img_ground = imgs_[0].copy()
        found, corners, detected_target = detect_chessboard_pattern(img_ground, target.PatternSize)
        pattern_points2 = pattern_points+pattern_pts_offset[cam_ind]
        pattern_points = create_target_points(target.PatternSize, target.SquareSize)
        R_, T_, pts_cam_space_, rvec_, tvec_, R0_, T0_ = calc_target_pose(corners, pattern_points2, K[cam_ind], kc[cam_ind])

        # cv2.imshow('ground', detected_target)
        # cv2.waitKey()

        imgs.append(imgs_)
        rvec.append(rvec_)
        tvec.append(tvec_)

    #3 create top view 
    # project plane to image 
    offsetx1, offsetx2 = 500, 500
    offsety1, offsety2 = 1000, 250
    xmin, xmax = -offsetx1, (target.PatternSize[0]-1)*target.SquareSize+offsetx2*1
    ymin, ymax = -offsety1, (target.PatternSize[1]-1)*target.SquareSize+offsety2*1


    for img1, img2 in zip(imgs[0], imgs[1]): 

        img = [img1, img2]
        img_top = np.zeros((0))
        for cam_ind in range(cams_nb):

            pix_x, pix_y, w_new, h_new = project_plane_to_image(xmin, xmax, ymin, ymax, K2[cam_ind], kc2[cam_ind], rvec[cam_ind], tvec[cam_ind], sampling_rate=1)

            # img_proj_pts = draw_project_pts(img[cam_ind], pix_x, pix_y, map1[cam_ind], map2[cam_ind])
            # cv2.imshow(f'projected plane points', img_proj_pts)

            # sample projected plane points from image
            img2 = bilinear_interpolate(img[cam_ind], pix_x, pix_y , w_new, h_new)        
            if img_top.any() == False: img_top  = np.zeros((h_new, w_new, 3), dtype=np.uint8)
            ind = np.where(img2 !=0)[0]
            img_top[ind] = img2[ind]
            
            cv2.imshow(f'input {cam_ind}', img[cam_ind])

        img_top = cv2.resize(img_top, (int(w_new/2), int(h_new/2)))
        cv2.imshow(f'top view ', img_top)
        cv2.waitKey()

        
