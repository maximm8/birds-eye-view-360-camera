import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from utils import *
import target

if __name__ == '__main__':

    # set parameters
    data_folder = 'data/'
    calib_params_filename = f'{data_folder}/calib/calib_params.json'
    calib_img_files = glob.glob(f'{data_folder}/calib/cam1_color_*.png')
    img_files = glob.glob(f'{data_folder}/imgs/cam1_color_*.png')
    
    #1 get camera params
    if os.path.exists(calib_params_filename):
        K, kc, w, h = load_camera_params(calib_params_filename)
    else:
        K, kc, w, h = calibrate_camera(calib_img_files, target.PatternSize, target.SquareSize)

    K2, kc2, map1, map2 = get_undistorted_intrinsics(K, kc, w, h)

    #2 find ground location with respect to the camera
    # load images
    imgs = [cv2.imread(fname) for fname in img_files]
    # estimate ground pose
    img_ground = imgs[0].copy()
    found, corners, detected_target = detect_chessboard_pattern(img_ground, target.PatternSize)    
    pattern_points = create_target_points(target.PatternSize, target.SquareSize)
    R, T, pts_cam_space, rvec, tvec, R0, T0 = calc_target_pose(corners, pattern_points, K, kc)
    # cv2.imshow('ground', detected_target)
    # cv2.waitKey()

    #3 create top view 
    # project plane to image 
    offsetx, offsety = 200, 500
    # offsetx, offsety = 50, 50
    xmin, xmax = -offsetx, (target.PatternSize[0]-1)*target.SquareSize+offsetx*1
    ymin, ymax = -offsety, (target.PatternSize[1]-1)*target.SquareSize+offsety*1
    pix_x, pix_y, w_new, h_new = project_plane_to_image(xmin, xmax, ymin, ymax, K2, kc2, rvec, tvec, sampling_rate=2)

    for img in imgs:
        
        # img_proj_pts = draw_project_pts(img, pix_x, pix_y, map1, map2)
        # cv2.imshow('projected plane points', img_proj_pts)

        # sample projected plane points from image
        img2 = bilinear_interpolate(img, pix_x, pix_y , w_new, h_new)        
        
        cv2.imshow('input', img)
        cv2.imshow('top view', img2)
        cv2.waitKey()

 
        
