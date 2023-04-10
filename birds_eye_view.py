import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

import utils

def draw_project_pts(img, pix_x, pix_y, map1, map2):
    img2 = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    for px, py in zip(pix_x, pix_y):
        if px>0 and px<img2.shape[1] and py>0 and py<img2.shape[0]:
            cv2.circle(img2, (int(px), int(py)), 5, (0,0,255), 1)
    
    return img2

def detect_chessboard_pattern(img, patter_size):
    detection_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    refine_flags = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001
     
    pattern_found, corners = cv2.findChessboardCorners(img, patter_size, detection_flags)
    if pattern_found:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerSubPix(img_gray, np.float32(corners),(5,5),(-1,-1), refine_flags)
        cv2.drawChessboardCorners(img, patter_size, corners, pattern_found)

    return pattern_found, corners, img

def create_target_points(pattern_size, square_size):
    
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    return pattern_points

def calibrate_camera(files, pattern_size, square_size):

    pattern_points = create_target_points(pattern_size, square_size)
    img_pts = []
    obj_pts = []

    for fname in files:
        img = cv2.imread(fname)

        found, corners, detected_target = detect_chessboard_pattern(img, pattern_size)
        if found:
            img_pts.append(corners)
            obj_pts.append(pattern_points)

            cv2.imshow('target', detected_target)
            cv2.waitKey()

    w, h = img.shape[1], img.shape[0]
    rms, K, kc, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)

    utils.save_camera_params(calib_params_filename, K, kc, w, h)

    return K, kc, w, h

def get_undistorted_intrinsics(K, kc, w, h):
            
    K2, roi = cv2.getOptimalNewCameraMatrix(K, kc, (w,h), 0)
    kc2 = np.zeros_like(kc)
    map1, map2 = cv2.initUndistortRectifyMap(K, kc, np.eye(3), K2, (w,h), cv2.CV_16SC2)

    return K2, kc2, map1, map2

def create_plane_points():
      #3 create top view 
    #define plane limits
    offsetx, offsety = 200, 200
    x0 = -offsetx
    x1 = (PatternSize[0]-1)*SquareSize+offsetx*2
    w_new = abs(x1-x0)

    y0 = -offsety
    y1 = (PatternSize[1]-1)*SquareSize+offsety*2
    h_new = abs(y1-y0)

    # project plane points to image
    ground_pts = utils.create_grid(x0, x1, w_new, y1, y0, h_new)
    ground_img_pts, jac= cv2.projectPoints(ground_pts, rvec, tvec, K2, kc2)
    pix_x, pix_y = ground_img_pts[:,0,0], ground_img_pts[:,0,1]

def project_plane_to_image(xmin, xmax, ymin, ymax, K2, kc2, rvec, tvec, sampling_rate=1):
    
    # get size of a new image
    w_new = abs(xmax-xmin)*sampling_rate
    h_new = abs(ymax-ymin)*sampling_rate

    # project plane points to image
    ground_pts = utils.create_grid(xmax, xmin, w_new, ymax, ymin, h_new)
    ground_img_pts, jac= cv2.projectPoints(ground_pts, rvec, tvec, K2, kc2)
    pix_x, pix_y = ground_img_pts[:,0,0], ground_img_pts[:,0,1]

    return pix_x, pix_y, w_new, h_new

if __name__ == '__main__':

    # set parameters
    PatternSize = (8,5)
    SquareSize = 30

    data_folder = 'data/'
    calib_params_filename = f'{data_folder}/calib/calib_params.json'
    calib_img_files = glob.glob(f'{data_folder}/calib/cam1_color_*.png')
    img_files = glob.glob(f'{data_folder}/imgs/cam1_color_*.png')
    
    #1 get camera params
    if os.path.exists(calib_params_filename):
        K, kc, w, h = utils.load_camera_params(calib_params_filename)
    else:
        K, kc, w, h = calibrate_camera(calib_img_files, PatternSize, SquareSize)

    K2, kc2, map1, map2 = get_undistorted_intrinsics(K, kc, w, h)

    #2 find ground location with respect to the camera
    # load images
    imgs = [cv2.imread(fname) for fname in img_files]
    # estimate ground pose
    img_ground = imgs[0].copy()
    found, corners, detected_target = detect_chessboard_pattern(img_ground, PatternSize)    
    pattern_points = create_target_points(PatternSize, SquareSize)
    R, T, pts_cam_space, rvec, tvec, R0, T0 = utils.calc_target_pose(corners, pattern_points, K, kc)
    # cv2.imshow('ground', detected_target)
    # cv2.waitKey()

    #3 create top view 
    # project plane to image 
    offsetx, offsety = 200, 500
    # offsetx, offsety = 50, 50
    xmin, xmax = -offsetx, (PatternSize[0]-1)*SquareSize+offsetx*1
    ymin, ymax = -offsety, (PatternSize[1]-1)*SquareSize+offsety*1
    pix_x, pix_y, w_new, h_new = project_plane_to_image(xmin, xmax, ymin, ymax, K2, kc2, rvec, tvec, sampling_rate=2)

    for img in imgs:
        
        # img_proj_pts = draw_project_pts(img, pix_x, pix_y, map1, map2)
        # cv2.imshow('projected plane points', img_proj_pts)

        # sample projected plane points from image
        img2 = utils.bilinear_interpolate(img, pix_x, pix_y , w_new, h_new)        
        
        cv2.imshow('input', img)
        cv2.imshow('top view', img2)
        cv2.waitKey()

 
        
