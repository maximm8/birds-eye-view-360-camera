import os
import glob
import json
import cv2
import numpy as np
import codecs
import matplotlib.pyplot as plt

def load_camera_params(filename):
    p = []
    
    with open(filename, 'r') as openfile:
        p = json.load(openfile)

    K = np.array(p['intrinsics']).reshape(3,3)
    kc = np.array(p['distortions']).reshape(-1, 1)
    w = p['img_width'] 
    h = p['img_height']
    return K, kc, w, h

def save_camera_params(filename, K, kc, w, h):    
    with codecs.open(filename, 'w', encoding='utf-8') as f:
        cam_params = {}
        cam_params['intrinsics']  = K.tolist()
        cam_params['distortions'] = kc.tolist()[0]
        cam_params['img_width'] = w
        cam_params['img_height'] = h
        json.dump(cam_params, f, indent = 4, sort_keys=True)

def calc_target_pose(img_pts, obj_pts, K, kc): 
    retval, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, kc)#, flags = cv2.SOLVEPNP_UPNP)    
    rmat, jacobian = cv2.Rodrigues(rvec)
 
    R = np.linalg.inv(rmat)
    T = -np.matmul(R, tvec)
     
    pointsInCameraSpace = obj_points_to_cam_space(obj_pts, R, T)

    return R, T, pointsInCameraSpace, rvec, tvec, rmat, tvec

def obj_points_to_cam_space(obj_pts, R, T):
    pointsInCameraSpace = np.dot(obj_pts, R) + T.T

    return pointsInCameraSpace

def create_grid(xmin, xmax, num_x, ymin, ymax, num_y):
    x = np.linspace(xmin, xmax, num_x)
    y = np.linspace(ymin, ymax, num_y)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()     
    z = np.zeros_like(x)
    coords = np.stack([x, y, z], axis=1)

    return coords

def bilinear_interpolate(img, x, y, w, h):
    x = x.reshape((-1,1))
    y = y.reshape((-1,1))

    ch = img.shape[2]

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img.shape[1]-1);
    x1 = np.clip(x1, 0, img.shape[1]-1);
    y0 = np.clip(y0, 0, img.shape[0]-1);
    y1 = np.clip(y1, 0, img.shape[0]-1);

    Ia = np.squeeze(img[ y0, x0,: ])
    Ib = np.squeeze(img[ y1, x0,: ])
    Ic = np.squeeze(img[ y0, x1, : ])
    Id = np.squeeze(img[ y1, x1, : ])

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    img_out = wa*Ia + wb*Ib + wc*Ic + wd*Id
    
    return img_out.reshape((h, w, ch)).astype(np.uint8)

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
    ground_pts = create_grid(xmax, xmin, w_new, ymax, ymin, h_new)
    ground_img_pts, jac= cv2.projectPoints(ground_pts, rvec, tvec, K2, kc2)
    pix_x, pix_y = ground_img_pts[:,0,0], ground_img_pts[:,0,1]

    return pix_x, pix_y, w_new, h_new