import os
import glob
import json
import cv2
import numpy as np
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