import cv2
import glob
import numpy as np
import os

h, w = 32, 96
E = np.array([[ 1.44517640e-03, -5.55392339e-17,  7.06741913e-01],
       [ 4.49689111e-02,  1.79205986e-20, -2.27127300e-02],
       [-7.05673939e-01,  2.76993333e-17, -8.52795705e-17]])
scale = 4
res_w2, res_h2 = 2048//scale, 1536//scale
f2 = 64. # (mm)
sensor_width = 36.
pix_size = sensor_width/res_w2

image_coor2 = np.meshgrid(np.arange(res_h2), np.arange(res_w2))
image_coor2 = [image_coor2[0].transpose(), image_coor2[1].transpose()]
image_coor2_in_mm = [image_coor2[0] * pix_size - res_h2//2 * pix_size + 0.5 * pix_size, 
            image_coor2[1] * pix_size - res_w2//2 * pix_size + 0.5 * pix_size]
cam_coor2 = np.stack([image_coor2_in_mm[0], 
            image_coor2_in_mm[1],
            -f2*np.ones_like(image_coor2_in_mm[0])], axis=2)

cam_coor = cam_coor2.copy()

def calc_lam(points):
    x_l, x_r, y_l, y_r = points
    p = cam_coor[x_l:x_r, y_l:y_r].reshape(-1, 3).T
    lam = E.dot(p)
    return lam

def calc_pos_mat(points, lam):
    x_l, x_r, y_l, y_r = points
    dist = np.abs(cam_coor2[x_l:x_r, y_l:y_r].reshape(-1, 3).dot(lam))/np.expand_dims((lam[0, :]**2+lam[1, :]**2)**0.5, axis=0)
    k = max(x_r - x_l, y_r - y_l) #dist < pix_size * 1.414 : but k for each point will be different
    pps = np.argpartition(dist, k, axis=0)[:k].T
    xxs,yys = pps//(y_r - y_l), pps%(y_r - y_l)
    return xxs, yys

lr0_list = sorted(glob.glob('whitex1/*.png'))
lr1_list = sorted(glob.glob('whitex2/*.png'))

for i in range(len(lr0_list)):
    img_0 = cv2.imread(lr0_list[i])
    img_1 = cv2.imread(lr1_list[i])
    
    img_lr0 = cv2.resize(img_0, (0, 0), fx=1/scale, fy=1/scale,interpolation=cv2.INTER_CUBIC)
    img_lr1 = cv2.resize(img_1, (0, 0), fx=1/scale, fy=1/scale,interpolation=cv2.INTER_CUBIC)

    idx_patch=1
    for j in range(7):
        for k in range(7):
            x_l, y_l = (400+80*j)//scale, (400+160*k)//scale
            x_r, y_r = x_l+32, y_l+96
            x_hr, y_hr = (x_l)*scale, (y_l)*scale
            hr_patch_0 = img_0[x_hr:x_hr+32*scale, y_hr:y_hr+96*scale]
            hr_patch_1 = img_1[x_hr:x_hr+32*scale, y_hr:y_hr+96*scale]            
            lr_patch_0 = img_lr0[x_l:x_r, y_l:y_r]
            lr_patch_1 = img_lr1[x_l:x_r, y_l:y_r]

            path = '../blender_patches_corrected/patches_x{}/{:04d}_{:03d}'.format(scale,i+1,j*7+k+1)
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(path+'/hr0.png', hr_patch_0)
            cv2.imwrite(path+'/hr1.png', hr_patch_1)
            cv2.imwrite(path+'/lr0.png', lr_patch_0)
            cv2.imwrite(path+'/lr1.png', lr_patch_1)

            lam = calc_lam((x_l, x_r, y_l, y_r))
            xxs, yys = calc_pos_mat((x_l, x_r, y_l, y_r), lam)

            np.save('../blender_patches_corrected/patches_x{}/xxs_{:03d}.npy'.format(scale,j*7+k+1),xxs.astype(np.int16))
            np.save('../blender_patches_corrected/patches_x{}/yys_{:03d}.npy'.format(scale,j*7+k+1),yys.astype(np.int16))
