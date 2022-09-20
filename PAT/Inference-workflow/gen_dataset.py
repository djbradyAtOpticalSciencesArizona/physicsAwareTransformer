import numpy as np
import cv2
import os

def print_stat(narray, narray_name = 'Array'):
    print(narray_name + " stat: shape: {}, dtype: {}".format(narray.shape, narray.dtype))
    arr = narray.flatten()
    print("max: {}, min: {}, mean: {}, std: {}".format(arr.max(), arr.min(), arr.mean(), arr.std()))

res_w1, res_h1 = 1280, 800
res_w2, res_h2 = 1280, 800
F1 = np.array([[-4.82421044278448e-08,	-1.71285515462556e-06,	0.0135650778093583],
[1.72556528840005e-06,	1.67656609303512e-09,	-0.0143328867069307],
[-0.00912167234149013,	0.00993775597697036,	-0.681697861665831]])
F2 = np.array([[-3.97581572992089e-08,	-3.95153001829531e-07,	0.0134920153619862],
[3.79192112912345e-07,	-5.29231028279867e-09,	-0.000385164443855964],
[-0.00889027295478548,	0.000476491990734183,	-2.94954746159413]])
F3 = np.array([[-1.72325742774190e-08,	-1.70967758549794e-06,	0.000401656453872548],
[1.77514485090898e-06,	1.00163736098407e-08,	-0.0143002582536094],
[-0.000502122564603341,	0.00983482138825411,	2.53255407512145]])

image_coor = np.meshgrid(np.arange(res_h1), np.arange(res_w1))
cam_coor = np.stack([image_coor[1].T,
                     image_coor[0].T,
                     np.ones_like(image_coor[1].T)], axis=2)
image_coor2 = np.meshgrid(np.arange(res_h2), np.arange(res_w2))
cam_coor2 = np.stack([image_coor2[1].T,
                      image_coor2[0].T,
                      np.ones_like(image_coor2[1].T)], axis=2)

img1 = cv2.imread('cam4_expo150.png', 0)
img2 = cv2.imread('cam1_expo500.png', 0)
img3 = cv2.imread('cam2_expo300.png', 0)
img4 = cv2.imread('cam3_expo800.png', 0)


for i in range(2):
    for j in range(2):
        
        ID = i*2+j+5
        
        print(ID)
        xl, xu, yl, yu = i*400, i*400+400, j*640, j*640+640
        
        xxss = []
        yyss = []

        for point_x in range(xl, xu):
            for point_y in range(yl, yu):
                lam = F1.dot(np.array([point_y, point_x, 1]).reshape((-1, 3)).T)
                x_lb = int(point_x/1.5 + 140) #140
                x_ub = int(point_x/1.5 + 220) #220
                y_lb = int(point_y/1.5 + 190) #190
                y_ub = int(point_y/1.5 + 270) #270
                dist = np.abs(cam_coor2[x_lb:x_ub, y_lb:y_ub].reshape((-1, 3)).dot(lam))
                n = 80
                pps = np.argpartition(dist, n, axis=0)[:n].T
                xxs, yys = pps//n + x_lb, pps%n+y_lb
                xxss.append(xxs[0])
                yyss.append(yys[0])

        xxss = np.stack(xxss, axis=0).astype(np.int16)
        yyss = np.stack(yyss, axis=0).astype(np.int16)

        print(xxss.shape)

        np.save('xxs_{:04d}_1.npy'.format(ID), xxss)
        np.save('yys_{:04d}_1.npy'.format(ID), yyss)
        
        if not os.path.exists('hr/{:04d}'.format(ID)):
            os.makedirs('hr/{:04d}'.format(ID))
        cv2.imwrite('hr/{:04d}/hr0.png'.format(ID), img1[xl:xu, yl:yu])
        cv2.imwrite('hr/{:04d}/hr1.png'.format(ID), img2)

        xxss = []
        yyss = []

        for point_x in range(xl, xu):
            for point_y in range(yl, yu):
                lam = F2.dot(np.array([point_y, point_x, 1]).reshape((-1, 3)).T)
                x_lb = int(point_x/1.5 + 140) #140
                x_ub = int(point_x/1.5 + 220) #220
                y_lb = int(point_y/1.5 + 190) #190
                y_ub = int(point_y/1.5 + 270) #270
                dist = np.abs(cam_coor2[x_lb:x_ub, y_lb:y_ub].reshape((-1, 3)).dot(lam))
                n = 80
                pps = np.argpartition(dist, n, axis=0)[:n].T
                xxs, yys = pps//n + x_lb, pps%n+y_lb
                xxss.append(xxs[0])
                yyss.append(yys[0])

        xxss = np.stack(xxss, axis=0).astype(np.int16)
        yyss = np.stack(yyss, axis=0).astype(np.int16)

        print(xxss.shape)

        np.save('xxs_{:04d}_2.npy'.format(ID), xxss)
        np.save('yys_{:04d}_2.npy'.format(ID), yyss)
        
        cv2.imwrite('hr/{:04d}/hr2.png'.format(ID), img3)
        
        xxss = []
        yyss = []

        for point_x in range(xl, xu):
            for point_y in range(yl, yu):
                lam = F3.dot(np.array([point_y, point_x, 1]).reshape((-1, 3)).T)
                x_lb = int(point_x/1.5 + 140) #140
                x_ub = int(point_x/1.5 + 220) #220
                y_lb = int(point_y/1.5 + 190) #190
                y_ub = int(point_y/1.5 + 270) #270
                dist = np.abs(cam_coor2[x_lb:x_ub, y_lb:y_ub].reshape((-1, 3)).dot(lam))
                n = 80
                pps = np.argpartition(dist, n, axis=0)[:n].T
                xxs, yys = pps//n + x_lb, pps%n+y_lb
                xxss.append(xxs[0])
                yyss.append(yys[0])

        xxss = np.stack(xxss, axis=0).astype(np.int16)
        yyss = np.stack(yyss, axis=0).astype(np.int16)

        print(xxss.shape)

        np.save('xxs_{:04d}_3.npy'.format(ID), xxss)
        np.save('yys_{:04d}_3.npy'.format(ID), yyss)

        cv2.imwrite('hr/{:04d}/hr3.png'.format(ID), img4)
