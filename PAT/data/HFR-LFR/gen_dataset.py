import numpy as np
import cv2
import os

def print_stat(narray, narray_name = 'Array'):
    print(narray_name + " stat: shape: {}, dtype: {}".format(narray.shape, narray.dtype))
    arr = narray.flatten()
    print("max: {}, min: {}, mean: {}, std: {}".format(arr.max(), arr.min(), arr.mean(), arr.std()))

res_w1, res_h1 = 1456, 1088
res_w2, res_h2 = 1280, 800
F1 = np.array([[4.07015327825553e-08,	4.17561843453929e-07,	-0.00808725180313401],
[-1.95715470638663e-07,	6.97119492853064e-08,	-0.0317664092598450],
[0.00536427411067715,	0.0236442325009365,	1.59796926454643]])
F2 = np.array([[4.84985763781936e-08,	-3.22308340577842e-07,	-0.00715097517336016],
[4.46350737251892e-07,	4.77193410572063e-08,	-0.0188197924505962],
[0.00500787791406712,	0.0141203268780633,	0.658266562896339]])
F3 = np.array([[1.66712445457951e-07,	-2.52883591320674e-07,	-0.0206496197252677],
[5.66035189455234e-07,	1.10559973826720e-07,	-0.0320562113764354],
[0.0147028462393742,	0.0241689869067774,	3.58052752388209]])

image_coor = np.meshgrid(np.arange(res_h1), np.arange(res_w1))
cam_coor = np.stack([image_coor[1].T,
                     image_coor[0].T,
                     np.ones_like(image_coor[1].T)], axis=2)
image_coor2 = np.meshgrid(np.arange(res_h2), np.arange(res_w2))
cam_coor2 = np.stack([image_coor2[1].T,
                      image_coor2[0].T,
                      np.ones_like(image_coor2[1].T)], axis=2)
folder = 'pillow_1'
img1 = cv2.imread(f'{folder}/{141}.png', 0)
img1 = ((img1/img1.max())**(1/1.6)*255.0).astype(np.uint8)
img2 = cv2.imread(f'{folder}/cam1_4.png', 0)
img3 = cv2.imread(f'{folder}/cam2_4.png', 0)
img3 = (img3/255.0*3./5.*255.0).astype(np.uint8)
img4 = cv2.imread(f'{folder}/cam3_4.png', 0)
img4 = (np.clip(img4/255.0*8./5., 0., 1.)*255.0).astype(np.uint8)

for i in range(2):
    for j in range(4):

        ID = i*4+j+1

        print(ID)
        xl, xu, yl, yu = i*544, i*544+544, j*364, j*364+364

        xxss = []
        yyss = []

        for point_x in range(xl, xu):
            for point_y in range(yl, yu):
                lam = F1.dot(np.array([point_y, point_x, 1]).reshape((-1, 3)).T)
                if (0.728*point_x - 50) < 0:
                    x_lb, x_ub = 0, 100
                elif (0.728*point_x + 50) > res_w2:
                    x_lb, x_ub = res_w2 - 100, res_w2
                else:
                    x_lb, x_ub = int(.728*point_x - 50), int(.728*point_x + 50)
                y_ub = min(int(.738*point_y + 228), res_w2)
                y_lb = y_ub - 100
                dist = np.abs(cam_coor2[x_lb:x_ub, y_lb:y_ub].reshape((-1, 3)).dot(lam))
                n = 100
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
                if (0.728*point_x - 71) < 0:
                    x_lb, x_ub = 0, 100
                elif (0.728*point_x + 29) > res_w2:
                    x_lb, x_ub = res_w2 - 100, res_w2
                else:
                    x_lb, x_ub = int(.728*point_x - 71), int(.728*point_x + 29)
                y_ub = min(int(.738*point_y + 205), res_w2)
                y_lb = y_ub - 100
                dist = np.abs(cam_coor2[x_lb:x_ub, y_lb:y_ub].reshape((-1, 3)).dot(lam))
                n = 100
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
                if (0.728*point_x - 60) < 0:
                    x_lb, x_ub = 0, 100
                elif (0.728*point_x + 40) > res_w2:
                    x_lb, x_ub = res_w2 - 100, res_w2
                else:
                    x_lb, x_ub = int(.728*point_x - 60), int(.728*point_x + 40)
                y_ub = min(int(.738*point_y + 236), res_w2)
                y_lb = y_ub - 100
                dist = np.abs(cam_coor2[x_lb:x_ub, y_lb:y_ub].reshape((-1, 3)).dot(lam))
                n = 100
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