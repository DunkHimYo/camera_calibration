# v0 - calculates the homography from scratch at each step
import cv2
import numpy as np
import math
from object_module import *
import sys
from my_constants import *
from utils import get_extended_RT
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def augment(img, obj, projection, scale=4):
    # takes the captureed image, object to augment, and transformation matrix
    # adjust scale to make the object smaller or bigger, 4 works for the fox

    vertices = obj.vertices
    img = np.ascontiguousarray(img, dtype=np.uint8)
    w,h,_=img.shape
    w,h=-w,-h
    # projecting the faces to pixel coords and then drawing
    for face in obj.faces:
        # a face is a list [face_vertices, face_tex_coords, face_col]
        face_vertices = face[0]

        points = np.array([vertices[vertex - 1] for vertex in face_vertices])  # -1 because of the shifted numbering
        points = scale * points
        points = np.array([[p[2] + w / 2, p[0] + h / 2, p[1]] for p in points])  # shifted to centre

        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)  # transforming to pixel coords
        imgpts = np.int32(dst)

        cv2.fillConvexPoly(img, imgpts, face[-1])

    return img


if __name__ == '__main__':
    obj = three_d_object('data/3d_objects/low-poly-fox-by-pixelmannen.obj', 'data/3d_objects/texture.png')

    print("trying to access the webcam")
    vc = cv2.VideoCapture(0)
    hc, wc = 7, 10


    # considering all 4 rotations
    objpoints = []  # 3d co-ordinate
    imagepoints = []  # 2d co-ordinate
    objp = np.zeros((wc * hc, 3), np.float32)
    objp[:, :2] = np.mgrid[0:wc, 0:hc].T.reshape(-1, 2)

    def ani(i):

        rval, frame = vc.read()  # fetch frame from webcam
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (wc, hc), None)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret == True:
            objpoints.append(objp)
            imagepoints.append(corners)
            img = cv2.drawChessboardCorners(frame, (wc, hc), corners,ret)
            frame=np.flip(frame, axis=1)

            H = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -5]]).reshape(-1, 3)  # axis point for draw axis

            R_T = get_extended_RT(A, H)
            transformation = A.dot(R_T)

            augmented = np.flip(augment(frame, obj, transformation), axis=1)  # flipped for better control
            return plt.imshow(augmented, animated=True),
        else:
            # print('homograpy est failed')
            return plt.imshow(frame, animated=True),


    fig = plt.figure()
    anime = animation.FuncAnimation(fig, ani, interval=1, blit=True)
    plt.show()





