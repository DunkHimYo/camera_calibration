import matplotlib.animation as animation

import numpy as np
import cv2
import matplotlib.pyplot as plt


def calibration(frame):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    wc = 10  ## 체스 보드 가로 패턴 개수 - 1
    hc = 7  ## 체스 보드 세로 패턴 개수 - 1
    objp = np.zeros((wc * hc, 3), np.float32)
    objp[:, :2] = np.mgrid[0:wc, 0:hc].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    _img = cv2.resize(frame, dsize = (640, 480), interpolation = cv2.INTER_AREA)
    gray=cv2.cvtColor(_img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (wc, hc), None)  ## 체스 보드 찾기


    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria) ## Canny86 알고리즘으로

        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(_img, (wc, hc), corners2, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)  ## 왜곡 펴기
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0)

        dst = cv2.undistort(img, mtx, dist) ## getOptimalNewCameraMatrix 함수를 쓰지 않은 이미지
        dst2 = cv2.undistort(img, mtx, dist, None, newcameramtx) ## 함수를 쓴 이미지
        return dst2
    else:
        return gray


def ani(i):
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = capture.read()
    frame=calibration(frame)

    T = plt.imshow(frame, animated=True)
    return T,

fig = plt.figure()
anime = animation.FuncAnimation(fig, ani, interval=1, blit=True)
plt.show()