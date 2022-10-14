import cv2
import numpy as np
from object_module import *
from my_constants import *
from utils import get_extended_RT
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class WebCam():
    """
    웹캠 켜는 클래스
    """
    def __init__(self):
        self.video_capture=cv2.VideoCapture(0)
        #카메라 설정 On
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #width = 640
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        #height = 480

    def read(self) -> "image":
        ret, img=self.video_capture.read()
        #img 데이터 읽기
        if ret:
            return img
        else:
            return None

class Applications(WebCam):
    """
    프로젝트 시연 클래스
    """
    def __init__(self,wc=10,hc=7):
        super().__init__()

        self.wc=wc
        #width chessboard
        self.hc=hc
        #height chessboard
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #cornerSubPix용도
        self.objp = np.zeros((wc * hc, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:wc, 0:hc].T.reshape(-1, 2)
        self.objpoints = []  # 3d co-ordinate
        self.imgpoints = []  # 2d co-ordinate
        self.obj = three_d_object('data/3d_objects/low-poly-fox-by-pixelmannen.obj', 'data/3d_objects/texture.png') #여우 모델
        
    def _find_corners(self):

        img = self.read()
        gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, self.corners = cv2.findChessboardCorners(gray_img, (self.wc, self.hc), None)
        if ret:
            self.objpoints.append(self.objp)
            self.imgpoints.append(self.corners)
            img = cv2.drawChessboardCorners(img, (self.wc, self.hc), self.corners, ret)

            return True, img, gray_img.shape
        else:
            return False, img, None
    def _3d_axis(self):
        """
        3차원 축 좌표
        """
        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)  # axis point for draw axis
        return axis
        # this for draw a x,y,z axis
    def _cube_axis(self):
        """
        cube 축 좌표
        """
        axis=np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])    # this for draw a cube
        return axis

    def draw_cube(self,img, imgpts):
        """
        cube 그리는 메소드
        """
        imgpts = np.int32(imgpts).reshape(-1, 2)
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 3)
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
        return img

    def draw_axis(self, img, corners, imgpts):
        """
        축을 그리는 메소드
        """
        corner = tuple(corners[0].ravel())
        print([int(i) for i in imgpts[0].ravel()])
        img = cv2.line(img, [int(i) for i in corner], [int(i) for i in imgpts[0].ravel()], (255, 0, 0),5)
        img = cv2.line(img, [int(i) for i in corner], [int(i) for i in imgpts[1].ravel()],(0, 255, 0),5)
        img = cv2.line(img, [int(i) for i in corner], [int(i) for i in imgpts[2].ravel()],(0, 0, 255),5)
        return img

    def calibration(self, img, gray_img_shape):
        """
        
        """
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray_img_shape[::-1], None,
                                                           None)  ## 왜곡 펴기

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 0)
        ## mtx = getOptimalNewCameraMatrix parameter alpha
        ## dist = Free scaling parameter
        ## 4번째 인자 = between 0 (when all the pixels in the undistorted image are valid) and 1 (when all the source image pixels are retained in the undistorted image)
        ## 1에 가까울수록 왜곡을 펼 때 잘라낸 부분들을 더 보여준다
        ## 전체를 보고 싶다면 1, 펴진 부분만 보고 싶다면 0에 가깝게 인자 값을 주면 된다
        # dst = cv2.undistort(img, mtx, dist)  ## getOptimalNewCameraMatrix 함수를 쓰지 않은 이미지
        img = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)  ## 함수를 쓴 이미지
        return img

    def augment(self, img, obj, projection, scale=4):
        # takes the captureed image, object to augment, and transformation matrix
        # adjust scale to make the object smaller or bigger, 4 works for the fox

        vertices = obj.vertices
        img = np.ascontiguousarray(img, dtype=np.uint8)
        w, h, _ = img.shape
        w, h = -w, -h
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

    def run(self, draws=['fox','cube']):

        ret, img, gray_img=self._find_corners()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if ret:
            img=self.calibration(img,gray_img)


            if 'cube' in draws:
                _, rvec, tvec, _ = cv2.solvePnPRansac(self.objp, self.corners, self.mtx, self.dist)
                imgpts, _ = cv2.projectPoints(self._cube_axis(), rvec, tvec, self.mtx, self.dist)

                img=self.draw_cube(img=img, imgpts=imgpts)
                print('chk1')

            if 'axis' in draws:
                _, rvec, tvec, _ = cv2.solvePnPRansac(self.objp, self.corners, self.mtx, self.dist)
                imgpts, _ = cv2.projectPoints(self._3d_axis(), rvec, tvec, self.mtx, self.dist)
                img = self.draw_axis(img=img, corners=self.corners, imgpts=imgpts)


            if 'fox' in draws:
                frame = np.flip(img, axis=1)

                H = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -5]]).reshape(-1, 3)  # axis point for draw axis

                R_T = get_extended_RT(A, H)
                transformation = A.dot(R_T)

                img = np.flip(self.augment(frame, self.obj, transformation), axis=1)  # flipped for better control



            return img

        else:
            return img


if __name__=='__main__':
    cali=Calibration()

    def ani(i):

        frame=cali.run()
        return plt.imshow(frame, animated=True),

    fig = plt.figure()
    anime = animation.FuncAnimation(fig, ani, interval=1, blit=True)
    plt.show()
