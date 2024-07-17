import numpy as np
import cv2

def apply_filter_a(src:np.ndarray):
        src_copy = np.copy(src)
        f1 = np.array([[[ 0,  0,  0,  0,  0],
            [ 0,  1,  0,  0,  0],
            [ 0,  0, -1,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0]],

        [[ 0,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  0],
            [ 0,  0, -1,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0]],

        [[ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  1,  0],
            [ 0,  0, -1,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0]],

        [[ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0, -1,  1,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0]],

        [[ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0, -1,  0,  0],
            [ 0,  0,  0,  1,  0],
            [ 0,  0,  0,  0,  0]],

        [[ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0, -1,  0,  0],
            [ 0,  0,  1,  0,  0],
            [ 0,  0,  0,  0,  0]],

        [[ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0, -1,  0,  0],
            [ 0,  1,  0,  0,  0],
            [ 0,  0,  0,  0,  0]],

        [[ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  1, -1,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0]]])

        img = cv2.filter2D(src=src_copy, kernel=f1[0], ddepth=-1)
        for filter in f1[1:]:
            img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

        return img//8


def apply_filter_b(src:np.ndarray):
        src_copy = np.copy(src)
        f2 = np.array([[[ 0,  0,  0,  0,  0],
                        [ 0,  2,  1,  0,  0],
                        [ 0,  1, -3,  0,  0],
                        [ 0,  0,  0,  1,  0],
                        [ 0,  0,  0,  0,  0]],

                        [[ 0,  0, -1,  0,  0],
                        [ 0,  0,  3,  0,  0],
                        [ 0,  0, -3,  0,  0],
                        [ 0,  0,  1,  0,  0],
                        [ 0,  0,  0,  0,  0]],

                        [[ 0,  0,  0,  0,  0],
                        [ 0,  0,  1,  2,  0],
                        [ 0,  0, -3,  1,  0],
                        [ 0,  1,  0,  0,  0],
                        [ 0,  0,  0,  0,  0]],

                        [[ 0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0],
                        [ 0,  1, -3,  3, -1],
                        [ 0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0]],

                        [[ 0,  0,  0,  0,  0],
                        [ 0,  1,  0,  0,  0],
                        [ 0,  0, -3,  1,  0],
                        [ 0,  0,  1,  2,  0],
                        [ 0,  0,  0,  0,  0]],

                        [[ 0,  0,  0,  0,  0],
                        [ 0,  0,  1,  0,  0],
                        [ 0,  0, -3,  0,  0],
                        [ 0,  0,  3,  0,  0],
                        [ 0,  0, -1,  0,  0]],

                        [[ 0,  0,  0,  0,  0],
                        [ 0,  0,  0,  1,  0],
                        [ 0,  1, -3,  0,  0],
                        [ 0,  2,  1,  0,  0],
                        [ 0,  0,  0,  0,  0]],

                        [[ 0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0],
                        [-1,  3, -3,  1,  0],
                        [ 0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0]]])

        img = cv2.filter2D(src=src_copy, kernel=f2[0], ddepth=-1)
        for filter in f2[1:]:
            img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

        return img//8


def apply_filter_c(src:np.ndarray):
        src_copy=np.copy(src)
        f3 = np.array([[[ 0,  0,  0,  0,  0],
                        [ 0,  0,  1,  0,  0],
                        [ 0,  0, -2,  0,  0],
                        [ 0,  0,  1,  0,  0],
                        [ 0,  0,  0,  0,  0]],

                        [[ 0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0],
                        [ 0,  1, -2,  1,  0],
                        [ 0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0]],

                        [[ 0,  0,  0,  0,  0],
                        [ 0,  1,  0,  0,  0],
                        [ 0,  0, -2,  0,  0],
                        [ 0,  0,  0,  1,  0],
                        [ 0,  0,  0,  0,  0]],

                        [[ 0,  0,  0,  0,  0],
                        [ 0,  0,  0,  1,  0],
                        [ 0,  0, -2,  0,  0],
                        [ 0,  1,  0,  0,  0],
                        [ 0,  0,  0,  0,  0]]])

        img = cv2.filter2D(src=src_copy, kernel=f3[0], ddepth=-1)
        for filter in f3[1:]:
            img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

        return img//4


def apply_filter_d(src:np.ndarray):
        src_copy=np.copy(src)
        f4 = np.array([[[ 0,  0,  0,  0,  0],
                        [ 0, -1,  2, -1,  0],
                        [ 0,  2, -4,  2,  0],
                        [ 0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0]],

                        [[ 0,  0,  0,  0,  0],
                        [ 0, -1,  2,  0,  0],
                        [ 0,  2, -4,  0,  0],
                        [ 0, -1,  2,  0,  0],
                        [ 0,  0,  0,  0,  0]],

                        [[ 0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0],
                        [ 0,  2, -4,  2,  0],
                        [ 0, -1,  2, -1,  0],
                        [ 0,  0,  0,  0,  0]],

                        [[ 0,  0,  0,  0,  0],
                        [ 0,  0,  2, -1,  0],
                        [ 0,  0, -4,  2,  0],
                        [ 0,  0,  2, -1,  0],
                        [ 0,  0,  0,  0,  0]]])

        img = cv2.filter2D(src=src_copy, kernel=f4[0], ddepth=-1)
        for filter in f4[1:]:
            img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

        return img//4


def apply_filter_e(src:np.ndarray):
        src_copy=np.copy(src)
        f5 = np.array([[[  1,   2,  -2,   2,   1],
                        [  2,  -6,   8,  -6,   2],
                        [ -2,   8, -12,   8,  -2],
                        [  0,   0,   0,   0,   0],
                        [  0,   0,   0,   0,   0]],

                    [[  1,   2,  -2,   0,   0],
                        [  2,  -6,   8,   0,   0],
                        [ -2,   8, -12,   0,   0],
                        [  2,  -6,   8,   0,   0],
                        [  1,   2,  -2,   0,   0]],

                    [[  0,   0,   0,   0,   0],
                        [  0,   0,   0,   0,   0],
                        [ -2,   8, -12,   8,  -2],
                        [  2,  -6,   8,  -6,   2],
                        [  1,   2,  -2,   2,   1]],

                    [[  0,   0,  -2,   2,   1],
                        [  0,   0,   8,  -6,   2],
                        [  0,   0, -12,   8,  -2],
                        [  0,   0,   8,  -6,   2],
                        [  0,   0,  -2,   2,   1]]])

        img = cv2.filter2D(src=src_copy, kernel=f5[0], ddepth=-1)
        for filter in f5[1:]:
            img=cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

        return img//4


def apply_filter_f(src:np.ndarray):
        src_copy=np.copy(src)
        f5 = np.asarray([[ 0,  0,  0,  0,  0],
                        [ 0,  -1,  2, -1,  0],
                        [ 0,  2,  -4,  2,  0],
                        [ 0,  -1,  2, -1,  0],
                        [ 0,  0,  0,  0,  0]])

        img = cv2.filter2D(src=src_copy, kernel=f5, ddepth=-1)
        return img


def apply_filter_g(src:np.ndarray):
        src_copy=np.copy(src)
        f5 = np.asarray([[ -1,   2,  -2,   2,  -1],
                        [  2,  -6,   8,  -6,   2],
                        [ -2,   8, -12,   8,  -2],
                        [  2,  -6,   8,  -6,   2],
                        [ -1,   2,  -2,   2,  -1]])

        img = cv2.filter2D(src=src_copy, kernel=f5, ddepth=-1)
        return img


def apply_all_filters(src:np.ndarray):
        src_copy = np.copy(src)
        img = np.array(cv2.cvtColor((apply_filter_a(src_copy) + apply_filter_b(src_copy) + apply_filter_c(src_copy) + \
                apply_filter_d(src_copy) + apply_filter_e(src_copy) + apply_filter_f(src_copy) + apply_filter_g(src_copy)), cv2.COLOR_RGB2GRAY)//7)
        img_thresh = np.median(img)+2
        return cv2.threshold(img,img_thresh,255,cv2.THRESH_BINARY)[1]