import json
import sys
import random
from functools import reduce

import numpy as np
from threading import Thread
import matplotlib.pyplot as plt
import time
import cv2
import cv2 as cv
from math import sqrt
import math
from sklearn.metrics.pairwise import euclidean_distances


class ThreadedCamera(object):
    def __init__(self, src=''):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)

    def show_frame(self, cam_name=''):
        cv2.imshow(cam_name, self.frame)
        cv2.waitKey(self.FPS_MS)

    def get_frame(self):
        return self.frame, self.FPS_MS


def draw_grid(img, points, color):

    for i in range(len(points)):
        img = cv.circle(img, (points[i][0], points[i][1]), radius=5, color=color, thickness=1)

    # for i in range(len(points)):
    #     if i == 0 or i == 1 or i == 3 or i == 4:
    #         img = cv.line(img, (points[i][0], points[i][1]), (points[i+1][0], points[i+1][1]), color, thickness=1)
    #     if i == 0 or i == 1 or i == 2:
    #         img = cv.line(img, (points[i][0], points[i][1]), (points[i+3][0], points[i+3][1]), color, thickness=1)

    return img


def reproject(homo_matrix, src_pts):
    regenerated_pts = []
    for i in src_pts:
        point = []
        test_point = np.array([[i[0]], [i[1]], [1]])
        reprojected = np.matmul(homo_matrix, test_point)
        reprojected = reprojected / reprojected[2]
        x2 = int(reprojected[0])
        y2 = int(reprojected[1])
        point.append(x2)
        point.append(y2)
        regenerated_pts.append(point)

    regenerated_pts = np.array(regenerated_pts)

    return regenerated_pts


def click_event(event, x, y, flags, param):
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)


img_points = np.array([
    [292,291],
    [286,217],
    [283,170],
    [283,136],
    [282,110],
    [282,92],


    [104,307],
    [140,229],
    [164,177],
    [176,143],
    [188,117],
    [195,96],

    [467,283],
    [426,214],
    [398,167],
    [377,134],
    [361,110],
    [351,90]

    
])

world_points = np.array([
    [0,150],
    [0,200],
    [0,250],
    [0,300],
    [0,350],
    [0,400],

    [-50,150],
    [-50,200],
    [-50,250],
    [-50,300],
    [-50,350],
    [-50,400],

    [50,150],
    [50,200],
    [50,250],
    [50,300],
    [50,350],
    [50,400]

  

])

# Directory to write the homography matrix


## Train test split
temp = list(zip(img_points, world_points))

random.shuffle(temp)
img_src_pts, world_dst_pts = zip(*temp)

img_src_pts = np.asarray(img_src_pts)

world_dst_pts = np.asarray(world_dst_pts)

indices = np.random.permutation(img_src_pts.shape[0])

print("Total size of points: {}, training size: {}, testing size: {}".format(len(img_src_pts), int(0.80 * len(img_src_pts)),
                                                                             int(0.20 * len(img_src_pts))+1))
training_idx, test_idx = indices[:int(0.80 * len(img_src_pts))], indices[int(0.80 * len(img_src_pts)):]

num_iter = 5
x_error = []
y_error = []
homography_matrix = []
for i in range(num_iter):
#     # Train and val split
    random.shuffle(training_idx)
    mini_train_idx, mini_val_idx = training_idx[:int(0.80*len(training_idx))], training_idx[int(0.80*len(training_idx)):]

#     # Computer the homography matrix
    homography_matrix, status = cv.findHomography(img_src_pts[mini_train_idx], world_dst_pts[mini_train_idx])
#     # Re-project the validation points
    reproject_points = reproject(homography_matrix, img_src_pts[mini_val_idx])
    x_offset = []
    y_offset = []
    for dt_pt, rp_pt in zip(world_dst_pts[mini_val_idx], reproject_points):
        # x_err = abs(dt_pt[0] - rp_pt[0])/dt_pt[0] * 100
        # y_err = abs(dt_pt[1] - rp_pt[1])/dt_pt[1] * 100
        x_err=(dt_pt[0]-rp_pt[0])**2
        y_err=(dt_pt[1]-rp_pt[1])**2
        x_offset.append(x_err)
        y_offset.append(y_err)

    # print("x_offset: ", x_offset)
    # print("y_offset: ", y_offset)
    x_offset = sqrt(reduce(lambda a, b: a + b, x_offset)) / len(x_offset)
    y_offset = sqrt(reduce(lambda a, b: a + b, y_offset)) / len(y_offset)
    x_error.append(x_offset)
    y_error.append(y_offset)

print("x_error: ", x_error)
print("y_error: ", y_error)
x_error = reduce(lambda a, b: a + b, x_error) / len(x_error)
y_error = reduce(lambda a, b: a + b, y_error) / len(y_error)
print("x_error: ", x_error)
print("y_error: ", y_error)

reproject_points = reproject(homography_matrix, img_src_pts[test_idx])
x_err = []
y_err = []
for dt_Tpt, rp_Tpt in zip(world_dst_pts[test_idx], reproject_points):
    # x_err.append(abs(dt_Tpt[0] - rp_Tpt[0]) / dt_Tpt[0] * 100)
    # y_err.append(abs(dt_Tpt[1] - rp_Tpt[1]) / dt_Tpt[1] * 100)
    x_err.append((dt_Tpt[0]-rp_Tpt[0])**2)
    y_err.append((dt_Tpt[1]-rp_Tpt[1])**2)

print("Test x error: ", x_err)
print("Test y error: ", y_err)
print('TEST points        Reprojected points' )
for i in range(len(test_idx)):
    print(world_dst_pts[test_idx[i]],'          ',reproject_points[i])
# print("Test points: ", world_dst_pts[test_idx])
# print("Reprojected points: ", reproject_points)
x_err = sqrt(reduce(lambda a, b: a + b, x_err)) / len(x_err)
y_err = sqrt(reduce(lambda a, b: a + b, y_err) )/ len(y_err)
print("Test x error: ", x_err)
print("Test y error: ", y_err)
np.save('homography_matrix',homography_matrix)

# data = {"camera_matrix": homography_matrix.tolist()}
# json_filename = ''.join([directory, "homography_matrix.json"])
# print("Writing homography_matrix")
# with open(json_filename, "w") as f:
#     json.dump(data, f)
# print("Finished writing homography_matrix")
