import cv2 as cv
# # import os
# # import re
# # import matplotlib.pyplot as plt
import numpy as np
# # def mouse_handler(event, x, y, flags, data) :
    
# #     if event == cv.EVENT_LBUTTONDOWN :

# #         print(x,' ',y)

# # def erode_dilate(image):
# #     """
# #     Function to perform morphological operations to fill the 'holes' in the threshold image
# #     """
# #     kernel_erosion = np.ones((3, 3), np.uint8)
# #     kernel_dilation=np.ones((7,7),np.uint8)

# #     img_erosion = cv.erode(image, kernel_erosion, iterations=1)
# #     img_dilation = cv.dilate(img_erosion, kernel_dilation, iterations=2)

# #     return img_dilation

def find_centroid(single_contour):
	M=cv.moments(single_contour)
	if M['m00']!=0:
		cx=int(M['m10']/M['m00'])
		cy=int(M['m01']/M['m00'])
		return (cx,cy)
img=cv.imread('./countertop_with_markers/RGB_new/rgb18.jpg')
blank_outer=np.zeros(img.shape,dtype='uint8')
blank_inner=np.zeros(img.shape,dtype='uint8')
# img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
img_hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
img_yuv=cv.cvtColor(img,cv.COLOR_BGR2YUV)
img_lab=cv.cvtColor(img,cv.COLOR_BGR2LAB)
l,a,b1=cv.split(img_lab)
img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
y,u,v1=cv.split(img_yuv)
h,s,v=cv.split(img_hsv)
r,g,b=cv.split(img)
# cv.imshow("IMG",img)
# mask1=cv.inRange(b1,143,165)
# mask2=cv.inRange(b1,65,80)
# outer_marker_mask=cv.inRange(v1,90,110)
outer_marker_mask=cv.inRange(a,100,110)

inner_marker_mask=cv.inRange(a,155,185)

# mask=erode_dilate(mask)
# cv.imshow("B",b)
# cv.imshow("G",g)
# cv.imshow("R",r)
# cv.imshow("H",h)
# cv.imshow("S",s)
# cv.imshow("V",v)
# cv.imshow("Y",y)
# cv.imshow("U",u)
# cv.imshow("V1",v1)
# cv.imshow("L",l)
# cv.imshow("A",a)
# cv.imwrite("./countertop_with_markers/results/a_channel_space.jpg",a)
# cv.imshow("B1",b1)
# cv.imshow("GRAY",img_gray)
# cv.setMouseCallback("IMG", mouse_handler)

for i in range(len(outer_marker_mask)):
	for j in range(len(outer_marker_mask[0])):
		if outer_marker_mask[i][j]==255:
			blank_outer[i][j]=(0,255,0)

for i in range(len(inner_marker_mask)):
	for j in range(len(inner_marker_mask[0])):
		if inner_marker_mask[i][j]==255:
			blank_inner[i][j]=(0,0,255)

contours_outer, hierarchy1 = cv.findContours(outer_marker_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# contours_outer = sorted(contours_outer, key=cv.contourArea)
contours_inner, hierarchy2 = cv.findContours(inner_marker_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# contours_inner=sorted(contours_inner,key=cv.contourArea)
outer_centre_points=[]
inner_centre_points=[]
for contour in contours_inner:
	# for i in range(len(contour)):
	# 	img=cv.circle(img,(contour[i][0][0],contour[i][0][1]),1,(0,0,0),-1)
	centre_inner=find_centroid(contour)
	inner_centre_points.append(centre_inner)
	img=cv.circle(img,centre_inner,3,(0,0,0),-1)
	# img=cv.putText(img,str(centre_inner),(centre_inner[0]-50,centre_inner[1]-20),cv.FONT_HERSHEY_SIMPLEX, 
 #                   0.8, (7,237,229),2, cv.LINE_AA)

for contour in contours_outer:
	# for i in range(len(contour)):
	# 	img=cv.circle(img,(contour[i][0][0],contour[i][0][1]),2,(0,0,0),-1)
	centre_outer=find_centroid(contour)
	outer_centre_points.append(centre_outer)
	img=cv.circle(img,centre_outer,3,(0,0,0),-1)
	# img=cv.putText(img,str(centre_outer),(centre_outer[0]-50,centre_outer[1]-20),cv.FONT_HERSHEY_SIMPLEX, 
 #                   0.8, (7,237,229),2, cv.LINE_AA)
img=cv.line(img,outer_centre_points[0],outer_centre_points[1],(0,255,0),2)
img=cv.line(img,outer_centre_points[1],outer_centre_points[3],(0,255,0),2)
img=cv.line(img,outer_centre_points[3],outer_centre_points[2],(0,255,0),2)
img=cv.line(img,outer_centre_points[2],outer_centre_points[0],(0,255,0),2)

img=cv.line(img,inner_centre_points[0],inner_centre_points[1],(0,0,255),2)
img=cv.line(img,inner_centre_points[1],inner_centre_points[2],(0,0,255),2)
img=cv.line(img,inner_centre_points[2],inner_centre_points[3],(0,0,255),2)
img=cv.line(img,inner_centre_points[3],inner_centre_points[0],(0,0,255),2)

print(outer_centre_points)
print(inner_centre_points)
# print(contours2[1])
# cv.drawContours(img, contours2, 0, (0,0,0), 3)
cv.imshow("IMG",img)
# cv.imwrite('./countertop_with_markers/results/img_with_lines19.jpg',img)
# cv.imshow("MASK1",mask1)
# cv.imshow("MASK2",mask2)
cv.imshow("Outer Mask",blank_outer)
cv.imshow("Inner mask",blank_inner)
# cv.imwrite('./countertop_with_markers/results/outer_mask19.jpg',blank_outer)
# cv.imwrite('./countertop_with_markers/results/inner_mask19.jpg',blank_inner)


# plt.imshow(img)
# plt.show()
cv.waitKey(0)


# from skimage.segmentation import slic
# from skimage.segmentation import mark_boundaries,find_boundaries
# from skimage.util import img_as_float
# from skimage import io
# import matplotlib.pyplot as plt
# import argparse
# # construct the argument parser and parse the arguments
# blank=np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
# blank2=np.zeros(img.shape,dtype='uint8')
# blank3=np.zeros(img.shape,dtype='uint8')

# # load the image and convert it to a floating point data type
# image = img_as_float(img)
# # loop over the number of segments
# numSegments=2
# # apply SLIC and extract (approximately) the supplied number
# # of segments
# segments = slic(image, n_segments = numSegments,compactness=20,sigma=5)
# print(np.unique(segments))
# for i in range(len(segments)):
# 	for j in range(len(segments[0])):
# 		if segments[i][j]==1:
# 			blank2[i][j]=(0,0,255)
# 			blank[i][j]=255

# contours, hierarchy = cv.findContours(blank, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# max_contour=max(contours,key=cv.contourArea)
# hull = cv.convexHull(max_contour)
# epsilon = 0.03 * cv.arcLength(hull, True)
# 	# print(epsilon)
# approximations = cv.approxPolyDP(hull, epsilon, True)
# cv.drawContours(blank2, [approximations], 0, (0,255,0), 3)
# # cv.drawContours(blank2, [hull], 0, (0,255,0), 3)


# cv.imshow("BLANK",blank2)
# bond=find_boundaries(segments)
# print(np.unique(bond))
# for i in range(len(bond)):
# 	for j in range(len(bond)):
# 		if bond[i][j]==True:
# 			blank3[i][j]=(255,0,0)
# # cv.imshow("NEW",blank3)			
# # show the output of SLIC
# fig = plt.figure("Superpixels -- %d segments" % (numSegments))
# ax = fig.add_subplot(1, 1, 1)

# ax.imshow(mark_boundaries(image, segments))
# plt.axis("off")
# # show the plots
# plt.show()

# cv.waitKey(0)