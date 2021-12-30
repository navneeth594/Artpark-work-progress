import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import os


# mask1=cv.inRange(img_hsv, (160,100,20), (179,255,255))
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
def get_centroid_util(threshold_img):
    """
    Function to obtain the largest contour and calculate the centroid of that contour
    Returns the tuple consisting the x and y coordinates of the contour
    """
    contours, hierarchy = cv.findContours(threshold_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cx = 0
    cy = 0
    radius = 0
    if len(contours) != 0:
        # find the biggest contour (c) by the area
        c = max(contours, key=cv.contourArea)

        # compute the center of the contour
        m = cv.moments(c)
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        area = cv.contourArea(c)
        radius = math.sqrt(area/math.pi)

    return cx, cy, radius


def reproject(homo_matrix,src_pts):
    regenerated_pts=[]
    for i in src_pts:
        point=[]

        test_point=np.array([[i[0]],[i[1]],[1]])
        reprojected=np.matmul(homo_matrix,test_point)
        reprojected=reprojected/reprojected[2]
        x2=reprojected[0][0]
        y2=reprojected[1][0]

        point.append(x2)
        point.append(y2)
        regenerated_pts.append(np.array(point))


    regenerated_pts=np.array(regenerated_pts)
    regenerated_pts=np.float32(regenerated_pts)
    return regenerated_pts
def erode_dilate(image):
    """
    Function to perform morphological operations to fill the 'holes' in the threshold image
    """
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv.erode(image, kernel, iterations=1)
    img_dilation = cv.dilate(img_erosion, kernel, iterations=2)

    return img_dilation


# dustbin_distances=[159,142,172,129,141,165,110,176,263,164]
plain_img=cv.imread('./plain_image_Color.png')
images=os.listdir('./dustbin_images/')

images.sort(key=natural_keys)


homography_matrix=np.load('homography_matrix.npy')

for k in range(len(images)):
    img=cv.imread("./dustbin_images/"+images[k])
    img_hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    h,s,v=cv.split(img_hsv)

    mask=cv.inRange(s, 0,140)
    mask1=cv.bitwise_not(mask)


    mask1=erode_dilate(mask1)

    # cv.imshow("MASk",mask1)
    contour_points=[]
    contours, hierarchy = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     # drawing = np.zeros((mask1.shape[0], mask1.shape[1], 3), np.uint8)

    max_contour=max(contours,key=cv.contourArea) 
    for i in range(len(max_contour)):
        contour_points.append(max_contour[i][0])

    contour_points=np.array(contour_points)
    # number_of_rows=contour_points.shape[0]
    # choices=np.random.choice(number_of_rows,5)
    # print(contour_points[choices,:])
    

    x,y,w,h = cv.boundingRect(max_contour)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv.circle(img,(x+(w//2),y+h),8,(255,0,0),-1)

    bottom_point=np.array([[x+(w//2),y+h]])

    actual_distance=int(images[k][:3])

    re_points=reproject(homography_matrix,bottom_point)

    distance=round(math.sqrt((re_points[0][0]**2)+(re_points[0][1]**2)),1)

    cv.drawContours(image=img, contours=max_contour, contourIdx=-1, color=(0,0,255), thickness=2, lineType=cv.LINE_AA)

    ratio=round(actual_distance/distance,3)

    cv.putText(img,"Calculated distance : "+str(distance)+" cm",(0,50),cv.FONT_HERSHEY_SIMPLEX,1,
                        (0, 0,255), 2)
    cv.putText(img,"Actual distance : "+str(actual_distance)+" cm",(0,100),cv.FONT_HERSHEY_SIMPLEX,1,
                        (0, 0,255), 2)
    cv.circle(plain_img,(x+(w//2),y+h),5,(255,0,0),-1)
    cv.putText(plain_img,"A:"+str(actual_distance)+" C:"+str(distance),((x+(w//2))+8,y+h),cv.FONT_HERSHEY_SIMPLEX,0.6,
                        (0, 0,255), 2)
    # cv.putText(img,"Actual :  Calculated = "+str(ratio),(0,150),cv.FONT_HERSHEY_SIMPLEX,1,
    #                     (0, 0,255), 2)
    # cv.imshow("Original image",img)
    cv.imwrite("./dustbin_images_results/"+images[k],img)
cv.imshow("PLAIN",plain_img)
cv.imwrite("./combined.jpg",plain_img)
cv.waitKey(0)

