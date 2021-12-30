import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from math import sqrt
img=cv.imread('frame.jpg')

def draw_grid(img,points,color):
	
	for i in range(len(points)):
		
		img = cv.circle(img, (int(points[i][0]),int(points[i][1])), radius=5, color=color, thickness=1)

	for i in range(len(points)):
		if i==0 or i==1 or i==3 or i==4:
			img=cv.line(img,(int(points[i][0]),int(points[i][1])),(int(points[i+1][0]),int(points[i+1][1])),color,thickness=1)

		if i==0 or i==1 or i==2:
			img=cv.line(img,(int(points[i][0]),int(points[i][1])),(int(points[i+3][0]),int(points[i+3][1])),color,thickness=1)
	return img

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

def mouse_handler(event, x, y, flags, data) :
    
    if event == cv.EVENT_LBUTTONDOWN :
        cv.circle(data['im'], (x,y),3, (0,0,255), 5, 16);
        cv.imshow("Image", data['im']);
        if len(data['points']) < 4 :
            data['points'].append([x,y])

def get_two_points(im):
    
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    
    #Set the callback function for any mouse event
    cv.imshow("Image",im)
    cv.setMouseCallback("Image", mouse_handler, data)
    cv.waitKey(0)
    
    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)
    
    return points


dest_pts=np.array([
      [ 557, 678 ],
      [ 557, 590 ],
      [ 557, 533 ],
      [ 785, 657 ],
	  [ 740, 578 ],
	  [ 707, 521 ]
    ])

src_pts=np.array([
      [ 415, 0],
      [ 515, 0],
      [ 615, 0],
      [ 415 , -100],
	  [ 515, -100],
	  [ 615, -100]
    ])


homography_matrix, status = cv.findHomography(src_pts, dest_pts)

img=draw_grid(img,dest_pts,(0,0,255))

regenerated_pts=reproject(homography_matrix,src_pts)
print(regenerated_pts)

print(dest_pts)
img=draw_grid(img,regenerated_pts,(0,255,0))



#Calculating distance of road
homography_matrix_inv=np.linalg.inv(homography_matrix)

# distance_points=np.array([[400,490],
# 	        			  [1150,420]])
distance_points = get_two_points(img)
distance_points=np.float32(distance_points)
new_points=reproject(homography_matrix_inv,distance_points)

distance=round((sqrt(((new_points[0][0]-new_points[1][0])**2)+((new_points[0][1]-new_points[1][1])**2))*3.28)/100,3)
print(distance)

img = cv.circle(img, (distance_points[0][0],distance_points[0][1]), radius=5, color=(0,0,255), thickness=-1)
img = cv.circle(img, (distance_points[1][0],distance_points[1][1]), radius=5, color=(0,0,255), thickness=-1)

img=cv.line(img,(distance_points[0][0],distance_points[0][1]),(distance_points[1][0],distance_points[1][1]),(0,0,255),thickness=1)

img=cv.putText(img,"Distance = "+str(distance)+" ft",(550,375),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

cv.imshow("Frame",img)
cv.imwrite('./result.jpg',img)
cv.waitKey(0)


