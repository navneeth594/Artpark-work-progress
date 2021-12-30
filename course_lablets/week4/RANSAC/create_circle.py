import cv2 as cv
import numpy as np
import random
from math import sqrt
from tqdm import tqdm

def findCircle(x1, y1, x2, y2, x3, y3) :
	x12 = x1 - x2;
	x13 = x1 - x3;
	y12 = y1 - y2;
	y13 = y1 - y3;
	y31 = y3 - y1;
	y21 = y2 - y1;
	x31 = x3 - x1;
	x21 = x2 - x1;
	sx13 = pow(x1, 2) - pow(x3, 2);
	sy13 = pow(y1, 2) - pow(y3, 2);
	sx21 = pow(x2, 2) - pow(x1, 2);
	sy21 = pow(y2, 2) - pow(y1, 2);
	f = (((sx13) * (x12) + (sy13) *
		  (x12) + (sx21) * (x13) +
		  (sy21) * (x13)) // (2 *
		  ((y31) * (x12) - (y21) * (x13))));
			 
	g = (((sx13) * (y12) + (sy13) * (y12) +
		  (sx21) * (y13) + (sy21) * (y13)) //
		  (2 * ((x31) * (y12) - (x21) * (y13))));
	c = (-pow(x1, 2) - pow(y1, 2) -
		 2 * g * x1 - 2 * f * y1);
	h = -g;
	k = -f;
	sqr_of_r = h * h + k * k - c;
	r = round(sqrt(sqr_of_r), 5);
	centre=(k,h)
	radius=r
	return centre,radius

blank=np.zeros((600,600),dtype='uint8')
org_circle=cv.circle(blank,(200,300),100,255,1)
points_of_circle=[]
for i in range(len(org_circle)):
	for j in range(len(org_circle[0])):
		if org_circle[i][j]==255:
			points_of_circle.append([i,j])

# cv2.imshow(org_circle)

points_of_circle=np.array(points_of_circle)
noise = np.random.normal(0, 2, points_of_circle.shape)
new_points=points_of_circle+noise

new_blank=np.zeros((600,600),dtype='uint8')
for i in range(len(new_points)):
	new_blank[int(new_points[i][0]),int(new_points[i][1])]=255

# cv2_imshow(new_blank)

number_sp_noise=1500

for count in range(number_sp_noise):
	i_cord=np.random.randint(0,600) # 600 is chosen since the size of original image is (600,600)
	j_cord=np.random.randint(0,600)
	new_blank[i_cord][j_cord]=255


# cv2_imshow(new_blank)

white_coordinates=[]
for i in range(len(new_blank)):
	for j in range(len(new_blank[0])):
		if new_blank[i][j]==255:
			white_coordinates.append([i,j])

circle_details=[]
inlier_details=[]
number_of_trials=1
for iter_no in tqdm(range(number_of_trials)):

	test_points=[]
	while len(test_points)<=3:
		point=random.choice(white_coordinates)
		if point[0]!=0 or point[1]!=0:
		  test_points.append(point)

	point1=test_points[0]
	point2=test_points[1]
	point3=test_points[2]
	try:
		new_centre,new_radius=findCircle(point1[0],point1[1],point2[0],point2[1],point3[0],point3[1])
		inlier=0
		for point_c in white_coordinates:
			distance=abs(sqrt(((point_c[1]-new_centre[0])**2)+((point_c[0]-new_centre[1])**2))-new_radius)
		
			if distance<=5:
				inlier+=1
	except:
		pass		
    
	circle_details.append([new_centre,new_radius])
	inlier_details.append(inlier)

optimal_circle_details=circle_details[inlier_details.index(max(inlier_details))]
retraced_circle=cv.circle(new_blank,optimal_circle_details[0],int(optimal_circle_details[1]),255,1)
cv.imshow("NEW",retraced_circle)
cv.waitKey(0)