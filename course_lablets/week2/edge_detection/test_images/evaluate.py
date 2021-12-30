import cv2 as cv
import numpy as np
import os
import csv
from math import log10, sqrt
import matplotlib.pyplot as plt
from calcIOU import interoverunion
from tqdm import tqdm

real_images=os.listdir('./real')
ground_images=os.listdir('./ground_truth')
real_images.sort()
ground_images.sort()

for j in range(len(real_images)):
	
	img=cv.imread('./real/'+real_images[j])
	ground_img=cv.imread('./ground_truth/'+ground_images[j])
	img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	ground_img=cv.cvtColor(ground_img,cv.COLOR_BGR2GRAY)

	lap=cv.Laplacian(img,cv.CV_64F)
	lap=np.uint8(np.abs(lap))

	sobelx=cv.Sobel(img,cv.CV_64F,1,0)
	sobely=cv.Sobel(img,cv.CV_64F,0,1)
	abs_grad_x = cv.convertScaleAbs(sobelx)
	abs_grad_y = cv.convertScaleAbs(sobely)
	sobel = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


	thresh_values=np.arange(10,250)
	iou_values_lap=[]
	iou_values_sobel=[]
	lap_copy=lap.copy()
	sobel_copy=sobel.copy()

	for i in tqdm(range(len(thresh_values))):
		ret,lap=cv.threshold(lap_copy,thresh_values[i],255,cv.THRESH_BINARY)
		ret,sobel=cv.threshold(sobel_copy,thresh_values[i],255,cv.THRESH_BINARY)

		iou_lap=interoverunion(ground_img,lap)
		iou_sobel=interoverunion(ground_img,sobel)
		iou_values_lap.append(round((iou_lap*100),3))
		iou_values_sobel.append(round((iou_sobel*100),3))



	max_index_lap=iou_values_lap.index(max(iou_values_lap))
	print("Max iou Laplacian : ",max(iou_values_lap))
	print("threshold value Laplacian: ",thresh_values[max_index_lap])

	max_index_sobel=iou_values_sobel.index(max(iou_values_sobel))
	print("Max iou Sobel : ",max(iou_values_sobel))
	print("threshold value SObel: ",thresh_values[max_index_sobel])

	ret,lap=cv.threshold(lap_copy,thresh_values[max_index_lap],255,cv.THRESH_BINARY)
	ret,sobel=cv.threshold(sobel_copy,thresh_values[max_index_sobel],255,cv.THRESH_BINARY)

	plot1=plt.figure(1)
	plt.plot(thresh_values,iou_values_lap)

	plt.xlabel("Threshold values")
	plt.ylabel("IOU")
	plt.title(real_images[j])
	plt.plot(thresh_values[max_index_lap],max(iou_values_lap),marker='o',markerfacecolor="red")
	# plt.savefig("./graphs/laplacian/"+real_images[j])
	plt.show()

	plot2=plt.figure(2)
	plt.plot(thresh_values,iou_values_sobel)
	plt.xlabel("Threshold values")
	plt.ylabel("IOU")
	plt.title(real_images[j])
	plt.plot(thresh_values[max_index_sobel],max(iou_values_sobel),marker='o',markerfacecolor="red")
	# plt.savefig("./graphs/sobel/"+real_images[j])
	plt.show()
	cv.imwrite('./results/laplacian/'+real_images[j],lap)
	cv.imwrite('./results/sobel/'+real_images[j],sobel)

	
	print("Image "+str(j)+" completed")
	
	





