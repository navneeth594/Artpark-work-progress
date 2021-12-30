import cv2 as cv
import csv
import os
import numpy as np
from calcIOU import interoverunion
from tqdm import tqdm

low_thresh=np.arange(10,150,10)
high_thresh=np.arange(100,250,10)
real_images=os.listdir('./real')
ground_truth_images=os.listdir('./ground_truth')
real_images.sort()
ground_truth_images.sort()

# real_images=['RGB_006.jpg']
# ground_truth_images=['RGB_006.png']

for k in range(len(real_images)): 
	img=cv.imread("./real/"+real_images[k])
	ground_img=cv.imread('./ground_truth/'+ground_truth_images[k])
	img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

	ground_img=cv.cvtColor(ground_img,cv.COLOR_BGR2GRAY)

	iou_values=[]
	pairs=[]
	data=[['low thresh','high thresh','IOU']]
	for i in tqdm(range(len(low_thresh))):
		for j in range(len(high_thresh)):
			field=[]
			if low_thresh[i]<high_thresh[j]:
				canny=cv.Canny(img,low_thresh[i],high_thresh[j])
				iou=interoverunion(ground_img,canny)
				iou_values.append(round(iou*100,3))
				field.append(low_thresh[i])
				field.append(high_thresh[j])
				field.append(round(iou*100,3))
				pairs.append((low_thresh[i],high_thresh[j]))
				data.append(field)

				
	print("IMG NAME : ",real_images[k])
	print("Max IOU : ",max(iou_values))
	print("Thresh values : ",pairs[iou_values.index(max(iou_values))])

	optimal_low_thresh=pairs[iou_values.index(max(iou_values))][0]
	optimal_high_thresh=pairs[iou_values.index(max(iou_values))][1]
	canny_final=cv.Canny(img,optimal_low_thresh,optimal_high_thresh)
	cv.imwrite('./canny_results/Canny_'+real_images[k],canny_final)
	with open(real_images[k][:-4]+'.csv', 'w') as f:
	      
	    # using csv.writer method from CSV package
	    write = csv.writer(f)
	    write.writerows(data)

	print("image ",str(k)," Completed")