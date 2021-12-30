import cv2 as cv

l=['saved_box_img_2.jpg','saved_box_img_3.jpg','saved_box_img_4.jpg','saved_box_img_5.jpg','saved_box_img_6.jpg','saved_box_img_7.jpg']

for i in l:
	img=cv.imread('./PnP_images/'+i)
	new_img=img[:,:img.shape[1]//2,:]

	cv.imwrite('./PnP_images/'+i,new_img)
