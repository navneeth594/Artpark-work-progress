import cv2 as cv

img=cv.imread('./PnP_images/saved_box_img_2.jpg')
# img_hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
b,g,r=cv.split(img)
cv.imshow("B",b)
cv.imshow("G",g)
cv.imshow("R",r)
cv.waitKey(0)