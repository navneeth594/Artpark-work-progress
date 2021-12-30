from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv

def plot_graph(csv_file):
	data=pd.read_csv(csv_file)
	x=data['low thresh'].tolist()
	y=data['high thresh'].tolist()
	z=data['IOU'].tolist()

	max_ind=z.index(max(z))

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter3D(x, y, z, c=z, cmap='Greens');
	ax.scatter(x[max_ind],y[max_ind],max(z),c='red',marker='o',s=250)
	plt.show()


filenames=['RGB_114.csv','RGB_134.csv','RGB_140.csv','RGB_159.csv']
plot_graph('RGB_159.csv')

# img1=cv.imread('./real/RGB_114.jpg')
# img1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
# canny1=cv.Canny(img1,140,240)
# cv.imwrite('./canny_results/RGB_114.jpg',canny1)

# img2=cv.imread('./real/RGB_134.jpg')
# img2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
# canny2=cv.Canny(img2,140,240)
# cv.imwrite('./canny_results/RGB_134.jpg',canny2)

# img3=cv.imread('./real/RGB_140.jpg')
# img3=cv.cvtColor(img3,cv.COLOR_BGR2GRAY)
# canny3=cv.Canny(img3,140,240)
# cv.imwrite('./canny_results/RGB_140.jpg',canny3)

# img4=cv.imread('./real/RGB_159.jpg')
# img4=cv.cvtColor(img4,cv.COLOR_BGR2GRAY)
# canny4=cv.Canny(img4,140,240)
# cv.imwrite('./canny_results/RGB_159.jpg',canny4)