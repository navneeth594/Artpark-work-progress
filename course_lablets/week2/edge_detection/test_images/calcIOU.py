
def interoverunion(img1,img2):
	
	num=0
	den=0
	img1_copy=img1.copy()
	img2_copy=img2.copy()
	img1_copy[img1_copy==255]=1
	img2_copy[img2_copy==255]=1
	# for i in range(len(img1)):
	# 	for j in range(len(img1[0])):
	# 		if img1[i][j]==255:
	# 			img1_copy[i][j]=1
	# 		if img2[i][j]==255:
	# 			img2_copy[i][j]=1
	


	for i in range(len(img1)):
		for j in range(len(img1[0])):

			num=num+(img1_copy[i][j] & img2_copy[i][j])
			den=den+(img1_copy[i][j] | img2_copy[i][j])

	return num/den

