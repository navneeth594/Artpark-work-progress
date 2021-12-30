import pyrealsense2 as rs
import numpy as np
import cv2 as cv
config = rs.config()
# config.enable_stream(rs.stream.depth, rs.format.z16, 30)
# config.enable_stream(rs.stream.color,720,1280, rs.format.bgr8, 30)
pipeline = rs.pipeline()
pipe_profile = pipeline.start(config)
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
extrinsics = color_frame.profile.get_extrinsics_to(depth_frame.profile)
depth_sensor = pipe_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(depth_intrin)
align_to = rs.stream.color
align = rs.align(align_to)
def get_depth_at_pixel(depth_frame, pixel_x, pixel_y):
    """
    Get the depth value at the desired image point
    Parameters:
    -----------
    depth_frame      : rs.frame()
                           The depth frame containing the depth information of the image coordinate
    pixel_x              : double
                           The x value of the image coordinate
    pixel_y              : double
                            The y value of the image coordinate
    Return:
    ----------
    depth value at the desired pixel
    """
    return depth_frame.as_depth_frame().get_distance(round(pixel_x), round(pixel_y))
def calculate_depth_corner_points(color_img,depth_fr):
	color_img_hsv=cv.cvtColor(color_img,cv.COLOR_RGB2HSV)
	h,s,v=cv.split(color_img_hsv)
	mask=cv.inRange(s, 0,140)
	mask1=cv.bitwise_not(mask)
	mask1=erode_dilate(mask1)
	contours, hierarchy = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     # drawing = np.zeros((mask1.shape[0], mask1.shape[1], 3), np.uint8)

	max_contour=max(contours,key=cv.contourArea)
	hull = cv.convexHull(max_contour)
	extLeft = tuple(hull[hull[:, :, 0].argmin()][0])
	extRight = tuple(hull[hull[:, :, 0].argmax()][0])
	extTop = tuple(hull[hull[:, :, 1].argmin()][0])
	extBot = tuple(hull[hull[:, :, 1].argmax()][0])
	left_point_distance=get_depth_at_pixel(depth_fr,extLeft[0],extLeft[1])
	right_point_distance=get_depth_at_pixel(depth_fr,extRight[0],extRight[1])

	return extLeft,extRight,left_point_distance,right_point_distance

def erode_dilate(image):
    """
    Function to perform morphological operations to fill the 'holes' in the threshold image
    """
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv.erode(image, kernel, iterations=1)
    img_dilation = cv.dilate(img_erosion, kernel, iterations=2)

    return img_dilation
# Streaming loop
image_name="50_150"
while True:        # Get frameset of color and depth
	frames = pipeline.wait_for_frames()

	depth_frame = frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
	color_frame = frames.get_color_frame()
	

	# Validate that both frames are valid
	depth_image = np.asanyarray(depth_frame.get_data())
	color_image = np.asanyarray(color_frame.get_data())
	color_image=cv.cvtColor(color_image,cv.COLOR_BGR2RGB)
	cv.imshow("IMAGE",color_image)
	key=cv.waitKey(1)
	if key==ord('s'):
		l,r,ld,rd=calculate_depth_corner_points(color_image,depth_frame)
		print(ld,rd)
		color_image=cv.circle(color_image,l,5,(0,255,0),-1)
		color_image=cv.circle(color_image,r,5,(0,255,0),-1)
		midpoint=((l[0]+r[0])//2,(l[1]+r[1])//2)
		if ld!=0 and rd!=0:
			depth=(ld+rd)/2
		elif ld==0:
			depth=rd
		elif rd==0:
			depth=ld
		result=rs.rs2_deproject_pixel_to_point(depth_intrin, [midpoint[0], midpoint[1]], depth)
		X,Y,Z=result[0],result[1],result[2]
		X=round(X*100,0)
		Y=round(Y*100,0)
		Z=round(Z*100,0)
		print(X,Y,Z)
		cv.putText(color_image,"X : "+str(X)+' Y : '+str(Y)+" Z : "+str(Z),(0,100),cv.FONT_HERSHEY_SIMPLEX,1,
                        (0, 0,255), 2)
		color_image=cv.circle(color_image,midpoint,5,(0,255,0),-1)


		cv.imshow("NEW",color_image)
		cv.imwrite('./'+image_name+".jpg",color_image)

	if key==ord('q'):
		break

pipeline.stop()

