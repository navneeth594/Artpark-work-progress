import cv2 as cv
import numpy as np
import pyrealsense2 as rs

def erode_dilate(image):
    """
    Function to perform morphological operations to fill the 'holes' in the threshold image
    """
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv.erode(image, kernel, iterations=1)
    img_dilation = cv.dilate(img_erosion, kernel, iterations=2)

    return img_dilation

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

depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics


def convert_to_camera_cord(u, v, d,intrin):
    x_over_z = (intrin.ppx - u) / intrin.fx
    y_over_z = (intrin.ppy - v) / intrin.fy
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = -1*(x_over_z * z)
    y = -1*(y_over_z * z)
    # x=round(x,2)
    # y=round(y,2)
    # z=round(z,2)
    return x, y, z
def convert_to_2dpixel(X3d,Y3d,Z3d,intrin):
    x=int((intrin.fx*X3d+intrin.ppx*Z3d)/Z3d)
    y=int((intrin.fy*Y3d+intrin.ppy*Z3d)/Z3d)
    return x,y


def camera_to_world(Xc,Yc,Zc,transformation_matrix):
    transformation_matrix_inv=np.linalg.inv(transformation_matrix)
    camera_cords=np.array([[Xc],[Yc],[Zc],[1]])
    world_cords=np.matmul(transformation_matrix_inv,camera_cords)
    print(world_cords)
    Xw,Yw,Zw=world_cords[0][0],world_cords[1][0],world_cords[2][0]
    return Xw,Yw,Zw

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
count=0

last_row=np.array([[0,0,0,1]])
transformation=np.concatenate((transformation,last_row))

while True:        # Get frameset of color and depth
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = frames.get_color_frame()
    

    # Validate that both frames are valid
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image=cv.cvtColor(color_image,cv.COLOR_BGR2RGB)
    cv.imshow("IMAGE",color_image)
    # cv.imwrite('plain_img.jpg',color_image)
    key=cv.waitKey(1)
    if key==ord('s'):
        count+=1
        image=color_image.copy()
        img_gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        # img_hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
        # h,s,v=cv.split(img_hsv)
        mask=cv.inRange(img_gray, 0,20)
        # cv.imshow('GRAY',mask)
        image_new=image.copy()
        # mask1=cv.bitwise_not(mask)
        mask1=erode_dilate(mask)
        canny=cv.Canny(mask1,100,175)
        # circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1.2, 100)
        # print(circles)
        contours, hierarchy = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        # cv.imshow("MASK",mask1)
            # drawing = np.zeros((mask1.shape[0], mask1.shape[1], 3), np.uint8)
        contour_points=[]
        max_contour=max(contours,key=cv.contourArea)
        cv.drawContours(image,contours,-1,(0,255,0),3)
        for single_contour in contours:
            for point in single_contour:
                contour_points.append(point[0])
        contour_points=np.array(contour_points)

        final_contour_points=[]
        for i in range(len(contour_points)):
            contour_point_depth=get_depth_at_pixel(depth_frame,contour_points[i][0],contour_points[i][1])
            if contour_point_depth!=0:

                final_contour_points.append([contour_points[i][0],contour_points[i][1],contour_point_depth])
        camera_cordinate_points=[]
        for point2d in final_contour_points:
            X3d,Y3d,Z3d=convert_to_camera_cord(point2d[0],point2d[1],point2d[2],depth_intrin)
            camera_cordinate_points.append([X3d,Y3d,Z3d])
        camera_cordinate_points=np.array(camera_cordinate_points)
        centroid=np.mean(camera_cordinate_points,axis=0)
        centroid_x,centroid_y,centroid_z=centroid[0],centroid[1],centroid[2]
        centre2dx,centre2dy=convert_to_2dpixel(centroid_x,centroid_y,centroid_z,depth_intrin)
        centroid_x=round(centroid_x*100,2)
        centroid_y=round(centroid_y*100,2)

        centroid_z=round(centroid_z*100,2)
        cv.putText(image,"X: "+str(centroid_x)+' Y: '+str(centroid_y)+' Z: '+str(centroid_z),(0,50),cv.FONT_HERSHEY_SIMPLEX,1,
                        (0, 0,255), 2)
        # xw,yw,zw=camera_to_world(centroid_x,centroid_y,centroid_z,transformation)
        # print(xw,' ',yw,' ',zw)

        image=cv.circle(image,(centre2dx,centre2dy),5,(255,0,0),-1)
        cv.imshow("FInal image",image)
        cv.imwrite('with_text_centre'+str(count)+'.jpg',image)

    if key==ord('q'):
        break


pipeline.stop()



