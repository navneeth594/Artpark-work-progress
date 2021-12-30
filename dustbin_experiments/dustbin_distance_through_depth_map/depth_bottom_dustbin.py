import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))


# Start streaming
pipeline.start(config)
count_no=0

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

def erode_dilate(image):
    """
    Function to perform morphological operations to fill the 'holes' in the threshold image
    """
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(image, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=2)

    return img_dilation
def segment_dustbin(img):
    img_hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(img_hsv)

    mask=cv2.inRange(s, 0,140)
    mask1=cv2.bitwise_not(mask)


    mask1=erode_dilate(mask1)
    # cv.imshow("MASK",mask1)
    contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # drawing = np.zeros((mask1.shape[0], mask1.shape[1], 3), np.uint8)
    contour_points=[]
    max_contour=max(contours,key=cv2.contourArea) 

    # for i in range(len(max_contour)):
    #     contour_points.append(max_contour[i][0])

    # contour_points=np.array(contour_points)
    # number_of_rows=contour_points.shape[0]
    # choices=np.random.choice(number_of_rows,60)
    white_points=[]
    for i in range(len(mask1)):
        for j in range(len(mask1[0])):
            if mask1[i][j]==255:
                white_points.append([j,i])
    desired_points=white_points[len(white_points)-15:len(white_points)]
    

    return contour_points,desired_points




try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            
            images = np.hstack((color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)

        if key == ord("q"):
            # count_no+=1
            # cv2.imwrite('./dustbin_images/for_heatmap/110_'+str(count_no)+'.jpg',color_image)
            break
        if key == ord("d"):
            count_no+=1
            points_dustbin,few_points=segment_dustbin(color_image)
            distances_arr=[]
            for point in few_points:
                distance=get_depth_at_pixel(depth_frame,point[0],point[1])
                distance=round(distance*100,1)
                distances_arr.append(distance)

            distances_arr=np.array(distances_arr)
            distances_arr=distances_arr[distances_arr!=0]

            
            distances_arr=distances_arr[distances_arr<450]
            avg_distance=round(np.mean(distances_arr),1)

            # result_distance=round(sqrt((avg_distance**2)-(90**2)),1)
            result_distance=avg_distance

            print(result_distance)
            # bins=np.arange(100,400,1)
            # plt.hist(distances_arr,bins=bins)
            true_distance='370'
            # plt.savefig('./depth_method/day2_afternoon/hist_plots/'+true_distance+'_'+str(count_no)+'.jpg')
            # plt.clf()
            file1 = open("./depth_files_new/distances.txt","a")
            file1.write("\n"+true_distance+"_"+str(count_no)+"          "+str(result_distance)+'\n')
            file1.close()
            cv2.imwrite('./depth_files_new/'+true_distance+'_'+str(count_no)+'.jpg',images)  
            # cv2.imwrite('./top_floor_homo/dustbin_images/'+true_distance+'_'+str(count_no)+'.jpg',color_image)    

        # cv2.imwrite('./countertop_video/counter_'+str(count_no)+'.jpg',images)
        cv2.waitKey(1)
        
finally:

    # Stop streaming
    pipeline.stop()