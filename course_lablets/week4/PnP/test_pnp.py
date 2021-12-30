import cv2
import numpy as np

camera_matrix=np.array([[699.713,0,641.876],
						[0,699.713,377.047],
						[0,0,1]])


dist_coeffs = np.zeros((4,1))

points_2D = np.array([
                        (630, 597), 
 
                        (632, 444),  
 
                        (820, 563),  
 
                        (825, 447),  
 
                        (432, 536),  
 
                        (434, 422),

                        (557, 605),

                        (559, 443)   
 
                      ], dtype="double")
 
 
 
points_3D = np.array([
 
                      (2.5, 0, 2.85),       
 
                      (2.5, 0, 10.7),  
 
                      (17.95, 0, 2.95),
 
                      (18.25, 0, 10.1),  
 
                      (0, 18.8, 2.9),
 
                      (0, 19.1, 11.1),
                        (0, 2.4, 2.5),
                        (0, 2.55, 10.7)
 
  
 
                     ])


success, rotation_vector, translation_vector = cv2.solvePnP(points_3D, points_2D, camera_matrix, dist_coeffs, flags=0)

print(rotation_vector)
print('============')
print(translation_vector)
R,_=cv2.Rodrigues(rotation_vector)

trans=np.concatenate((R,translation_vector),axis=1)
test_point=np.array([[2.5], [0], [2.85],[1]])
new=np.matmul(trans,test_point)
new_x=np.matmul(camera_matrix,new)
print(new_x[0]//new_x[2])
print(new_x[1]//new_x[2])
