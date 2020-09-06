import math
import cv2
import numpy as np 

def distance_func(x1,y1,x2,y2):
    dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist

def compute_midpoints_and_threshhold_func(reference_point,inference_point,threshold=50):
    p1,p2,p3,p4 = reference_point
    
    q1,q2,q3,q4 = inference_point

    m1 = ((p1+p3)/2)
    m2 = ((q1+q3)/2)
    
    dist = distance_func(m1.item(),p4.item(),m2.item(),q4.item())
    #print("dist:",dist)
    #return dist
    return dist

def find_color_of_box(reference,lst_points,threshold):
    all_distances = []
    for i in range(len(lst_points)):
        val = compute_midpoints_and_threshhold_func(reference,lst_points[i])
        all_distances.append(val)
    try:
        if min(all_distances)<threshold:
            return 'red'
        else:
            return 'green'
    except:
        return 'green'

def draw_box(image,reference_point,color):
    start_point = (int(reference_point[0]),int(reference_point[1]))
    end_point = (int(reference_point[2]),int(reference_point[3]))
    # Blue color in BGR 
    if color == "red":
        color = (0, 0, 255) 
    else:
        color = (0,255,0)

    # Line thickness of 2 px 
    thickness = 2

    # Draw a rectangle with blue line borders of thickness of 2 px 
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image