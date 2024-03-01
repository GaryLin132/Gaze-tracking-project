import cv2
import mediapipe as mp
import numpy as np
import time
import functions as f


# landmark indices
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
LEFT_CORNER = [362, 374, 263, 386]
RIGHT_CORNER = [33, 145, 133, 159]
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


def eye_vector(results, image, mesh_points, head_vector, depth, scaling):
    img_h, img_w, img_c = image.shape
    
    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
    center_left = np.array([l_cx, l_cy], dtype=np.int32)
    center_right = np.array([r_cx, r_cy], dtype=np.int32)
    cv2.circle(image, center_left, int(l_radius), (255,0,255), 1, cv2.LINE_AA)
    cv2.circle(image, center_right, int(r_radius), (255,0,255), 1, cv2.LINE_AA)

    # plot iris
    cv2.circle(image, mesh_points[473], 1, (255,0,255), 1, cv2.LINE_AA)
    cv2.circle(image, mesh_points[468], 1, (255,0,255), 1, cv2.LINE_AA)

    # get eye center
    # leye = (mesh_points[LEFT_CORNER[0]]+mesh_points[LEFT_CORNER[2]])/2
    leye = 0
    reye = 0
    # reye = (mesh_points[RIGHT_CORNER[0]]+mesh_points[RIGHT_CORNER[2]])/2
    for i in range(0,4):
        cv2.circle(image, mesh_points[LEFT_CORNER[i]], 1, (255,0,255), 4, cv2.LINE_AA)
        leye += mesh_points[LEFT_CORNER[i]]   # average from four eye corners
        reye += mesh_points[RIGHT_CORNER[i]]
    leye = (leye)/4
    reye = (reye)/4
    # cv2.circle(image, mesh_points[LEFT_CORNER[0]], 1, (255,0,255), 4, cv2.LINE_AA)
    # cv2.circle(image, mesh_points[LEFT_CORNER[2]], 1, (255,0,255), 4, cv2.LINE_AA)
    leye_center = np.array(leye,dtype=np.int32)
    reye_center = np.array(reye,dtype=np.int32)
    cv2.circle(image, leye_center, 1, (0,0,255), 1, cv2.LINE_AA)
    cv2.circle(image, reye_center, 1, (0,0,255), 1, cv2.LINE_AA)
    # get eye vector and average it
    lefteye_vec,llen = f.calculate_eyeVector(mesh_points[473], leye_center)
    righteye_vec,rlen = f.calculate_eyeVector(mesh_points[468], reye_center)
    aver_vec = (lefteye_vec+righteye_vec)/2
    aver_len = (llen+rlen)/2
    # print("average_vec: ",aver_vec)
    x,y = np.array(aver_vec, dtype=np.float64)
    # print("x: ",x , "y: ",y)

    p1 = (mesh_points[473][0], mesh_points[473][1])
    p2 = (p1[0] + int(x * 100) , p1[1] + int(y * 100))
    
    cv2.line(image, p1, p2, (255, 0, 0), 3)

    eye_vector = f.calculate_eyeAngle(head_vector, aver_vec, aver_len, depth, scaling)

    return image, eye_vector

def blink_detect():
    pass
 