import cv2
import numpy as np
import math

fx, fy = [971.2388, 971.5315]      # relative to pixel units
cx, cy = [667.937, 379.023]
k1, k2 = [0.0237, -0.0956]         # radial distortion
p1, p2 = [0, 0]                    # tangential distortion
camera_matrix = np.array([ [fx, 0, cx],
                            [0, fy,cy],
                            [0, 0, 1]],dtype=np.float64)
distort_matrix = np.array([k1, k2, p1, p2], dtype=np.float64)

class cam_parameters:
    def __init__(self, image):
        self.img_h = image.shape[0]
        self.img_w = image.shape[1]
        
    def get_param(self):
        f = self.img_w
        cx = self.img_h/2
        cy = self.img_w/2
        camera_matrix = np.array([ [f, 0, cx],
                                    [0, f,cy],
                                    [0, 0, 1]],dtype=np.float64)
        distort_matrix = np.array([0, 0, 0, 0], dtype=np.float64)
        return camera_matrix, distort_matrix

    

iris_length_mm = 11.7
eye_radius_mm = 13.5

def convert_to_mm(length, depth):
    length_mm = depth*length/fx
    return length_mm

def calculate_distance(p1, p2):
    p1x, p1y = p1.ravel()
    p2x, p2y = p2.ravel()
    distance = math.sqrt((p1x-p2x)**2+(p1y-p2y)**2)
    return distance

def calculate_depth(iris_length_px):
    '''knowing that radius of iris is 11.7 mm'''
    depth = iris_length_mm*fx/iris_length_px  # fx almost equal to fy
    return depth                         # unit mm


def calculate_eyeVector(iris_point, rotation_center):
    # assume rotation_center equals to eye center at first
    len = calculate_distance(iris_point, rotation_center)
    # print("len: ",len)
    x_dis = iris_point[0] - rotation_center[0]
    y_dis = iris_point[1] - rotation_center[1]
    if len>0:
        vector = np.array([x_dis, y_dis], dtype=np.float64)   # used to be x_dis/len
    else:
        vector = np.array([0,0])
    return vector, len

def calculate_eyeAngle(head_angles, eye_angles, len, depth, scaling):    # eye_angles and len come from calculate_eyeVector
    if abs(eye_angles[0])<=0.0001 and abs(eye_angles[1])<=0.0001:
        return (0,0)
    else:
        # project head angle to eye gazing angle
        head_vec = np.array([head_angles[0], head_angles[1]])  # due to camera coordinate difference
        head_len = math.sqrt(head_vec[0]**2 + head_vec[1]**2)
        # head_z = head_angles[2] * 360
        eye_vec = np.array([eye_angles[0], eye_angles[1]])
        eye_len = math.sqrt(eye_vec[0]**2 + eye_vec[1]**2)
        # eye_z = eye_angles[2]
        alpha1 = math.atan2(head_vec[1], head_vec[0])  # atan2 returns radian
        alpha2 = math.atan2(eye_vec[1], eye_vec[0])
        # alpha1 = math.atan(head_vec[1]/head_vec[0])  # atan2 returns radian
        # alpha2 = math.atan(eye_vec[1]/eye_vec[0])
        new_headLen = head_len*abs(math.cos(alpha1+alpha2))
        new_headVec = np.array([math.cos(alpha2)*new_headLen, math.sin(alpha2)*new_headLen])


        # calculate tan(head_project+eye)
        tan_newhead = new_headLen*scaling['head']/depth
        cos_newhead = 1/math.sqrt(1+tan_newhead**2)    # just consider positive because length still be positive
        tan_eye = convert_to_mm(len, depth)/(cos_newhead*eye_radius_mm)
        tan_neweye = (tan_newhead+tan_eye)/(1-(tan_newhead*tan_eye)) - tan_newhead   # calculated seperately because head and eye have different scaling
                                                                                     
        return (tan_eye*math.cos(alpha2), tan_eye*math.sin(alpha2))   # use to use tan_neweye

        # direction for improvement: try to reduce to calculation of tan and cos to reduce dot jumping

# add abs at line 77




    

