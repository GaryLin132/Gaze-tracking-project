import cv2
import mediapipe as mp
import numpy as np
import time
import functions as f
import head_pos
import eye_track
import pyautogui as pyGUI

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# the parameter to be determined is (head_scaling, eye_scaling multiple) which is (0.3, 26)
head_scaling = 0.015                    # 0.15*1.1
eye_scaling = head_scaling*5         # 3*1.1
scaling = {'head':head_scaling, 'eye':eye_scaling}

LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

### webcam
cap = cv2.VideoCapture(0)

### read in video in vids and out write the result
# cap = cv2.VideoCapture('E:\\projects\\gaze-tracking-project\\vids\\eye_test.mp4')
# filename = 'eye_test.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(filename, fourcc, 20, (1280,  720))  # the out in the bottom has to be uncommented

# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # cv2.CAP_DSHOW and cv2.destroyAllWindows() is to deal with ip camera
# address = "http://192.168.0.46:8080/video"
# cap.open(address)
# remember to connect to same WiFi

pyGUI.PAUSE = 0   # so that pyGUI actions don't hesitate
pyGUI.FAILSAFE = False
last_pos = (500, 500)
buffer_count = 0
# determine scaling parameter

while cap.isOpened():
    success, image = cap.read()

    if not success:
        break
    # cv2.namedWindow('Head Pose Estimation', cv2.WINDOW_FULLSCREEN)
    # cv2.setWindowProperty('Head Pose Estimation', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FULLSCREEN)

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    

    if results.multi_face_landmarks:
        mesh_points=np.array([np.multiply([p.x, p.y], [image.shape[1], image.shape[0]]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        # calculate iris length pixel
        iris_lenPx_left = f.calculate_distance(mesh_points[LEFT_IRIS[0]], mesh_points[LEFT_IRIS[2]])
        depth_left = f.calculate_depth(iris_lenPx_left)

        iris_lenPx_right = f.calculate_distance(mesh_points[RIGHT_IRIS[0]], mesh_points[RIGHT_IRIS[2]])
        depth_right = f.calculate_depth(iris_lenPx_right)

        depth = (depth_left + depth_right)/2        # unit mm
        # print("depth: ",depth)
        scaling['head']=head_scaling*depth
        scaling['eye']=eye_scaling*depth

        image_head, head_vector= head_pos.head_vector(results, image, scaling)
        image_headEye, eye_vector = eye_track.eye_vector(results, image_head, mesh_points, head_vector, depth, scaling)

        # draw from middle of two eyes
        # mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        mid_eye = (mesh_points[362]+mesh_points[133])/2
        p1 = ( int(mid_eye[0]), int(mid_eye[1]) )
        
        x2 = p1[0]+int(eye_vector[0])
        y2 = p1[1]+int(eye_vector[1])
        if x2>image.shape[1]:
            x2 = int(image.shape[1])
        if y2>image.shape[0]:
            y2 = image.shape[0]
        p2 = (x2, y2)


        cv2.line(image_headEye, p1, p2, (0, 255, 0), 3)
        cv2.circle(image_headEye, p2, 1, (0,0,0), 10, cv2.LINE_AA)
        cv2.putText(image_headEye, f'DEPTH: {int(depth/10)}', (20,300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        # control computer
        last_pos_arr = np.asarray(last_pos)  # can try use pyGUI.position()
        p2_arr = np.asarray(p2)     # convert to array so as calculate distance can work
        buffer_point = (int((last_pos[0]+p2[0])/2), int((last_pos[1]+p2[1])/2))
        if f.calculate_distance(p2_arr, last_pos_arr)>=30:
            # print("dis: ",f.calculate_distance(p2_arr, last_pos_arr))
            pyGUI.moveTo(buffer_point)
            last_pos = p2
        if p2[1]>image_headEye.shape[0]*0.7 and p2[0]>image_headEye.shape[1]*0.7 :   # beware image.shape[0] is height
            # pyGUI.typewrite('Next Page!')
            # pyGUI.press('right')
            # pyGUI.moveTo(buffer_point)
            cv2.circle(image_headEye, p2, 1, (0,0,255), 50, cv2.LINE_AA)
            buffer_count+=1
            if buffer_count>=fps*0.1:   # means after about 0.1 sec
                pyGUI.scroll(-buffer_count*30)
                # cv2.circle(image_headEye, p2, 1, (0,0,255), 50, cv2.LINE_AA)
                print("see right")
                buffer_count = 0
        
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        # print("FPS: ", fps)

        # cv2.putText(image_headEye, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        # mp_drawing.draw_landmarks(
        #             image=image,
        #             landmark_list=face_landmarks,
        #             connections=mp_face_mesh.FACEMESH_CONTOURS,
        #             landmark_drawing_spec=drawing_spec,
        #             connection_drawing_spec=drawing_spec)

    # cv2.setWindowProperty('Head Pose Estimation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.namedWindow('Head Pose Estimation', cv2.WINDOW_FULLSCREEN)
    # cv2.setWindowProperty('Head Pose Estimation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Head Pose Estimation', image)
    #### cv2.imwrite(filename,image)
    # out.write(image)

    if cv2.waitKey(5) & 0xFF == ord('q'):     # 0xFF == 27
        break


cap.release()
# out.release()
cv2.destroyAllWindows()
