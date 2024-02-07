import cv2
import mediapipe as mp
import numpy as np
import functions as f
import math

def head_vector(results, image, scaling):
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    for face_landmarks in results.multi_face_landmarks:
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z*3000 )  # use to be 3000
                    # print("nose_3d x: ",lm.x * img_w,"nose_3d y: ",lm.y * img_h,"nose_3d z: ",lm.z*3000)
                    # print("nose_3d y: ",lm.y * img_h)
                    # print("nose_3d z: ",lm.z*3000)

                x, y = int(lm.x * img_w), int(lm.y * img_h)

                # Get the 2D Coordinates
                face_2d.append([x, y])

                # Get the 3D Coordinates
                face_3d.append([x, y, lm.z])
                    
        
        # Convert it to the NumPy array
        face_2d = np.array(face_2d, dtype=np.float64)

        # Convert it to the NumPy array
        face_3d = np.array(face_3d, dtype=np.float64)

        cam_matrix, dist_matrix = f.cam_parameters(image).get_param()

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
        # print("x: ",x, "y: ",y, "z: ",z)

        # # See where the user's head tilting
        # if y < -10:
        #     text = "Looking Left"
        # elif y > 10:
        #     text = "Looking Right"
        # elif x < -10:
        #     text = "Looking Down"
        # elif x > 10:
        #     text = "Looking Up"
        # else:
        #     text = "Forward"

        # Display the nose direction
        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        # print(nose_3d_projection.ravel()[0])

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * scaling['head']) , int(nose_2d[1] - x * scaling['head']))
        # p2 = (int(nose_3d_projection.ravel()[0]) , int(nose_3d_projection.ravel()[1]))
        vector = (y, -x)
        
        cv2.line(image, p1, p2, (255, 0, 0), 3)

        # Add the text on the image
        # cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        # cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    
    return image, vector