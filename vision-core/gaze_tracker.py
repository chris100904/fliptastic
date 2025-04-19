import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

LEFT_IRIS = [468, 469, 470, 471]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

RIGHT_IRIS = [473, 474, 475, 476]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)  # mirror display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            mesh_points = results.multi_face_landmarks[0].landmark
            
            left_iris = [mesh_points[p] for p in LEFT_IRIS]
            left_eye = [mesh_points[p] for p in LEFT_EYE]
            
            right_iris = [mesh_points[p] for p in RIGHT_IRIS]
            right_eye = [mesh_points[p] for p in RIGHT_EYE]
            
            h, w, _ = frame.shape
            left_iris_px = [(int(p.x * w), int(p.y * h)) for p in left_iris]
            right_iris_px = [(int(p.x * w), int(p.y * h)) for p in right_iris]
            
            # draw left eye landmarks (green)
            for point in left_iris_px + [(int(mesh_points[p].x * w), int(mesh_points[p].y * h)) for p in LEFT_EYE]:
                cv2.circle(frame, point, 1, (0, 255, 0), -1)
                
            # draw right eye landmarks (red)
            for point in right_iris_px + [(int(mesh_points[p].x * w), int(mesh_points[p].y * h)) for p in RIGHT_EYE]:
                cv2.circle(frame, point, 1, (0, 0, 255), -1)
            
            print("\nLeft Iris Center:", (left_iris[0].x, left_iris[0].y))
            print("Right Iris Center:", (right_iris[0].x, right_iris[0].y))
        
        cv2.imshow("Eye Tracking", frame)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()