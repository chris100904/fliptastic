import dlib
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_EYE_CORNER = 36
RIGHT_EYE_CORNER = 45
NOSE_TIP = 30

# adjust parameters
TILT_THRESHOLD = 5       # pixels difference for tilt detection
CONFIRMATION_FRAMES = 10  # number of frames to confirm intent

cap = cv2.VideoCapture(0)
confirmation_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        # --- Head Tilt Detection ---
        left_eye_y = landmarks.part(LEFT_EYE_CORNER).y
        right_eye_y = landmarks.part(RIGHT_EYE_CORNER).y
        tilt = right_eye_y - left_eye_y  # Positive = tilting left
        
        # --- Nose Position ---
        nose_x = landmarks.part(NOSE_TIP).x
        nose_y = landmarks.part(NOSE_TIP).y
        
        # --- Visual Feedback ---
        cv2.putText(frame, f"Tilt: {tilt:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw landmarks
        cv2.circle(frame, (int(nose_x), int(nose_y)), 5, (0, 0, 255), -1)
        cv2.circle(frame, (landmarks.part(LEFT_EYE_CORNER).x, int(left_eye_y)), 3, (255, 0, 0), -1)
        cv2.circle(frame, (landmarks.part(RIGHT_EYE_CORNER).x, int(right_eye_y)), 3, (255, 0, 0), -1)
        
        # --- Intent Verification Logic ---
        if abs(tilt) > TILT_THRESHOLD:
            confirmation_counter += 1
            cv2.putText(frame, "INTENT DETECTED!", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if confirmation_counter >= CONFIRMATION_FRAMES:
                print("PAGE TURN TRIGGERED!")  # placeholder
                confirmation_counter = 0
        else:
            confirmation_counter = max(0, confirmation_counter - 1)
    
    cv2.imshow('Head Pose Detection', frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()