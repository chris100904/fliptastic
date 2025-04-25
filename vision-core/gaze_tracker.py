import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Landmark indices
LEFT_IRIS = [468, 469, 470, 471]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_IRIS = [473, 474, 475, 476]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

class StableGazeTracker:
    def __init__(self):
        self.calibration_points = []
        self.calibration_data = []
        self.calibrated = False
        self.tri = None
        self.gaze_history = []
        self.history_length = 5
        self.movement_threshold = 0.02
        self.last_gaze = None
        
    def add_calibration_point(self, point, iris_center):
        self.calibration_points.append(point)
        self.calibration_data.append(iris_center)
        
    def calibrate(self):
        if len(self.calibration_points) < 3:
            return False
            
        self.calibration_points = np.array(self.calibration_points, dtype=np.float32)
        self.calibration_data = np.array(self.calibration_data, dtype=np.float32)
        
        try:
            self.tri = Delaunay(self.calibration_data)
            self.calibrated = True
            return True
        except:
            return False
            
    def estimate_gaze(self, iris_center):
        if not self.calibrated or self.tri is None:
            return None
            
        simplex = self.tri.find_simplex(iris_center)
        if simplex == -1:
            confidence = 0.1
            # Estimate using nearest neighbor if outside convex hull
            distances = np.linalg.norm(self.calibration_data - iris_center, axis=1)
            nearest_idx = np.argmin(distances)
            estimated_point = self.calibration_points[nearest_idx]
        else:
            # Barycentric interpolation
            X = self.tri.transform[simplex, :2]
            r = iris_center - self.tri.transform[simplex, 2]
            bary = np.dot(r, X)
            bary_coords = np.append(bary, 1.0 - bary.sum())
            vertices = self.tri.simplices[simplex]
            estimated_point = np.dot(bary_coords, self.calibration_points[vertices])
            
            # Confidence based on distance to vertices
            distances = [np.linalg.norm(iris_center - self.calibration_data[v]) 
                       for v in vertices]
            confidence = 1.0 - (min(distances) / 0.1)
            confidence = np.clip(confidence, 0.1, 1.0)
        
        # Temporal smoothing
        self.gaze_history.append((estimated_point, confidence))
        if len(self.gaze_history) > self.history_length:
            self.gaze_history.pop(0)
            
        # Weighted average
        points, weights = zip(*self.gaze_history)
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
            smoothed_point = np.average(points, axis=0, weights=weights)
        else:
            smoothed_point = estimated_point
            
        # Dead zone for small movements
        if self.last_gaze is not None:
            distance = np.linalg.norm(smoothed_point - self.last_gaze)
            if distance < self.movement_threshold:
                return self.last_gaze
                
        self.last_gaze = smoothed_point
        return smoothed_point
        
    def get_screen_quadrant(self, point, screen_size):
        if point is None:
            return None
            
        x, y = point
        w, h = screen_size
        center_threshold = 0.25  # 25% of screen is considered center
        
        # Check if in center
        if (w*(0.5-center_threshold) < x < w*(0.5+center_threshold) and 
            h*(0.5-center_threshold) < y < h*(0.5+center_threshold)):
            return "CENTER"
            
        # Determine quadrant
        if x < w/2:
            return "TOP-LEFT" if y < h/2 else "BOTTOM-LEFT"
        else:
            return "TOP-RIGHT" if y < h/2 else "BOTTOM-RIGHT"

# Initialize
cap = cv2.VideoCapture(0)
tracker = StableGazeTracker()
calibration_step = 0
calibration_positions = [
    (0.5, 0.5),  # Center
    (0.1, 0.1),  # Top-left
    (0.9, 0.1),  # Top-right
    (0.1, 0.9),  # Bottom-left
    (0.9, 0.9),   # Bottom-right
    (0.5, 0.25),  # Top
    (0.75, 0.5),  # Top-right
    (0.5, 0.75),  # Bottom
    (0.25, 0.5),  # Bottom-left
    (0.5, 0.9)    # Bottom-right
]

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip and convert
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            mesh_points = results.multi_face_landmarks[0].landmark
            
            # Get iris centers
            left_iris = np.array([(mesh_points[p].x, mesh_points[p].y) for p in LEFT_IRIS])
            right_iris = np.array([(mesh_points[p].x, mesh_points[p].y) for p in RIGHT_IRIS])
            left_center = left_iris.mean(axis=0)
            right_center = right_iris.mean(axis=0)
            iris_center = (left_center + right_center) / 2  # Average of both eyes
            
            # Calibration mode
            if calibration_step < len(calibration_positions):
                target_pos = calibration_positions[calibration_step]
                cv2.putText(image, f"Look at the circle and press SPACE", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.circle(image, (int(target_pos[0]*w), int(target_pos[1]*h)), 20, (0, 0, 255), 2)
                
                key = cv2.waitKey(1)
                if key == 32:  # SPACE pressed
                    tracker.add_calibration_point(target_pos, iris_center)
                    calibration_step += 1
                    if calibration_step == len(calibration_positions):
                        if not tracker.calibrate():
                            print("Calibration failed! Please try again.")
                            calibration_step = 0
                            tracker = StableGazeTracker()
            else:
                # Gaze estimation mode
                gaze_point = tracker.estimate_gaze(iris_center)
                if gaze_point is not None:
                    gaze_pixel = (int(gaze_point[0]*w), int(gaze_point[1]*h))
                    quadrant = tracker.get_screen_quadrant(gaze_pixel, (w, h))
                    
                    # Visual feedback
                    cv2.circle(image, gaze_pixel, 10, (0, 255, 255), -1)
                    cv2.putText(image, f"Looking at: {quadrant}", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Print detailed info
                    print(f"Gaze point: {gaze_point}")
                    print(f"Screen quadrant: {quadrant}")
                    print("-" * 40)
        
        # Show calibration progress
        if calibration_step < len(calibration_positions):
            cv2.putText(image, f"Calibration step {calibration_step+1}/{len(calibration_positions)}", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Eye Tracking', image)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()