# import cv2
# import mediapipe as mp
# import numpy as np
# from scipy.spatial import Delaunay
# from collections import deque
# from typing import Tuple, Optional
# import time

# mp_drawing = mp.solutions.drawing_utils
# mp_face_mesh = mp.solutions.face_mesh

# # Landmark indices
# LEFT_IRIS = [468, 469, 470, 471]
# LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# RIGHT_IRIS = [473, 474, 475, 476]
# RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# class KalmanFilter:
#     """Simple Kalman filter for 2D points with stability improvements"""
#     def __init__(self, process_noise=1e-5, measurement_noise=1e-2):
#         self.state = None
#         self.covariance = np.eye(2) * 0.1
#         self.process_noise = process_noise
#         self.measurement_noise = measurement_noise
#         self.transition_matrix = np.eye(2)
#         self.observation_matrix = np.eye(2)
        
#     def update(self, measurement):
#         if self.state is None:
#             self.state = np.array(measurement, dtype=np.float32)
#             return self.state
            
#         # Prediction
#         predicted_state = self.transition_matrix @ self.state
#         predicted_cov = self.transition_matrix @ self.covariance @ self.transition_matrix.T + np.eye(2) * self.process_noise
        
#         # Update
#         innovation = measurement - self.observation_matrix @ predicted_state
#         innovation_cov = self.observation_matrix @ predicted_cov @ self.observation_matrix.T + np.eye(2) * self.measurement_noise
        
#         # Numerically stable Kalman gain calculation
#         try:
#             kalman_gain = predicted_cov @ self.observation_matrix.T @ np.linalg.inv(innovation_cov)
#         except np.linalg.LinAlgError:
#             kalman_gain = np.zeros((2, 2))
        
#         self.state = predicted_state + kalman_gain @ innovation
#         self.covariance = (np.eye(2) - kalman_gain @ self.observation_matrix) @ predicted_cov
        
#         # Clamp state to reasonable values
#         self.state = np.clip(self.state, -1.0, 2.0)
#         return self.state

# class EnhancedGazeTracker:
#     def __init__(self, screen_size=(1920, 1080)):
#         self.calibration_points = []
#         self.calibration_data = []
#         self.calibrated = False
#         self.tri = None
#         self.screen_size = screen_size
        
#         # Tracking parameters with more conservative settings
#         self.left_kf = KalmanFilter(process_noise=1e-5, measurement_noise=1e-2)
#         self.right_kf = KalmanFilter(process_noise=1e-5, measurement_noise=1e-2)
#         self.gaze_history = deque(maxlen=10)  # Smaller history window
#         self.movement_threshold = 0.02
#         self.last_gaze = None
#         self.last_velocity = np.zeros(2)
        
#         # Confidence tracking
#         self.left_confidence = 0
#         self.right_confidence = 0
#         self.blink_counter = 0
#         self.min_confidence = 0.4  # Minimum confidence to use a gaze point
        
#         # Dynamic calibration
#         self.auto_calibration_points = []
#         self.auto_calibration_data = []
#         self.auto_calibration_threshold = 20
        
#     def _clamp_gaze_point(self, point: np.ndarray) -> np.ndarray:
#         """Clamp gaze point to reasonable values"""
#         if point is None:
#             return None
#         return np.clip(point, -1.0, 2.0)  # Allow slight overshoot but prevent extreme values

#     def add_calibration_point(self, point: Tuple[float, float], left_iris: np.ndarray, right_iris: np.ndarray):
#         """Add a calibration point with both eye data"""
#         self.calibration_points.append(point)
#         self.calibration_data.append({
#             'left': left_iris,
#             'right': right_iris,
#             'combined': (left_iris + right_iris) / 2
#         })
        
#     def add_auto_calibration_point(self, point: Tuple[float, float], left_iris: np.ndarray, right_iris: np.ndarray):
#         """Add points during normal operation to improve calibration"""
#         self.auto_calibration_points.append(point)
#         self.auto_calibration_data.append({
#             'left': left_iris,
#             'right': right_iris,
#             'combined': (left_iris + right_iris) / 2
#         })
        
#         # If we have enough points, try to augment our calibration
#         if len(self.auto_calibration_points) >= self.auto_calibration_threshold:
#             self._augment_calibration()
            
#     def _augment_calibration(self):
#         """Add auto-calibration points to the main calibration data"""
#         if not self.calibrated or len(self.auto_calibration_points) < 5:
#             return
            
#         # Find points that are significantly different from existing calibration
#         for i in range(len(self.auto_calibration_points)):
#             point = self.auto_calibration_points[i]
#             data = self.auto_calibration_data[i]
            
#             # Check distance to nearest calibration point
#             if len(self.calibration_points) > 0:
#                 distances = np.linalg.norm(np.array(self.calibration_points) - np.array(point), axis=1)
#                 min_dist = np.min(distances)
#                 if min_dist > 0.2:  # Only add if significantly different
#                     self.calibration_points.append(point)
#                     self.calibration_data.append(data)
        
#         # Recalibrate
#         self._perform_calibration()
#         self.auto_calibration_points = []
#         self.auto_calibration_data = []

#     def _perform_calibration(self):
#         """Internal calibration using current data"""
#         if len(self.calibration_points) < 5:
#             self.calibrated = False
#             return False
            
#         try:
#             # Create separate triangulations for left, right, and combined
#             left_data = np.array([d['left'] for d in self.calibration_data], dtype=np.float32)
#             right_data = np.array([d['right'] for d in self.calibration_data], dtype=np.float32)
#             combined_data = np.array([d['combined'] for d in self.calibration_data], dtype=np.float32)
            
#             self.left_tri = Delaunay(left_data)
#             self.right_tri = Delaunay(right_data)
#             self.combined_tri = Delaunay(combined_data)
            
#             self.calibration_points_array = np.array(self.calibration_points, dtype=np.float32)
#             self.calibrated = True
#             return True
#         except Exception as e:
#             print(f"Calibration failed: {e}")
#             self.calibrated = False
#             return False
            
#     def calibrate(self):
#         """Perform initial calibration"""
#         return self._perform_calibration()
    
#     def _estimate_for_eye(self, iris_center: np.ndarray, tri: Delaunay) -> Tuple[np.ndarray, float]:
#         """Estimate gaze point for a single eye"""
#         simplex = tri.find_simplex(iris_center)
        
#         if simplex == -1:
#             # Outside convex hull - use extrapolation
#             distances = np.linalg.norm(tri.points - iris_center, axis=1)
#             nearest_idx = np.argmin(distances)
#             estimated_point = self.calibration_points_array[nearest_idx]
            
#             # Confidence based on distance
#             min_distance = np.min(distances)
#             confidence = max(0.1, 1.0 - (min_distance / 0.15))  # 0.15 is max reasonable distance
#         else:
#             # Inside convex hull - use barycentric interpolation
#             transform = tri.transform[simplex, :2]
#             r = iris_center - tri.transform[simplex, 2]
#             bary = np.dot(r, transform)
#             bary_coords = np.append(bary, 1.0 - bary.sum())
#             vertices = tri.simplices[simplex]
#             estimated_point = np.dot(bary_coords, self.calibration_points_array[vertices])
            
#             # Confidence based on distance to vertices
#             distances = [np.linalg.norm(iris_center - tri.points[v]) for v in vertices]
#             confidence = 1.0 - (min(distances) / 0.1)
#             confidence = np.clip(confidence, 0.1, 1.0)
        
#         return estimated_point, confidence
        
#     def _detect_blink(self, eye_landmarks: np.ndarray) -> bool:
#         """Detect if eye is closed/blinking"""
#         if len(eye_landmarks) < 6:
#             return False
            
#         # Calculate eye aspect ratio (EAR)
#         # Vertical distances
#         v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
#         v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
#         # Horizontal distance
#         h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
#         ear = (v1 + v2) / (2.0 * h)
#         return ear < 0.2  # Threshold for blink detection
    
#     def estimate_gaze(self, left_iris: np.ndarray, right_iris: np.ndarray,
#                     left_eye_landmarks: Optional[np.ndarray] = None,
#                     right_eye_landmarks: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
#         """
#         Estimate gaze point with improved stability checks
#         Returns clamped (0-1) coordinates or None if unreliable
#         """
#         if not self.calibrated:
#             return None

#         # Blink detection
#         left_blink = False
#         right_blink = False
#         if left_eye_landmarks is not None and len(left_eye_landmarks) >= 6:
#             v1 = np.linalg.norm(left_eye_landmarks[1] - left_eye_landmarks[5])
#             v2 = np.linalg.norm(left_eye_landmarks[2] - left_eye_landmarks[4])
#             h = np.linalg.norm(left_eye_landmarks[0] - left_eye_landmarks[3])
#             left_ear = (v1 + v2) / (2.0 * h)
#             left_blink = left_ear < 0.2

#         if right_eye_landmarks is not None and len(right_eye_landmarks) >= 6:
#             v1 = np.linalg.norm(right_eye_landmarks[1] - right_eye_landmarks[5])
#             v2 = np.linalg.norm(right_eye_landmarks[2] - right_eye_landmarks[4])
#             h = np.linalg.norm(right_eye_landmarks[0] - right_eye_landmarks[3])
#             right_ear = (v1 + v2) / (2.0 * h)
#             right_blink = right_ear < 0.2

#         # Handle blinking
#         if left_blink and right_blink:
#             self.blink_counter += 1
#             if self.blink_counter > 3:
#                 return None
#             return self._clamp_gaze_point(self.last_gaze)
#         else:
#             self.blink_counter = 0

#         # Estimate for each eye with confidence
#         left_point, left_conf = self._estimate_for_eye(left_iris, self.left_tri)
#         right_point, right_conf = self._estimate_for_eye(right_iris, self.right_tri)

#         # Adjust confidence for blinking
#         if left_blink:
#             left_conf *= 0.3
#         if right_blink:
#             right_conf *= 0.3

#         self.left_confidence = left_conf
#         self.right_confidence = right_conf

#         # Only proceed if we have sufficient confidence
#         if max(left_conf, right_conf) < self.min_confidence:
#             return self._clamp_gaze_point(self.last_gaze)

#         # Weighted average
#         total_conf = left_conf + right_conf
#         if total_conf > 0:
#             gaze_point = (left_point * left_conf + right_point * right_conf) / total_conf
#         else:
#             gaze_point = (left_point + right_point) / 2

#         # Apply Kalman filtering with checks
#         try:
#             gaze_point = self.left_kf.update(gaze_point)
#             gaze_point = self.right_kf.update(gaze_point)
#         except:
#             gaze_point = self._clamp_gaze_point(self.last_gaze)

#         # Velocity-based prediction with damping
#         if self.last_gaze is not None:
#             current_velocity = gaze_point - self.last_gaze
#             smoothed_velocity = 0.2 * current_velocity + 0.8 * self.last_velocity
#             gaze_point += smoothed_velocity * 0.3  # More conservative prediction
#             self.last_velocity = smoothed_velocity

#         # Store in history
#         self.gaze_history.append((gaze_point.copy(), total_conf / 2))

#         # Weighted average of history
#         if len(self.gaze_history) > 0:
#             points, confs = zip(*self.gaze_history)
#             weights = np.array(confs)
#             weights_sum = weights.sum()
#             if weights_sum > 0:
#                 weights = weights / weights_sum
#             else:
#                 weights = np.ones(len(confs)) / len(confs)
#             gaze_point = np.average(points, axis=0, weights=weights)

#         # Dead zone for small movements
#         if self.last_gaze is not None:
#             distance = np.linalg.norm(gaze_point - self.last_gaze)
#             if distance < self.movement_threshold:
#                 gaze_point = self.last_gaze

#         self.last_gaze = gaze_point.copy()
#         return self._clamp_gaze_point(gaze_point)

#     def get_screen_position(self, point: np.ndarray) -> Tuple[int, int]:
#         """Convert normalized point to screen coordinates with bounds checking"""
#         if point is None:
#             return None
#         # Clamp to 0-1 range before conversion
#         clamped = np.clip(point, 0.0, 1.0)
#         return (
#             int(clamped[0] * self.screen_size[0]),
#             int(clamped[1] * self.screen_size[1])
#         )
    
#     def get_screen_quadrant(self, point: np.ndarray) -> str:
#         """Get which screen quadrant the point is in"""
#         if point is None:
#             return None
            
#         x, y = point
#         w, h = self.screen_size
#         center_threshold = 0.2  # 20% of screen is considered center
        
#         # Check if in center
#         if (w*(0.5-center_threshold) < x < w*(0.5+center_threshold)) and \
#            (h*(0.5-center_threshold) < y < h*(0.5+center_threshold)):
#             return "CENTER"
            
#         # Determine quadrant with finer granularity
#         if x < w/3:
#             if y < h/3: return "TOP-LEFT"
#             elif y < 2*h/3: return "LEFT"
#             else: return "BOTTOM-LEFT"
#         elif x < 2*w/3:
#             if y < h/3: return "TOP"
#             elif y < 2*h/3: return "CENTER"
#             else: return "BOTTOM"
#         else:
#             if y < h/3: return "TOP-RIGHT"
#             elif y < 2*h/3: return "RIGHT"
#             else: return "BOTTOM-RIGHT"

# def main():
#     cap = cv2.VideoCapture(0)
#     screen_width, screen_height = 1920, 1080
#     tracker = EnhancedGazeTracker(screen_size=(screen_width, screen_height))
    
#     calibration_step = 0
#     calibration_positions = [
#         (0.5, 0.5), (0.1, 0.1), (0.9, 0.1), 
#         (0.1, 0.9), (0.9, 0.9), (0.1, 0.5),
#         (0.9, 0.5), (0.5, 0.1), (0.5, 0.9)
#     ]

#     last_print_time = time.time()
#     print_interval = 1.0  # seconds

#     with mp_face_mesh.FaceMesh(
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.6,
#         min_tracking_confidence=0.6
#     ) as face_mesh:
        
#         while cap.isOpened():
#             success, image = cap.read()
#             if not success:
#                 continue

#             image = cv2.flip(image, 1)
#             h, w, _ = image.shape
#             rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(rgb_frame)
            
#             if results.multi_face_landmarks:
#                 mesh_points = results.multi_face_landmarks[0].landmark
                
#                 left_iris = np.array([(mesh_points[p].x, mesh_points[p].y) for p in LEFT_IRIS])
#                 right_iris = np.array([(mesh_points[p].x, mesh_points[p].y) for p in RIGHT_IRIS])
#                 left_center = left_iris.mean(axis=0)
#                 right_center = right_iris.mean(axis=0)
                
#                 left_eye = np.array([(mesh_points[p].x, mesh_points[p].y) for p in LEFT_EYE])
#                 right_eye = np.array([(mesh_points[p].x, mesh_points[p].y) for p in RIGHT_EYE])
                
#                 if calibration_step < len(calibration_positions):
#                     target_pos = calibration_positions[calibration_step]
#                     target_pixel = tracker.get_screen_position(target_pos)
                    
#                     cv2.putText(image, "Look at the circle and press SPACE", (50, 50), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                     cv2.circle(image, target_pixel, 20, (0, 0, 255), 2)
#                     cv2.putText(image, f"Calibration step {calibration_step+1}/{len(calibration_positions)}", 
#                                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
#                     key = cv2.waitKey(1)
#                     if key == 32:  # SPACE
#                         tracker.add_calibration_point(target_pos, left_center, right_center)
#                         calibration_step += 1
#                         if calibration_step == len(calibration_positions):
#                             if not tracker.calibrate():
#                                 print("Calibration failed! Please try again.")
#                                 calibration_step = 0
#                                 tracker = EnhancedGazeTracker(screen_size=(screen_width, screen_height))
#                 else:
#                     gaze_point = tracker.estimate_gaze(left_center, right_center, left_eye, right_eye)
                    
#                     if gaze_point is not None:
#                         gaze_pixel = tracker.get_screen_position(gaze_point)
#                         quadrant = tracker.get_screen_quadrant(gaze_pixel)
                        
#                         # Only draw if coordinates are valid
#                         if 0 <= gaze_pixel[0] <= w and 0 <= gaze_pixel[1] <= h:
#                             cv2.circle(image, gaze_pixel, 10, (0, 255, 255), -1)
                        
#                         # Print coordinates every second
#                         current_time = time.time()
#                         if current_time - last_print_time >= print_interval:
#                             print(f"Gaze: X={gaze_point[0]:.3f}, Y={gaze_point[1]:.3f} (Quadrant: {quadrant})")
#                             last_print_time = current_time
            
#             cv2.imshow('Eye Tracking', image)
#             if cv2.waitKey(1) == 27:
#                 break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Landmark indices
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Gaze parameters
GAZE_HISTORY_LENGTH = 5
SENSITIVITY = 50  # Increased sensitivity
gaze_history = deque(maxlen=GAZE_HISTORY_LENGTH)

# Screen regions
REGIONS = {
    "top-left": (0, 0, 0.4, 0.4),
    "top-right": (0.6, 0, 1.0, 0.4),
    "bottom-left": (0, 0.6, 0.4, 1.0),
    "bottom-right": (0.6, 0.6, 1.0, 1.0),
    "center": (0.4, 0.4, 0.6, 0.6)
}

cap = cv2.VideoCapture(0)

def get_region(x, y):
    for region, (x1, y1, x2, y2) in REGIONS.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return region
    return "unknown"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    image = cv2.flip(image, 1)
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    h, w = image.shape[:2]
    
    # Draw region rectangles
    for region, (x1, y1, x2, y2) in REGIONS.items():
        color = (100, 100, 100)
        cv2.rectangle(image, (int(x1 * w), int(y1 * h)), 
                     (int(x2 * w), int(y2 * h)), color, 2)
    
    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark
        
        # Get eye contours and iris positions
        left_eye_points = np.array([(mesh_points[p].x, mesh_points[p].y) for p in LEFT_EYE])
        right_eye_points = np.array([(mesh_points[p].x, mesh_points[p].y) for p in RIGHT_EYE])
        left_iris = np.array([(mesh_points[p].x, mesh_points[p].y) for p in LEFT_IRIS])
        right_iris = np.array([(mesh_points[p].x, mesh_points[p].y) for p in RIGHT_IRIS])
        left_center = left_iris.mean(axis=0)
        right_center = right_iris.mean(axis=0)
        
        # Calculate eye aspect ratio
        def eye_aspect_ratio(eye_points):
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            return (A + B) / (2.0 * C)
        
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        
        if left_ear > 0.2 and right_ear > 0.2:  # Eyes open
            # Calculate gaze direction with enhanced sensitivity
            def get_gaze(eye_points, iris_center):
                eye_center = eye_points.mean(axis=0)
                
                # Enhanced sensitivity calculation
                gaze_x = (iris_center[0] - eye_center[0]) * SENSITIVITY 
                gaze_y = (iris_center[1] - eye_center[1]) * SENSITIVITY
                
                # Map to screen coordinates
                screen_x = 0.5 + gaze_x
                screen_y = 0.5 + gaze_y
                
                return np.clip(screen_x, 0, 1), np.clip(screen_y, 0, 1)
            
            left_gaze = get_gaze(left_eye_points, left_center)
            right_gaze = get_gaze(right_eye_points, right_center)
            gaze = ((left_gaze[0] + right_gaze[0])/2, (left_gaze[1] + right_gaze[1])/2)
            
            gaze_history.append(gaze)
            smooth_gaze = np.mean(gaze_history, axis=0)
            
            # Get region and display
            current_region = get_region(*smooth_gaze)
            gaze_pixel = (int(smooth_gaze[0] * w), int(smooth_gaze[1] * h))
            
            cv2.circle(image, gaze_pixel, 10, (0, 255, 0), -1)
            cv2.putText(image, f"Region: {current_region}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Highlight current region
            if current_region != "unknown":
                x1, y1, x2, y2 = REGIONS[current_region]
                cv2.rectangle(image, (int(x1 * w), int(y1 * h)),
                            (int(x2 * w), int(y2 * h)), (0, 255, 0), 3)
    
    cv2.imshow('Improved Gaze Tracking', image)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()