import cv2
import mediapipe as mp
import time
import logging
import threading
from collections import deque
import numpy as np
from adafruit_servokit import ServoKit

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def open_camera(max_attempts=5):
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            logging.info(f"Camera opened successfully on attempt {attempt + 1}")
            return cap
        else:
            logging.warning(f"Failed to open camera on attempt {attempt + 1}")
            cap.release()
            time.sleep(1)  # Wait a second before trying again
    
    logging.error(f"Failed to open camera after {max_attempts} attempts")
    return None

class ServoController:
    def __init__(self):
        self.kit = ServoKit(channels=16)  # Initialize ServoKit for 16-channel servo hat
        self.current_pos = 90.0
        self.target_pos = 90.0
        self.velocity = 0.0
        self.last_update_time = time.time()
        self.position_history = deque(maxlen=10)  # Store last 10 positions for moving average
        self.dead_zone = 1.0  # Dead zone in degrees
        self.update_interval = 0.05  # 50ms between updates
        
    def update(self):
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        if dt < self.update_interval:
            return self.current_pos
        
        self.last_update_time = current_time
        
        # Calculate moving average of target position
        self.position_history.append(self.target_pos)
        avg_target = np.mean(self.position_history)
        
        # Simple proportional control with dead zone
        error = avg_target - self.current_pos
        if abs(error) > self.dead_zone:
            self.velocity = error * 1.0  # Reduced multiplier for smoother movement
            
            # Update position
            self.current_pos += self.velocity * dt
            self.current_pos = max(0, min(180, self.current_pos))
            
            self.set_servo_position(self.current_pos)
        
        return self.current_pos
        
    def set_servo_position(self, position):
        try:
            self.kit.servo[0].angle = position
            logging.debug(f"Set servo to position: {position}")
        except Exception as e:
            logging.error(f"Failed to set servo position: {e}")

class PoseDetector:
    def __init__(self):
        self.cap = open_camera()
        if self.cap is None:
            raise Exception("Failed to open camera")
        
        self.servo_controller = ServoController()
        self.is_running = True

    def servo_control_thread(self):
        while self.is_running:
            current_pos = self.servo_controller.update()
            logging.debug(f"Current servo position: {current_pos:.2f}")
            time.sleep(0.01)  # Small delay for stability

    def start(self):
        # Start servo control in a separate thread
        servo_thread = threading.Thread(target=self.servo_control_thread)
        servo_thread.start()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Failed to capture frame")
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                    nose_x = nose.x

                    # Convert nose_x from [0, 1] to [0, 180] for servo angle
                    new_position = (1 - nose_x) * 180
                    self.servo_controller.target_pos = new_position

                    logging.info(f"Nose position: {nose_x:.2f}, Target: {new_position:.2f}")
                else:
                    logging.info("No face detected")

        finally:
            self.is_running = False
            servo_thread.join()
            self.cap.release()
            logging.info("Pose detection terminated")

if __name__ == "__main__":
    detector = PoseDetector()
    detector.start()
