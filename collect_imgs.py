import os
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import json
from typing import Dict
import logging

class GestureDataCollector:
    def __init__(self, data_dir: str = './data'):
        """
        Initialize the gesture data collector.
        
        Args:
            data_dir: Directory to store collected images
        """
        self.DATA_DIR = data_dir
        self.setup_logging()
        self.setup_mediapipe()
        self.setup_gestures()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_mediapipe(self):
        """Initialize MediaPipe hands detector."""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
    def setup_gestures(self):
        """Define gesture classes and their descriptions."""
        self.gestures = {
            0: "Dhanyabadh (Prayer Pose) - One Hand",
            1: "Thank You (Both Hands)",
            2: "Thumbs Up (Both Hands)",
            3: "Peace Sign (Both Hands)"
        }
        
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)
            
        metadata = {
            "gestures": self.gestures,
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "requirements": "Single or two hands depending on gesture"
        }
        
        with open(os.path.join(self.DATA_DIR, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
            
    def check_hands_present(self, frame, required_hands=2) -> bool:
        """
        Check if the required number of hands are present in the frame.
        
        Args:
            frame: The video frame to process.
            required_hands: Number of hands required (1 or 2).
        
        Returns:
            bool: True if the required number of hands are detected.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            return len(results.multi_hand_landmarks) >= required_hands
        return False
        
    def draw_hands(self, frame):
        """Draw hand landmarks on the frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return frame
        
    def collect_data(self, dataset_size: int = 100, camera_index: int = 0):
        """
        Collect gesture images for the dataset.
        
        Args:
            dataset_size: Number of images to collect per gesture.
            camera_index: Index of the camera to use.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            self.logger.error(f"Failed to open camera {camera_index}")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        try:
            for gesture_id, gesture_desc in self.gestures.items():
                gesture_dir = os.path.join(self.DATA_DIR, str(gesture_id))
                if not os.path.exists(gesture_dir):
                    os.makedirs(gesture_dir)
                    
                self.logger.info(f'Collecting data for gesture: {gesture_desc}')
                
                required_hands = 1 if "One Hand" in gesture_desc else 2
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                        
                    frame = self.draw_hands(frame)
                    
                    cv2.putText(frame, f'Prepare for: {gesture_desc}', 
                                (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, 'Press "Q" when ready!', 
                                (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                    
                    hands_detected = self.check_hands_present(frame, required_hands)
                    status = f"✓ {required_hands} hand(s) detected" if hands_detected else f"✗ Need {required_hands} hand(s)"
                    color = (0, 255, 0) if hands_detected else (0, 0, 255)
                    cv2.putText(frame, status, (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                    
                    cv2.imshow('Gesture Collection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                counter = 0
                while counter < dataset_size:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                        
                    frame = self.draw_hands(frame)
                    
                    if self.check_hands_present(frame, required_hands):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f'{counter}_{timestamp}.jpg'
                        cv2.imwrite(os.path.join(gesture_dir, filename), frame)
                        
                        cv2.putText(frame, f'Capturing: {counter}/{dataset_size}', 
                                    (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                        counter += 1
                    else:
                        cv2.putText(frame, f'Please show {required_hands} hand(s)!', 
                                    (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                    
                    cv2.imshow('Gesture Collection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                self.logger.info(f'Completed collecting {counter} images for {gesture_desc}')
                
        except Exception as e:
            self.logger.error(f"Error during data collection: {e}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = GestureDataCollector()
    collector.collect_data(dataset_size=100)
