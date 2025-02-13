import pickle
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HandGestureClassifier:
    def __init__(self, model_path: str, min_detection_confidence: float = 0.3):
        """
        Initialize the hand gesture classifier.
        
        Args:
            model_path: Path to the pickle file containing the trained model
            min_detection_confidence: Minimum confidence value for hand detection
        """
        self.load_model(model_path)
        self.setup_mediapipe(min_detection_confidence)
        # Updated labels dictionary
        self.labels_dict: Dict[int, str] = {
            0: "Dhanyabadh (Prayer Pose) - One Hand",
            1: "Thank You (Both Hands)",
            2: "Thumbs Up (Both Hands)",
            3: "Peace Sign (Both Hands)"
        }
        self.fps_history: List[float] = []

    def load_model(self, model_path: str) -> None:
        """Load the trained model from pickle file."""
        try:
            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
                self.model = model_dict['model']
            logger.info("Model loaded successfully")
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error loading model: {e}")
            raise

    def setup_mediapipe(self, min_detection_confidence: float) -> None:
        """Initialize MediaPipe components."""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            max_num_hands=2  # Support detecting up to 2 hands
        )

    def preprocess_landmarks(self, hand_landmarks) -> tuple:
        """Process hand landmarks and return normalized coordinates."""
        x_ = []
        y_ = []
        data_aux = []

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        # Normalize coordinates
        min_x, min_y = min(x_), min(y_)
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min_x)
            data_aux.append(y - min_y)

        return x_, y_, data_aux

    def calculate_fps(self, start_time: float) -> float:
        """Calculate and smooth FPS."""
        fps = 1.0 / (time.time() - start_time)
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:  # Keep last 30 frames for averaging
            self.fps_history.pop(0)
        return sum(self.fps_history) / len(self.fps_history)

    def run(self, camera_index: int = 0) -> None:
        """Run the hand gesture classifier on video input."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_index}")
            return

        try:
            while True:
                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                # Process frame
                H, W, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                # Draw FPS
                fps = self.calculate_fps(start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

                        # Process landmarks
                        x_, y_, data_aux = self.preprocess_landmarks(hand_landmarks)

                        # Calculate bounding box
                        x_min = min([lm.x for lm in hand_landmarks.landmark]) * W
                        y_min = min([lm.y for lm in hand_landmarks.landmark]) * H
                        x_max = max([lm.x for lm in hand_landmarks.landmark]) * W
                        y_max = max([lm.y for lm in hand_landmarks.landmark]) * H

                        try:
                            # Make prediction
                            prediction = self.model.predict([np.asarray(data_aux)])
                            prediction_proba = self.model.predict_proba([np.asarray(data_aux)])
                            confidence = max(prediction_proba[0]) * 100

                            predicted_character = self.labels_dict[int(prediction[0])]

                            # Draw prediction and confidence
                            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                            cv2.putText(frame, f"{predicted_character} ({confidence:.1f}%)",
                                        (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        except Exception as e:
                            logger.error(f"Prediction error: {e}")

                cv2.imshow('Hand Gesture Classifier', frame)

                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logger.error(f"Runtime error: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        classifier = HandGestureClassifier(
            model_path='./models/model.p',
            min_detection_confidence=0.3
        )
        classifier.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
