import os
import sys
import pickle
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# Configuration
MODEL_PATH = './model_20250209_181811.p'
CAMERA_INDEX = 0 
MIN_DETECTION_CONFIDENCE = 0.3
MAX_IMAGE_SIZE = 1280
ESC_KEY = 27

# Reference images for instructions
INSTRUCTION_IMAGE_1 = './reference_images/image1.png'
INSTRUCTION_IMAGE_2 = './reference_images/image2.png'

# Set fixed display dimensions
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
# Make webcam frame wider (75% of display width)
WEBCAM_WIDTH = int(DISPLAY_WIDTH * 0.65)
REFERENCE_WIDTH = DISPLAY_WIDTH - WEBCAM_WIDTH
REFERENCE_HEIGHT = int(DISPLAY_HEIGHT)  # Reduce reference image height

# Dictionary mapping numeric labels to sign meanings
LABELS_DICT = {0: 'Ka', 1: 'Kha', 2: 'Ga', 3:"Gha", 4: 'nga', 5: 'च, ca', 
               6: 'chha', 7: 'ja', 8: 'jha', 9: 'nya', 10: 'Zero / Sunya', 
               11: 'One / Yek', 12: 'two / dui', 13:'three / teen', 14: 'four / char', 15: 'five / pach', 
               16: 'six / cha', 17: 'seven / sath', 18: 'eight / aath', 19: 'nine / nau', 20: 'I love you',
               21: 'Maaf garnu', 22: 'hudaina', 23: 'hunxa', 24: "sathi", 25: "bachha", 
               26: "Sahayog", 27: "kripaya", 28: "pariwar"}

def resize_if_needed(image: np.ndarray) -> np.ndarray:
    """Resize image if it exceeds maximum dimensions while maintaining aspect ratio."""
    height, width = image.shape[:2]
    max_dim = max(height, width)
    
    if max_dim > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def load_model(model_path: str):
    """Load the trained model from a pickle file with metadata validation."""
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist.")
        sys.exit(1)
    try:
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
        model = model_dict.get('model', None)
        metadata = model_dict.get('metadata', {})
        
        if model is None:
            raise ValueError("No model found in the pickle file.")
            
        if metadata:
            print("\nModel Information:")
            for key, value in metadata.items():
                print(f"- {key}: {value}")
                
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def initialize_mediapipe():
    """Initialize MediaPipe hands with consistent configuration."""
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        max_num_hands=2
    )
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    return hands_detector, drawing_utils, drawing_styles, mp_hands

def extract_features_from_hand(hand_landmarks):
    """Extract scale-invariant features from hand landmarks."""
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    
    min_x, min_y = min(x_coords), min(y_coords)
    max_x, max_y = max(x_coords), max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    scale = max(width, height) if max(width, height) > 0 else 1.0
    
    features = []
    for x, y in zip(x_coords, y_coords):
        features.append((x - min_x) / scale)
        features.append((y - min_y) / scale)
    return features

def draw_label_with_background(image, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                             font_scale=1.2, font_thickness=2,
                             text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with background for better visibility."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = pos
    
    padding = 10
    cv2.rectangle(image, 
                 (x - padding, y - text_height - padding), 
                 (x + text_width + padding, y + padding),
                 bg_color, 
                 -1)
    
    cv2.putText(image, text, (x, y), font, font_scale, text_color, font_thickness)

def run_inference(model):
    """Run real-time inference with modified display layout."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    
    hands_detector, drawing_utils, drawing_styles, mp_hands = initialize_mediapipe()
    
    print("\nStarting inference...")
    print("Controls:")
    print("- Press 'q' or 'ESC' to exit")
    print("- Press SPACE to switch between instruction images")
    
    frame_count = 0
    fps_start_time = datetime.now()
    showing_first_image = True
    
    cv2.namedWindow('Sign Script', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sign Script', DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Could not retrieve frame. Exiting.")
                break
            
            # Resize frame to webcam width
            frame = cv2.resize(frame, (WEBCAM_WIDTH, DISPLAY_HEIGHT))
            
            # Load and resize instruction image
            current_image_path = INSTRUCTION_IMAGE_1 if showing_first_image else INSTRUCTION_IMAGE_2
            if os.path.exists(current_image_path):
                instruction_img = cv2.imread(current_image_path)
                if instruction_img is not None:
                    # Resize instruction image to smaller dimensions
                    instruction_img = cv2.resize(instruction_img, (REFERENCE_WIDTH, REFERENCE_HEIGHT))
                    # Create a black background for the instruction image
                    instruction_background = np.zeros((DISPLAY_HEIGHT, REFERENCE_WIDTH, 3), dtype=np.uint8)
                    # Calculate vertical position to center the instruction image
                    y_offset = (DISPLAY_HEIGHT - REFERENCE_HEIGHT) // 2
                    instruction_background[y_offset:y_offset + REFERENCE_HEIGHT, :] = instruction_img
                    instruction_img = instruction_background
            else:
                instruction_img = np.zeros((DISPLAY_HEIGHT, REFERENCE_WIDTH, 3), dtype=np.uint8)
            
            # Process frame for hand detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(frame_rgb)
            
            features = []
            bbox = None
            
            if results.multi_hand_landmarks:
                hands_list = results.multi_hand_landmarks
                
                for hand_landmarks in hands_list:
                    drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style()
                    )
                
                try:
                    if len(hands_list) >= 2:
                        sorted_hands = sorted(hands_list, key=lambda hand: hand.landmark[0].x)
                        feat1 = extract_features_from_hand(sorted_hands[0])
                        feat2 = extract_features_from_hand(sorted_hands[1])
                        features = feat1 + feat2
                        
                        combined_landmarks = list(sorted_hands[0].landmark) + list(sorted_hands[1].landmark)
                        x1 = max(0, int(min([lm.x for lm in combined_landmarks]) * WEBCAM_WIDTH) - 10)
                        y1 = max(0, int(min([lm.y for lm in combined_landmarks]) * DISPLAY_HEIGHT) - 10)
                        x2 = min(WEBCAM_WIDTH, int(max([lm.x for lm in combined_landmarks]) * WEBCAM_WIDTH) + 10)
                        y2 = min(DISPLAY_HEIGHT, int(max([lm.y for lm in combined_landmarks]) * DISPLAY_HEIGHT) + 10)
                        bbox = (x1, y1, x2, y2)
                    
                    elif len(hands_list) == 1:
                        feat1 = extract_features_from_hand(hands_list[0])
                        features = feat1 + ([0] * 42)
                        
                        x1 = max(0, int(min([lm.x for lm in hands_list[0].landmark]) * WEBCAM_WIDTH) - 10)
                        y1 = max(0, int(min([lm.y for lm in hands_list[0].landmark]) * DISPLAY_HEIGHT) - 10)
                        x2 = min(WEBCAM_WIDTH, int(max([lm.x for lm in hands_list[0].landmark]) * WEBCAM_WIDTH) + 10)
                        y2 = min(DISPLAY_HEIGHT, int(max([lm.y for lm in hands_list[0].landmark]) * DISPLAY_HEIGHT) + 10)
                        bbox = (x1, y1, x2, y2)
                    
                except Exception as e:
                    print(f"Error processing hand landmarks: {e}")
                    features = []
            
            else:
                draw_label_with_background(frame, "No hands detected", (30, 60))
            
            if len(features) == 84:
                try:
                    prediction = model.predict([np.asarray(features)])
                    predicted_numeric = int(prediction[0])
                    predicted_label = LABELS_DICT.get(predicted_numeric, str(predicted_numeric))
                    
                    if bbox:
                        draw_label_with_background(frame, predicted_label, (bbox[0], bbox[1] - 30))
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    else:
                        draw_label_with_background(frame, predicted_label, (30, 60))
                        
                except Exception as e:
                    draw_label_with_background(frame, "Prediction error", (30, 60))
                    print(f"Prediction error: {e}")
            
            # Create combined display
            combined_frame = np.hstack((frame, instruction_img))
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = (datetime.now() - fps_start_time).total_seconds()
                fps = frame_count / elapsed_time
                draw_label_with_background(combined_frame, f"FPS: {fps:.1f}", (30, DISPLAY_HEIGHT - 30))
            
            cv2.imshow('Sign Script', combined_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ESC_KEY or key in [ord('q'), ord('Q')]:
                print("Exit key pressed. Exiting.")
                break
            elif key == ord(' '):
                showing_first_image = not showing_first_image
                print(f"Switched to {'first' if showing_first_image else 'second'} instruction image")
                
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)
    run_inference(model)

if __name__ == '__main__':
    main()
