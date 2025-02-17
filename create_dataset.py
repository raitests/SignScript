import os
import sys
import pickle
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Dict, Optional
from datetime import datetime

DATA_DIR = './data'
OUTPUT_PICKLE = 'data.pickle'
MIN_DETECTION_CONFIDENCE = 0.3
MAX_IMAGE_SIZE = 1280  # Maximum dimension for processing

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    max_num_hands=2  # Explicitly set maximum number of hands
)

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

def process_image(image_path: str) -> List[float]:
    """Process image and extract hand landmarks with enhanced error handling."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image '{image_path}'. Skipping.")
            return []
        
        # Resize large images to prevent processing issues
        img = resize_if_needed(img)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            print(f"Info: No hand landmarks detected in '{image_path}'. Skipping.")
            return []
        
        hands_list = results.multi_hand_landmarks
        features = []
        
        def extract_features(hand_landmarks) -> List[float]:
            """Extract normalized and scale-invariant features from hand landmarks."""
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            # Normalize coordinates
            min_x, min_y = min(x_coords), min(y_coords)
            max_x, max_y = max(x_coords), max(y_coords)
            
            # Prevent division by zero
            width = max_x - min_x
            height = max_y - min_y
            scale = max(width, height) if max(width, height) > 0 else 1.0
            
            feat = []
            for x, y in zip(x_coords, y_coords):
                # Normalize position and scale
                feat.append((x - min_x) / scale)
                feat.append((y - min_y) / scale)
            return feat
        
        if len(hands_list) >= 2:
            # Sort hands by x-coordinate to maintain consistent ordering
            sorted_hands = sorted(hands_list, key=lambda hand: hand.landmark[0].x)
            features += extract_features(sorted_hands[0])
            features += extract_features(sorted_hands[1])
        elif len(hands_list) == 1:
            features += extract_features(hands_list[0])
            features += [0] * 42  # Padding for single hand
        
        if len(features) != 84:
            print(f"Warning: Feature vector length {len(features)} != 84 for image '{image_path}'. Skipping.")
            return []
        
        return features
    
    except Exception as e:
        print(f"Error processing image '{image_path}': {str(e)}")
        return []

def create_dataset(data_dir: str) -> Tuple[List[List[float]], List[str]]:
    """Create dataset with progress tracking and validation."""
    dataset = []
    labels = []
    
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        sys.exit(1)
    
    class_folders = sorted(os.listdir(data_dir))
    total_classes = len(class_folders)
    
    for class_idx, class_label in enumerate(class_folders, 1):
        class_dir = os.path.join(data_dir, class_label)
        if not os.path.isdir(class_dir):
            continue
        
        print(f"\nProcessing class {class_idx}/{total_classes}: '{class_label}'")
        image_files = sorted(os.listdir(class_dir))
        total_images = len(image_files)
        successful = 0
        
        for idx, img_filename in enumerate(image_files, 1):
            img_path = os.path.join(class_dir, img_filename)
            feat_vector = process_image(img_path)
            
            if feat_vector:
                dataset.append(feat_vector)
                labels.append(class_label)
                successful += 1
            
            # Progress update
            if idx % 100 == 0 or idx == total_images:
                success_rate = (successful / idx) * 100
                print(f"Progress: {idx}/{total_images} images processed "
                      f"({success_rate:.1f}% success rate)")
    
    return dataset, labels

def save_dataset(dataset: List[List[float]], labels: List[str], output_path: str):
    """Save dataset with timestamp and metadata."""
    if not dataset:
        print("Error: No data to save.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{os.path.splitext(output_path)[0]}_{timestamp}.pickle"
    
    metadata = {
        'creation_date': timestamp,
        'num_samples': len(dataset),
        'num_classes': len(set(labels)),
        'feature_dimension': len(dataset[0]),
        'min_detection_confidence': MIN_DETECTION_CONFIDENCE,
        'max_image_size': MAX_IMAGE_SIZE
    }
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump({
                'data': dataset,
                'labels': labels,
                'metadata': metadata
            }, f)
        print(f"\nDataset saved successfully to '{output_path}'")
        print("\nDataset statistics:")
        for key, value in metadata.items():
            print(f"- {key}: {value}")
    except Exception as e:
        print(f"Error saving dataset: {e}")

def main():
    print("Starting dataset creation...")
    print(f"Using MediaPipe Hands with detection confidence: {MIN_DETECTION_CONFIDENCE}")
    
    data, labels = create_dataset(DATA_DIR)
    
    if not data:
        print("No valid data was processed. Exiting.")
        sys.exit(1)
    
    save_dataset(data, labels, OUTPUT_PICKLE)
    print("Dataset creation completed.")

if __name__ == '__main__':
    main()
