import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

def create_dataset():
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    DATA_DIR = './data'
    
    data = []
    labels = []
    
    # Print current working directory and data directory contents
    print(f"Current working directory: {os.getcwd()}")
    if os.path.exists(DATA_DIR):
        print(f"Contents of data directory: {os.listdir(DATA_DIR)}")
    else:
        print("Data directory not found. Please ensure the data directory exists.")
        return

    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        
        # Skip if not a directory
        if not os.path.isdir(dir_path):
            print(f"Skipping non-directory: {dir_}")
            continue
            
        print(f"Processing directory: {dir_}")
        print(f"Contents of {dir_}: {os.listdir(dir_path)}")

        for img_path in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            img_full_path = os.path.join(dir_path, img_path)
            img = cv2.imread(img_full_path)
            if img is None:
                print(f"Failed to read image: {img_full_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)

                    min_x = min(x_)
                    min_y = min(y_)

                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min_x)
                        data_aux.append(landmark.y - min_y)

                data.append(data_aux)
                labels.append(dir_)
                print(f"Processed image: {img_path}, data length: {len(data_aux)}")
            else:
                print(f"No hands detected in image: {img_path}")

    print(f"Total samples collected: {len(data)}")
    print(f"Total labels collected: {len(labels)}")

    if data and labels:
        try:
            with open('data.pickle', 'wb') as f:
                pickle.dump({'data': data, 'labels': labels}, f)
            print("Dataset saved successfully!")
            
            # Verify the file was created
            if os.path.exists('data.pickle'):
                print(f"Pickle file created successfully. Size: {os.path.getsize('data.pickle')} bytes")
            else:
                print("Warning: Pickle file not found after saving!")
        except Exception as e:
            print(f"Error saving pickle file: {e}")
    else:
        print("No data collected! Check if hands were detected in images.")

if __name__ == "__main__":
    create_dataset()
