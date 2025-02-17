import os
import cv2
import sys
import time
from datetime import datetime

# Configuration constants
DATA_DIR = './data'
NUMBER_OF_CLASSES = 29
DATASET_SIZE = 1500  # Updated to 1000 images per class
WINDOW_NAME = "Sign Script Data Collection"
COUNTDOWN_TIME = 3  # Seconds to wait before starting capture
CAPTURE_INTERVAL = 0.1  # Time between captures in seconds

def create_directory(path: str):
    """Create a directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"Error: Unable to create directory {path}: {e}")
        sys.exit(1)

def initialize_data_directories():
    """Initialize directories for all classes."""
    create_directory(DATA_DIR)
    for class_id in range(NUMBER_OF_CLASSES):
        class_dir = os.path.join(DATA_DIR, str(class_id))
        create_directory(class_dir)

def show_countdown(frame, seconds: int):
    """Display countdown on frame."""
    height, width = frame.shape[:2]
    font_scale = min(width, height) / 500  # Adaptive font size
    
    # Add semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Add countdown text
    text = str(seconds)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale * 3, 3)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(frame, text, (text_x, text_y), font,
                font_scale * 3, (255, 255, 255), 3, cv2.LINE_AA)
    return frame

def display_progress(frame, counter: int, total: int, class_id: int):
    """Display progress information on frame."""
    # Calculate progress percentage
    progress = (counter / total) * 100
    
    # Create progress bar
    bar_width = 400
    bar_height = 20
    filled_width = int(bar_width * (counter / total))
    
    # Position progress bar at bottom of frame
    height = frame.shape[0]
    start_x = (frame.shape[1] - bar_width) // 2
    start_y = height - 50
    
    # Draw progress bar background
    cv2.rectangle(frame, (start_x, start_y), 
                 (start_x + bar_width, start_y + bar_height),
                 (100, 100, 100), -1)
    
    # Draw filled portion
    cv2.rectangle(frame, (start_x, start_y),
                 (start_x + filled_width, start_y + bar_height),
                 (0, 255, 0), -1)
    
    # Add text
    status_text = f"Class {class_id}: {counter}/{total} ({progress:.1f}%)"
    cv2.putText(frame, status_text, (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def collect_images_for_class(cap: cv2.VideoCapture, class_id: int):
    """Collect images for a specific class with enhanced feedback."""
    class_dir = os.path.join(DATA_DIR, str(class_id))
    print(f"\nPreparing to collect data for class {class_id}")
    print("Press 'q' to start the countdown when ready")
    print("Press 'esc' to skip to next class")
    print("Press 'ctrl+c' to exit program")

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            return

        instruction = 'Ready? Press "q" to start countdown!'
        cv2.putText(frame, instruction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Countdown sequence
    start_time = time.time()
    while time.time() - start_time < COUNTDOWN_TIME:
        ret, frame = cap.read()
        if not ret:
            continue
        
        remaining = int(COUNTDOWN_TIME - (time.time() - start_time))
        frame = show_countdown(frame, remaining)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)

    # Start capturing
    counter = 0
    last_capture_time = time.time()
    print(f"Capturing {DATASET_SIZE} images for class {class_id}...")
    
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            continue

        current_time = time.time()
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            # Add timestamp to filename for uniqueness
            timestamp = datetime.now().strftime("%H%M%S%f")
            image_path = os.path.join(class_dir, f"{counter}_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            counter += 1
            last_capture_time = current_time

        # Display progress
        display_frame = display_progress(frame.copy(), counter, DATASET_SIZE, class_id)
        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("Skipping to next class...")
            break

    print(f"Finished capturing images for class {class_id}")
    time.sleep(1)  # Brief pause between classes

def main():
    """Main function with enhanced error handling and user feedback."""
    print("\nImage Collection Script")
    print(f"Configuration:")
    print(f"- Number of classes: {NUMBER_OF_CLASSES}")
    print(f"- Images per class: {DATASET_SIZE}")
    print(f"- Data directory: {DATA_DIR}")
    print(f"- Countdown time: {COUNTDOWN_TIME} seconds")
    print(f"- Capture interval: {CAPTURE_INTERVAL} seconds")
    
    initialize_data_directories()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        sys.exit(1)

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow(WINDOW_NAME)

    try:
        for class_id in range(NUMBER_OF_CLASSES):
            collect_images_for_class(cap, class_id)
    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nData collection completed.")
        print(f"Data saved in: {os.path.abspath(DATA_DIR)}")

if __name__ == "__main__":
    main()
