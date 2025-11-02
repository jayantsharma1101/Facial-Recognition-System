import cv2
from deepface import DeepFace
import time
import os

# --- INSTRUCTIONS & CONFIGURATION ---
# IMPORTANT: For this script to work, you MUST create a folder named 'my_db'
# in the same directory as this Python script. Place images of the people
# you want to recognize inside this 'my_db' folder (e.g., my_db/John_Doe.jpg).

# Built-in camera index and AVFoundation backend are kept for stability.
CAMERA_INDEX = 0
CAMERA_BACKEND = cv2.CAP_AVFOUNDATION

# Configuration for DeepFace Recognition
DATABASE_PATH = "my_db"
# VGG-Face is the default and robust model for recognition
MODEL_NAME = "VGG-Face"
# Metric (Cosine similarity is standard for VGG-Face)
DISTANCE_METRIC = "cosine"

def run_face_recognition():
    """Initializes the DeepFace recognition pipeline and runs the real-time recognition loop."""

    # --- 1. PREPARATION: Build the Face Database (Creates embeddings/representations.pkl) ---
    print(f"Building face database for directory: {DATABASE_PATH}")
    if not os.path.isdir(DATABASE_PATH):
        print("=" * 60)
        print(f"FATAL ERROR: Database folder '{DATABASE_PATH}' not found.")
        print("Please create a folder named 'my_db' and add images of known people.")
        print("=" * 60)
        return

    # DeepFace will automatically calculate embeddings and save them to a file named
    # 'representations_vgg_face.pkl' inside the my_db folder.
    try:
        # We call find once to force the database embedding creation
        # We pass a placeholder image since we don't have a live face yet.
        # This is a common trick to ensure the database is built/loaded first.
        DeepFace.find(
            img_path=f"{DATABASE_PATH}/" + os.listdir(DATABASE_PATH)[0],
            db_path=DATABASE_PATH,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False, # <--- CRITICAL FIX: Changed to False to prevent immediate database failure
            silent=True # Suppress console output during database build
        )
        print("Face representations loaded/created successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not build face database or load model ({MODEL_NAME}).")
        print(f"DeepFace Details: {e}")
        return

    # --- 2. CAMERA INITIALIZATION ---
    print(f"\nAttempting to open camera index {CAMERA_INDEX} using CV_AVFOUNDATION backend...")
    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)

    if not cap.isOpened():
        print("CRITICAL CAMERA FAILURE.")
        return

    # Set Resolution and CRUCIAL DELAY
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera initialized successfully. Press 'q' to exit the stream.")
    time.sleep(2) # Allow stream buffer to fill

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Error: Lost connection to camera. Attempting reconnect.")
                time.sleep(1)
                cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    time.sleep(2)
                    continue
                else:
                    print("Failed to reconnect. Breaking loop.")
                    break

            # --- 3. RECOGNITION PROCESSING ---
            recognized_frame = frame.copy()

            # DeepFace.find handles face detection, embedding generation, and comparison in one call.
            try:
                # The 'find' function returns a list of dataframes (one for each detected face)
                results = DeepFace.find(
                    img_path=frame,
                    db_path=DATABASE_PATH,
                    model_name=MODEL_NAME,
                    distance_metric=DISTANCE_METRIC,
                    enforce_detection=False, # This is already False for real-time processing
                    detector_backend='opencv', # Fast backend for real-time performance
                    silent=True
                )

                # Check results for recognized faces
                if results and isinstance(results, list):
                    for result_df in results:
                        # DeepFace find returns a DataFrame; we use the first match if available
                        if not result_df.empty:
                            # 4. Extract Identity and Bounding Box
                            identity_path = result_df.iloc[0]['identity']
                            # The identity column contains the path (e.g., 'my_db/Jane_Doe.jpg')
                            name = os.path.basename(identity_path).split('.')[0].replace('_', ' ') # Extracts 'Jane Doe'

                            # DeepFace detection coordinates (face_x, face_y, w, h)
                            x = int(result_df.iloc[0]['source_x'])
                            y = int(result_df.iloc[0]['source_y'])
                            w = int(result_df.iloc[0]['source_w'])
                            h = int(result_df.iloc[0]['source_h'])

                            confidence = 1 - result_df.iloc[0]['distance']
                            label = f"{name} ({confidence:.2f})"

                            # 5. Visual Annotation
                            cv2.rectangle(recognized_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.rectangle(recognized_frame, (x, y - 30), (x + w, y), (255, 0, 0), -1)
                            cv2.putText(recognized_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            except Exception as e:
                # This catch handles issues during the recognition search, though less common with enforce_detection=False
                print(f"Recognition search error: {e}")

            # 6. Display Output
            cv2.imshow('Real-time Face Recognition', recognized_frame)

            # 7. Loop Termination
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"\nAn unexpected runtime error occurred: {e}")
    finally:
        # 8. Cleanup
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete. Exiting application.")

if __name__ == "__main__":
    run_face_recognition()
