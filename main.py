# Import packages
import mediapipe as mp
import numpy as np 
import pydirectinput
import cv2
import time
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
# .venv\Scripts\Activate.ps1 -- for powershell 


# For drawing the skeleton on an image
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)
  return annotated_image

# For testing if the camera works 
def main():
    cap = cv2.VideoCapture(0)
    model_path = "model/pose_landmarker_full.task"

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options = BaseOptions(model_asset_path=model_path),
        running_mode = VisionRunningMode.VIDEO
    )

    # FPS is measured by counting frames over a 1-second window and only
    # refreshing the displayed value once per second, so the number is stable.
    fps_start_time = time.time()
    frame_count = 0
    display_fps = 0

    # FPS counter options
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 40)
    font_scale = 1
    color = (0, 128, 0)
    thickness = 2

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
        
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame. Exiting ...")
                break

            # Count frames and update the shown FPS at most once per second
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            fps_text = f"FPS: {int(display_fps)}"

            # Run pose detection on an RGB copy (MediaPipe wants RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # VIDEO mode needs a monotonically increasing timestamp in ms
            timestamp_ms = int(time.time() * 1000)
            # detect landmarkers for each frame in a video
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame = draw_landmarks_on_image(frame, result)

            # Step 3: normalizing for distance from the camera
            person = result.pose_landmarks[0]
            left_shoulder = person[11]
            right_shoulder = person[12]
            left_torso = person[23]
            right_torso = person[24]

            sh_x = (left_shoulder.x + right_shoulder.x) / 2
            sh_y = (left_shoulder.y + right_shoulder.y) / 2

            hip_x = (left_torso.x + right_torso.x) / 2
            hip_y = (left_torso.y + right_torso.y) / 2

            body_x = (sh_x + hip_x) / 2
            body_y = (sh_y + hip_y) / 2

            torso_size = ((sh_x - hip_x) ** 2 + (sh_y - hip_y) ** 2) ** 0.5


            # Display the resulting frame
            cv2.putText(frame, fps_text, position, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'): # "q" is the quit key
                break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()