import cv2
import os
import datetime
from deepface import DeepFace

class FaceMatcher:
    def __init__(self, target_img_path, output_folder='static/frames'):
        self.target_img_path = target_img_path
        self.output_folder = output_folder
        self.fps = 0
        os.makedirs(output_folder, exist_ok=True)

    def process_video(self, video_path):
        """
        Scans a video file for the target face.
        Returns a dictionary with match details if found, or None.
        """
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        frame_idx = 0
        FRAME_SKIP = 30  # Process 1 frame per second (assuming 30fps)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Optimization: Only process once per second
            if frame_idx % FRAME_SKIP == 0:
                try:
                    # Using DeepFace.verify with Facenet
                    # This handles detection + embedding + comparison in one go
                    result = DeepFace.verify(
                        img1_path=self.target_img_path,
                        img2_path=frame,
                        model_name="Facenet",
                        detector_backend="opencv", # Fast enough for video
                        enforce_detection=False,
                        threshold=0.4 # Tunable threshold
                    )
                    
                    if result['verified']:
                        # --- MATCH FOUND ---
                        
                        # 1. Calculate Timestamp
                        seconds = frame_idx / self.fps
                        timestamp_str = str(datetime.timedelta(seconds=int(seconds)))
                        
                        # 2. Save Matched Frame
                        match_filename = f"match_{os.path.basename(video_path)}_{frame_idx}.jpg"
                        save_path = os.path.join(self.output_folder, match_filename)
                        cv2.imwrite(save_path, frame)
                        
                        # 3. Return Match Data
                        return {
                            'found': True,
                            'video_file': os.path.basename(video_path),
                            'timestamp': timestamp_str,
                            'similarity_score': round((1 - result['distance']) * 100, 2), # Convert distance to %
                            'frame_path': save_path.replace("\\", "/") # Web-friendly path
                        }

                except Exception as e:
                    # print(e) # Debugging
                    pass

        cap.release()
        return None
