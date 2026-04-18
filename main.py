import cv2
import time
import urllib.request
import urllib.error
import os
import sys
import logging
import mediapipe as mp
import customtkinter as ctk
from PIL import Image

logging.basicConfig(
    filename='salah_tracker.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("--- Application Starting ---")

THEME_BG = "#000000"
THEME_BAR = "#050505"
COLOR_NEON_PURPLE = "#8A2BE2"
COLOR_HOVER_PURPLE = "#4B0082"
COLOR_CV2_PURPLE = (255, 100, 255) 
COLOR_CV2_SHADOW = (50, 0, 50)
COLOR_CV2_GREEN = (0, 255, 150)
COLOR_CV2_RED = (50, 50, 255)

class RakatTracker:
    def __init__(self):
        self.sujood_count = 0
        self.rakat_count = 0
        self.in_sujood = False
        self.last_sujood_time = 0
        self.is_tracking = False
        self.is_calibrated = False
        self.target_locked = False 
        self.calibration_data = {}
        
        self.model_path = 'pose_landmarker_lite.task'
        self._ensure_model_exists()

        try:
            BaseOptions = mp.tasks.BaseOptions
            # STRICT AI PROTOCOLS: Prevents tracking background objects (Basketballs)
            self.options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1, # Lock onto a single person only
                min_pose_detection_confidence=0.75, 
                min_pose_presence_confidence=0.75,  
                min_tracking_confidence=0.75
            )
            self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(self.options)
            logging.info("MediaPipe initialized with STRICT parameters.")
        except Exception as e:
            logging.critical(f"Failed to initialize MediaPipe: {e}")
            sys.exit("Critical Error: MediaPipe failed to load. Check logs.")

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            logging.info("Downloading AI model...")
            try:
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                    self.model_path
                )
            except urllib.error.URLError as e:
                logging.error(f"Network error: {e}")
                sys.exit("Critical Error: Could not download AI model.")

    def calibrate(self, landmarks):
        try:
            nose_y = landmarks[0].y
            avg_knee_y = (landmarks[25].y + landmarks[26].y) / 2
            self.calibration_data = {'nose_y': nose_y, 'knee_y': avg_knee_y}
            self.is_calibrated = True
            logging.info("Calibration successful.")
        except IndexError:
            pass

    def update_logic(self, landmarks):
        if not self.is_tracking or not self.is_calibrated:
            return

        try:
            nose = landmarks[0]
            
            # OCCLUSION CHECK: Ignore frame if face is mostly off-camera
            if nose.visibility < 0.5:
                self.target_locked = False
                return 
                
            self.target_locked = True
            nose_y = nose.y
            
            # Sujood Logic: Nose drops below calibrated knee level
            if nose_y > self.calibration_data['knee_y'] + 0.05:
                if not self.in_sujood:
                    self.in_sujood = True
                    self.sujood_count += 1
                    self.last_sujood_time = time.time()
                    if self.sujood_count % 2 == 0:
                        self.rakat_count += 1
            else:
                # Debounce: Must remain up for 1.5s to end Sujood state
                if self.in_sujood and time.time() - self.last_sujood_time > 1.5:
                    self.in_sujood = False
                    
        except Exception as e:
            logging.error(f"Logic error: {e}")

    def draw_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        try:
            for idx in [0, 11, 12, 25, 26]:
                point = landmarks[idx]
                if point.visibility > 0.5: # Only draw if AI is confident
                    cx, cy = int(point.x * w), int(point.y * h)
                    cv2.circle(frame, (cx, cy), 12, COLOR_CV2_PURPLE, 1)
                    cv2.circle(frame, (cx, cy), 6, COLOR_CV2_PURPLE, -1)
        except IndexError:
            pass 

    def reset(self):
        self.sujood_count = 0
        self.rakat_count = 0
        self.in_sujood = False
        self.is_tracking = False
        self.target_locked = False

    def cleanup(self):
        self.landmarker.close()

        
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.tracker = RakatTracker()
        
        self.title("Salah Rakat Counter Pro")
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda e: self.on_closing())
        
        ctk.set_appearance_mode("dark")
        self.configure(fg_color=THEME_BG) 
        
        self.grid_rowconfigure(0, weight=1) 
        self.grid_rowconfigure(1, weight=0) 
        self.grid_columnconfigure(0, weight=1)

        # Video Layer
        self.video_label = ctk.CTkLabel(self, text="Initializing Camera...", text_color=COLOR_NEON_PURPLE, font=("Arial", 24))
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Control Bar
        self.control_frame = ctk.CTkFrame(self, height=100, fg_color=THEME_BAR, corner_radius=0)
        self.control_frame.grid(row=1, column=0, sticky="ew")
        self.control_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        base_btn_style = {
            "font": ("Helvetica", 15, "bold"), "height": 50, "width": 180,
            "corner_radius": 25, "border_width": 2, "fg_color": THEME_BAR, "text_color": "#FFFFFF"
        }

        self.btn_calibrate = ctk.CTkButton(self.control_frame, text="1. CALIBRATE", 
                                           border_color=COLOR_NEON_PURPLE, hover_color=COLOR_HOVER_PURPLE, 
                                           command=self.start_calibration, **base_btn_style)
        self.btn_calibrate.grid(row=0, column=0, pady=20)

        self.btn_start = ctk.CTkButton(self.control_frame, text="2. START", 
                                       border_color=COLOR_NEON_PURPLE, hover_color=COLOR_HOVER_PURPLE,
                                       command=self.toggle_tracking, state="disabled", **base_btn_style)
        self.btn_start.grid(row=0, column=1, pady=20)

        self.btn_reset = ctk.CTkButton(self.control_frame, text="RESET", 
                                       border_color="#555555", hover_color="#333333", 
                                       command=self.reset_app, **base_btn_style)
        self.btn_reset.grid(row=0, column=2, pady=20)

        self.btn_exit = ctk.CTkButton(self.control_frame, text="EXIT", 
                                      border_color="#FF3366", hover_color="#990033", 
                                      command=self.on_closing, **base_btn_style)
        self.btn_exit.grid(row=0, column=3, pady=20)

        # Hardware Auto-Discovery
        self.cap = self._initialize_camera()
        
        self.status_text = "RAKAT COUNTER READY"
        self.is_calibrating = False
        self.calibration_start_time = 0
        self.camera_failed = False

        self.update_video_feed()

    def _initialize_camera(self):
        for cam_idx in [0, 1, 2]:
            cap = cv2.VideoCapture(cam_idx)
            if cap.isOpened() and cap.read()[0]:
                return cap
            cap.release()
        self.video_label.configure(text="CRITICAL: No Camera Detected.", text_color="#FF3366")
        return None

    def start_calibration(self):
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.status_text = "CALIBRATING... STAND STRAIGHT"

    def toggle_tracking(self):
        self.tracker.is_tracking = not self.tracker.is_tracking
        if self.tracker.is_tracking:
            self.btn_start.configure(text="PAUSE", fg_color=COLOR_HOVER_PURPLE)
            self.status_text = "TRACKING ACTIVE"
        else:
            self.btn_start.configure(text="RESUME", fg_color=THEME_BAR)
            self.status_text = "PAUSED"

    def reset_app(self):
        self.tracker.reset()
        self.btn_start.configure(text="2. START", state="disabled", fg_color=THEME_BAR)
        self.status_text = "COUNTER RESET"

    def update_video_feed(self):
        if self.cap is None: return

        ret, frame = self.cap.read()
        if not ret:
            self.video_label.configure(image="", text="CAMERA DISCONNECTED", text_color="#FF3366")
            self.after(1000, self.update_video_feed) 
            return

        if self.video_label.cget("text") != "":
            self.video_label.configure(text="")

        win_w = max(100, self.video_label.winfo_width())
        win_h = max(100, self.video_label.winfo_height())

        frame = cv2.flip(frame, 1)
        
        # 1. PERFECT ASPECT-FILL CROP (No more stretching)
        h_orig, w_orig, _ = frame.shape
        aspect_orig = w_orig / h_orig
        aspect_win = win_w / win_h

        if aspect_win > aspect_orig:
            new_w = win_w
            new_h = int(win_w / aspect_orig)
        else:
            new_h = win_h
            new_w = int(win_h * aspect_orig)
            
        frame = cv2.resize(frame, (new_w, new_h))
        y_start = max(0, (new_h - win_h) // 2)
        x_start = max(0, (new_w - win_w) // 2)
        frame = frame[y_start:y_start+win_h, x_start:x_start+win_w]
        
        try:
            # 2. AI PREDICTION
            rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_display)
            self.current_result = self.tracker.landmarker.detect_for_video(mp_image, int(time.time() * 1000))
            
            # 3. CINEMATIC OVERLAY
            overlay = rgb_display.copy()
            cv2.rectangle(overlay, (0, 0), (win_w, win_h), (0, 0, 0), -1)
            rgb_display = cv2.addWeighted(overlay, 0.85, rgb_display, 0.15, 0)

            # 4. LOGIC & DRAWING
            if hasattr(self, 'current_result') and self.current_result.pose_landmarks:
                landmarks = self.current_result.pose_landmarks[0]
                self.tracker.draw_landmarks(rgb_display, landmarks)
                self.tracker.update_logic(landmarks)

                if self.is_calibrating:
                    countdown = str(5 - int(time.time() - self.calibration_start_time))
                    if int(countdown) <= 0:
                        self.tracker.calibrate(landmarks)
                        self.is_calibrating = False
                        self.status_text = "CALIBRATION COMPLETE"
                        self.btn_start.configure(state="normal")
                    else:
                        c_font, c_scale, c_thick = cv2.FONT_HERSHEY_SIMPLEX, 8, 15
                        c_size = cv2.getTextSize(countdown, c_font, c_scale, c_thick)[0]
                        cv2.putText(rgb_display, countdown, ((win_w - c_size[0]) // 2, (win_h + c_size[1]) // 2), 
                                    c_font, c_scale, COLOR_CV2_PURPLE, c_thick)
            else:
                self.tracker.target_locked = False # AI lost the pose

            # 5. UI HUD (Massive Number & Telemetry)
            if self.tracker.is_tracking:
                # Math-Centered Number
                num_text = str(self.tracker.rakat_count)
                n_font, n_scale, n_thick = cv2.FONT_HERSHEY_SIMPLEX, 15, 25
                n_size = cv2.getTextSize(num_text, n_font, n_scale, n_thick)[0]
                nx, ny = (win_w - n_size[0]) // 2, (win_h + n_size[1]) // 2
                
                cv2.putText(rgb_display, num_text, (nx + 10, ny + 10), n_font, n_scale, COLOR_CV2_SHADOW, n_thick + 5)
                cv2.putText(rgb_display, num_text, (nx, ny), n_font, n_scale, COLOR_CV2_PURPLE, n_thick)

                # Target Lock Telemetry (Top Right)
                if self.tracker.target_locked:
                    lock_txt, lock_color = "[ MUMIN IN FRAME ]", COLOR_CV2_GREEN
                else:
                    lock_txt, lock_color = "[ LOOKING FOR MUMIN ]", COLOR_CV2_RED
                
                lock_size = cv2.getTextSize(lock_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.putText(rgb_display, lock_txt, (win_w - lock_size[0] - 30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, lock_color, 2)

            # Main Status (Top Left)
            cv2.putText(rgb_display, self.status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_CV2_PURPLE, 2)

            # Render
            img_pil = Image.fromarray(rgb_display)
            ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(win_w, win_h))
            self.video_label.configure(image=ctk_img)
            self.video_label.image = ctk_img

        except Exception as e:
            logging.error(f"Render loop error: {e}")

        self.after(15, self.update_video_feed)

    def on_closing(self):
        if self.cap: self.cap.release()
        self.tracker.cleanup()
        self.destroy()
        sys.exit(0)

if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        logging.critical(f"App crash: {e}")
