import cv2
import mediapipe as mp
import time
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
sujood_count = 0
rakat_count = 0
in_sujood = False
last_sujood_time = 0
tracking_active = False
calibrated = False
calibrating = False
calibration_data = {}
calibration_start_time = 0
message = ""
message_timer = 0

buttons = {
    "start": {"pos": (20, 20), "size": (180, 60), "label": "Begin"},
    "calibrate": {"pos": (220, 20), "size": (220, 60), "label": "Calibrate"},
    "reset": {"pos": (460, 20), "size": (160, 60), "label": "Reset"},
}

def draw_button(frame, key):
    x, y = buttons[key]["pos"]
    w, h = buttons[key]["size"]
    label = buttons[key]["label"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), -1)  # orange fill
    cv2.putText(frame, label, (x + 15, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

def check_button_click(x, y):
    for key, val in buttons.items():
        bx, by = val["pos"]
        bw, bh = val["size"]
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return key
    return None

def calibrate(landmarks):
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y
    avg_knee_y = (left_knee_y + right_knee_y) / 2
    return {'nose_y': nose_y, 'knee_y': avg_knee_y}

def get_center_point(landmarks, frame):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
    center_x = int((left_shoulder + right_shoulder) / 2 * frame.shape[1])
    return center_x

def mouse_callback(event, x, y, flags, param):
    global tracking_active, calibrated, calibrating
    global calibration_start_time, sujood_count, rakat_count
    global message, message_timer

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = check_button_click(x, y)
        if clicked == "start":
            if calibrated:
                tracking_active = not tracking_active
                buttons["start"]["label"] = "Stop" if tracking_active else "Start"
                message = "Tracking started." if tracking_active else "Tracking stopped."
            else:
                message = "Please calibrate first."
            message_timer = time.time()

        elif clicked == "calibrate":
            calibrating = True
            calibration_start_time = time.time()
            message = "Starting calibration..."
            message_timer = time.time()

        elif clicked == "reset":
            sujood_count = 0
            rakat_count = 0
            in_sujood = False
            message = "Counters reset."
            message_timer = time.time()

cv2.namedWindow("Salah Rakat Counter")
cv2.setMouseCallback("Salah Rakat Counter", mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    overlay = frame.copy()
    alpha = 0.9
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark

        if calibrating:
            countdown = 5 - int(time.time() - calibration_start_time)
            if countdown <= 0:
                calibration_data = calibrate(landmarks)
                calibrated = True
                calibrating = False
                message = "Calibration completed."
                message_timer = time.time()
            else:
                cv2.putText(frame, f'Calibrating in: {countdown}', (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

        elif tracking_active and calibrated:
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
            right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y
            avg_knee_y = (left_knee_y + right_knee_y) / 2

            if nose_y > calibration_data['knee_y'] + 0.05:
                if not in_sujood:
                    in_sujood = True
                    sujood_count += 1
                    last_sujood_time = time.time()

                    if sujood_count % 2 == 0:
                        rakat_count += 1
            else:
                if in_sujood and time.time() - last_sujood_time > 1.5:
                    in_sujood = False
            center_x = get_center_point(landmarks, frame)
            overlay_text = f'{rakat_count}'
            font_scale = 6
            thickness = 10
            size = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.putText(frame, overlay_text, (center_x - size[0] // 2, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    for key in buttons:
        draw_button(frame, key)

    status = "Tracking: ON" if tracking_active else "Tracking: OFF"
    cv2.putText(frame, status, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if tracking_active else (0, 0, 255), 2)

    if not calibrated:
        cv2.putText(frame, "Not calibrated. Click 'Calibrate' to start.", (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)

    if message and time.time() - message_timer < 3:
        cv2.putText(frame, message, (10, frame.shape[0] - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Salah Rakat Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
