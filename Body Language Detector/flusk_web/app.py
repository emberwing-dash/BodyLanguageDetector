# app.py
import cv2
import mediapipe as mp
import numpy as np
import time
import base64
import threading
from collections import deque
from flask import Flask
from flask_socketio import SocketIO, emit

# ---------------------------
# Flask + SocketIO
# ---------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# Use eventlet for best performance; you can also use gevent
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ---------------------------
# Insert your detection code here (copied & slightly adapted)
# Keep functions: detect_posture, detect_gestures, detect_object_in_hand, detect_emotion, is_waving, etc.
# For brevity I'm pasting the essential parts and using some of your constants.
# ---------------------------

from collections import deque
import threading

# MediaPipe init
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Configs
WAVE_HISTORY = 16
WAVE_MIN_CHANGES = 3
POINTING_DISTANCE_RATIO = 1.2
HAND_RAISE_OFFSET = -20
OBJECT_NON_SKIN_THRESHOLD = 120
ROI_SIZE = 50
FPS_SMOOTH = 0.9

behavior_counts = {
    "Leaning Forward": 0,
    "Leaning Back": 0,
    "Arms Crossed": 0,
    "Hands on Table": 0,
    "Upright Posture": 0,
    "Neutral": 0,
    "Hand Raised": 0,
    "Pointing": 0,
    "Waving": 0,
    "Object in Hand": 0
}

left_wrist_x_hist = deque(maxlen=WAVE_HISTORY)
right_wrist_x_hist = deque(maxlen=WAVE_HISTORY)

# reuse your popup helpers but on server we won't open Tk popups (optional)
popup_shown = False

# Helper functions (cut down versions)
def is_waving(wrist_hist):
    if len(wrist_hist) < WAVE_HISTORY:
        return False
    diffs = np.diff(np.array(wrist_hist))
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return False
    changes = np.sum(np.abs(np.diff(signs)) > 0)
    return changes >= WAVE_MIN_CHANGES

def detect_object_in_hand(frame, wrist_px):
    h, w, _ = frame.shape
    x, y = wrist_px
    x1, y1 = max(0, x - ROI_SIZE), max(0, y - ROI_SIZE)
    x2, y2 = min(w, x + ROI_SIZE), min(h, y + ROI_SIZE)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 10, 60], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    non_skin = cv2.bitwise_not(skin_mask)
    non_skin_count = cv2.countNonZero(non_skin)
    return non_skin_count > OBJECT_NON_SKIN_THRESHOLD

def detect_posture(p_landmarks, w, h):
    def lp(idx):
        lm = p_landmarks[idx]
        return int(lm.x * w), int(lm.y * h), lm.z

    ls_x, ls_y, ls_z = lp(mp_pose.PoseLandmark.LEFT_SHOULDER)
    rs_x, rs_y, rs_z = lp(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    lh_x, lh_y, lh_z = lp(mp_pose.PoseLandmark.LEFT_HIP)
    rh_x, rh_y, rh_z = lp(mp_pose.PoseLandmark.RIGHT_HIP)
    lw_x, lw_y, lw_z = lp(mp_pose.PoseLandmark.LEFT_WRIST)
    rw_x, rw_y, rw_z = lp(mp_pose.PoseLandmark.RIGHT_WRIST)
    nose_x, nose_y, nose_z = lp(mp_pose.PoseLandmark.NOSE)

    shoulder_mid_x = (ls_x + rs_x) / 2.0
    shoulder_mid_y = (ls_y + rs_y) / 2.0
    hip_mid_x = (lh_x + rh_x) / 2.0
    hip_mid_y = (lh_y + rh_y) / 2.0

    shoulder_width = np.linalg.norm([ls_x - rs_x, ls_y - rs_y])
    shoulder_z = (ls_z + rs_z) / 2.0
    z_diff = shoulder_z - nose_z

    wrist_dist = np.linalg.norm([lw_x - rw_x, lw_y - rw_y])
    if wrist_dist < 0.45 * shoulder_width and (lw_y < ls_y and rw_y < rs_y):
        return "Arms Crossed", (lw_x, lw_y), (rw_x, rw_y)

    if lw_y > hip_mid_y + 30 and rw_y > hip_mid_y + 30:
        return "Hands on Table", (lw_x, lw_y), (rw_x, rw_y)

    if z_diff > 0.04:
        return "Leaning Forward", (lw_x, lw_y), (rw_x, rw_y)
    elif z_diff < -0.04:
        return "Leaning Back", (lw_x, lw_y), (rw_x, rw_y)

    torso_vert = abs((shoulder_mid_y - hip_mid_y) / (shoulder_width + 1e-6))
    if torso_vert > 0.20:
        return "Upright Posture", (lw_x, lw_y), (rw_x, rw_y)

    return "Neutral", (lw_x, lw_y), (rw_x, rw_y)

def detect_gestures(p_landmarks, w, h):
    ls = p_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = p_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lw = p_landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    rw = p_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    le = p_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    re = p_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

    ls_px = (int(ls.x * w), int(ls.y * h))
    rs_px = (int(rs.x * w), int(rs.y * h))
    lw_px = (int(lw.x * w), int(lw.y * h))
    rw_px = (int(rw.x * w), int(rw.y * h))
    le_px = (int(le.x * w), int(le.y * h))
    re_px = (int(re.x * w), int(re.y * h))

    shoulder_width = np.linalg.norm([ls_px[0] - rs_px[0], ls_px[1] - rs_px[1]])
    results = {"Hand Raised": False, "Pointing": False, "Waving_L": False, "Waving_R": False}

    if lw_px[1] < ls_px[1] + HAND_RAISE_OFFSET:
        results["Hand Raised"] = True
    if rw_px[1] < rs_px[1] + HAND_RAISE_OFFSET:
        results["Hand Raised"] = True

    def angle(a, b, c):
        va = np.array([a[0] - b[0], a[1] - b[1]])
        vc = np.array([c[0] - b[0], c[1] - b[1]])
        denom = (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-6)
        cosang = np.clip(np.dot(va, vc) / denom, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    left_elbow_angle = angle(ls_px, le_px, lw_px)
    left_wrist_from_shoulder = np.linalg.norm([lw_px[0] - ls_px[0], lw_px[1] - ls_px[1]])
    if left_elbow_angle > 150 and left_wrist_from_shoulder > POINTING_DISTANCE_RATIO * shoulder_width:
        results["Pointing"] = True

    right_elbow_angle = angle(rs_px, re_px, rw_px)
    right_wrist_from_shoulder = np.linalg.norm([rw_px[0] - rs_px[0], rw_px[1] - rs_px[1]])
    if right_elbow_angle > 150 and right_wrist_from_shoulder > POINTING_DISTANCE_RATIO * shoulder_width:
        results["Pointing"] = True

    return results, lw_px, rw_px

def detect_emotion_from_mesh(face_landmarks):
    # your heuristic emotion detector from face mesh landmarks (copied minimal)
    lm = face_landmarks
    TOP_LIP = 13; BOTTOM_LIP = 14; LEFT_LIP = 61; RIGHT_LIP = 291
    EYE_TOP_L = 159; EYE_BOT_L = 145; EYE_TOP_R = 386; EYE_BOT_R = 374
    BROW_INNER_L = 70; BROW_INNER_R = 300; CHIN = 152; NOSE_TIP = 1
    mouth_open = abs(lm[TOP_LIP].y - lm[BOTTOM_LIP].y)
    mouth_width = abs(lm[LEFT_LIP].x - lm[RIGHT_LIP].x)
    eye_open_left = abs(lm[EYE_TOP_L].y - lm[EYE_BOT_L].y)
    eye_open_right = abs(lm[EYE_TOP_R].y - lm[EYE_BOT_R].y)
    avg_eye_open = (eye_open_left + eye_open_right) / 2
    brow_sep = abs(lm[BROW_INNER_L].y - lm[BROW_INNER_R].y)
    mouth_corner_drop = (lm[RIGHT_LIP].y + lm[LEFT_LIP].y) / 2 - lm[TOP_LIP].y

    if avg_eye_open > 0.045 and mouth_open > 0.05:
        return "Surprised"
    if avg_eye_open < 0.015:
        return "Tired"
    if mouth_width > 0.08 and mouth_open < 0.03:
        return "Smiling"
    if mouth_open > 0.045:
        return "Talking"
    if brow_sep < 0.015 and mouth_open < 0.03:
        return "Angry"
    if mouth_width < 0.06 and mouth_corner_drop < 0.0:
        return "Sad"
    return "Neutral"

# ---------------------------
# Camera & thread to emit frames
# ---------------------------
cap = cv2.VideoCapture(0)
emit_thread = None
thread_lock = threading.Lock()

def background_camera_emit():
    global cap, behavior_counts
    prev_time = time.time()
    fps = 0.0
    while True:
        success, frame = cap.read()
        if not success:
            socketio.sleep(0.01)
            continue

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = pose.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)
        face_detect_results = face_detector.process(frame_rgb)

        posture = "Neutral"
        lw_px = rw_px = (0, 0)
        emotion_label = "Neutral"
        feedback = ""

        # face detection boxes + intruder logic
        intruder_detected = False
        if face_detect_results and face_detect_results.detections:
            face_boxes = []
            for det in face_detect_results.detections:
                bboxC = det.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)
                area = max(0, bw * bh)
                face_boxes.append((area, x, y, bw, bh))
            face_boxes.sort(key=lambda t: t[0], reverse=True)
            if len(face_boxes) > 1:
                intruder_detected = True

            # draw boxes (server-side rendering optional)
            for idx, (area, x, y, bw, bh) in enumerate(face_boxes):
                color = (0,255,0) if idx==0 else (0,0,255)
                cv2.rectangle(frame, (x,y), (x+bw, y+bh), color, 2)

        # pose-based
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            p_landmarks = pose_results.pose_landmarks.landmark
            posture, lw_px, rw_px = detect_posture(p_landmarks, w, h)
            behavior_counts[posture] = behavior_counts.get(posture, 0) + 1

            gestures, lw_px, rw_px = detect_gestures(p_landmarks, w, h)
            if gestures["Hand Raised"]:
                behavior_counts["Hand Raised"] += 1
            if gestures["Pointing"]:
                behavior_counts["Pointing"] += 1

            left_wrist_x_hist.append(lw_px[0])
            right_wrist_x_hist.append(rw_px[0])
            if is_waving(left_wrist_x_hist) or is_waving(right_wrist_x_hist):
                behavior_counts["Waving"] += 1

            left_object = detect_object_in_hand(frame, lw_px)
            right_object = detect_object_in_hand(frame, rw_px)
            if left_object or right_object:
                behavior_counts["Object in Hand"] += 1

            for px in [lw_px, rw_px]:
                x,y = px
                cv2.rectangle(frame, (x-ROI_SIZE, y-ROI_SIZE), (x+ROI_SIZE, y+ROI_SIZE), (255,120,0), 1)

        # face emotion via mesh
        if face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                emotion_label = detect_emotion_from_mesh(face_landmarks.landmark)

        # simple feedback rules (you can expand or plug GPT-2 here)
        if emotion_label == "Angry" and posture == "Leaning Forward":
            feedback = "Aggressive posture detected"
        elif emotion_label == "Smiling" and behavior_counts["Waving"] > 0:
            feedback = "Friendly greeting"
        else:
            feedback = f"{emotion_label} / {posture}"

        # encode frame as jpeg + base64
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        payload = {
            'image': 'data:image/jpeg;base64,' + jpg_as_text,
            'emotion': emotion_label,
            'posture': posture,
            'feedback': feedback,
            'behaviors': behavior_counts,
            'intruder': intruder_detected
        }

        # emit to clients
        socketio.emit('frame', payload)
        # small sleep to yield to socketio (controls frame rate)
        socketio.sleep(0.03)

@socketio.on('connect')
def handle_connect():
    global emit_thread
    with thread_lock:
        if emit_thread is None:
            emit_thread = socketio.start_background_task(background_camera_emit)
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # Use eventlet (pip install eventlet) — recommended for socketio streaming
    socketio.run(app, host='0.0.0.0', port=5000)
