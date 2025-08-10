import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import threading
import tkinter as tk

# ---------------------------
# Init MediaPipe
# ---------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# ---------------------------
# Config / thresholds
# ---------------------------
WAVE_HISTORY = 16            # frames to look at for waving
WAVE_MIN_CHANGES = 3         # number of direction changes in history to call it waving
POINTING_DISTANCE_RATIO = 1.2  # wrist distance relative to shoulder width to call "extended"
HAND_RAISE_OFFSET = -20      # pixels: wrist y < shoulder y + offset => raised (negative offset raises threshold)
OBJECT_NON_SKIN_THRESHOLD = 120  # pixel count of non-skin found in wrist ROI to indicate object
ROI_SIZE = 50                # wrist ROI half-size in px
FPS_SMOOTH = 0.9

# Behavior counters
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
emotion_label = "Neutral"

# Wrist history for waving
left_wrist_x_hist = deque(maxlen=WAVE_HISTORY)
right_wrist_x_hist = deque(maxlen=WAVE_HISTORY)

# ---------------------------
# Popup helpers (non-blocking)
# ---------------------------
popup_event = None
popup_thread = None
popup_shown = False

def intruder_popup_thread_fn(event: threading.Event):
    """Thread target to show a Tkinter popup; will close when event is set."""
    root = tk.Tk()
    root.title("Alert")
    # make the window small, always-on-top, centered
    root.attributes("-topmost", True)
    # Remove minimize/resize (platform dependent)
    root.resizable(False, False)

    label = tk.Label(root, text="INTRUDER DETECTED!", font=("Helvetica", 28, "bold"), fg="white", bg="red", padx=30, pady=20)
    label.pack()

    # center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 3) - (height // 2)
    root.geometry(f"+{x}+{y}")

    def check_event():
        if event.is_set():
            try:
                root.destroy()
            except Exception:
                pass
        else:
            root.after(100, check_event)

    root.after(100, check_event)
    try:
        root.mainloop()
    except Exception:
        pass

def show_intruder_popup():
    global popup_event, popup_thread, popup_shown
    if popup_shown:
        return
    popup_event = threading.Event()
    popup_thread = threading.Thread(target=intruder_popup_thread_fn, args=(popup_event,), daemon=True)
    popup_thread.start()
    popup_shown = True

def close_intruder_popup():
    global popup_event, popup_thread, popup_shown
    if not popup_shown:
        return
    try:
        popup_event.set()
    except Exception:
        pass
    # don't block too long on join; thread is daemon so it's fine
    popup_thread = None
    popup_event = None
    popup_shown = False

# ---------------------------
# Helpers (same as yours)
# ---------------------------
def to_pixel_coords(landmark, w, h):
    """Return integer pixel coords for a normalized landmark."""
    return int(landmark.x * w), int(landmark.y * h)

def avg_z(landmarks, indices):
    """Return average z for a list of pose landmark indices."""
    zs = [landmarks[i].z for i in indices]
    return sum(zs) / len(zs)

def detect_emotion(face_landmarks):
    lm = face_landmarks  # list-like with attributes x,y,z

    # Indices chosen from Mediapipe face mesh common mapping
    TOP_LIP = 13
    BOTTOM_LIP = 14
    LEFT_LIP = 61
    RIGHT_LIP = 291
    EYE_TOP_L = 159
    EYE_BOT_L = 145
    EYE_TOP_R = 386
    EYE_BOT_R = 374
    BROW_INNER_L = 70
    BROW_INNER_R = 300
    CHIN = 152
    NOSE_TIP = 1

    mouth_open = abs(lm[TOP_LIP].y - lm[BOTTOM_LIP].y)
    mouth_width = abs(lm[LEFT_LIP].x - lm[RIGHT_LIP].x)
    eye_open_left = abs(lm[EYE_TOP_L].y - lm[EYE_BOT_L].y)
    eye_open_right = abs(lm[EYE_TOP_R].y - lm[EYE_BOT_R].y)
    avg_eye_open = (eye_open_left + eye_open_right) / 2
    brow_sep = abs(lm[BROW_INNER_L].y - lm[BROW_INNER_R].y)
    nose_chin = abs(lm[NOSE_TIP].y - lm[CHIN].y)
    mouth_corner_drop = (lm[RIGHT_LIP].y + lm[LEFT_LIP].y) / 2 - lm[TOP_LIP].y

    # Heuristics (tune thresholds for your camera)
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

# ---------------------------
# Main loop
# ---------------------------
cap = cv2.VideoCapture(0)
prev_time = time.time()
fps = 0.0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe processing
        pose_results = pose.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)
        face_detect_results = face_detector.process(frame_rgb)

        # default posture values
        posture = "Neutral"
        lw_px = rw_px = (0, 0)

        # --- Face detection (multi-face) & intruder logic ---
        intruder_detected = False
        if face_detect_results and face_detect_results.detections:
            # collect bounding boxes and sizes to pick main person (largest area)
            face_boxes = []
            for det in face_detect_results.detections:
                bboxC = det.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)
                area = max(0, bw * bh)
                face_boxes.append((area, x, y, bw, bh))

            # sort by area descending
            face_boxes.sort(key=lambda t: t[0], reverse=True)
            num_faces = len(face_boxes)
            if num_faces > 1:
                intruder_detected = True
            else:
                intruder_detected = False

            # Draw boxes: largest = main (green), others = intruders (red)
            for idx, (area, x, y, bw, bh) in enumerate(face_boxes):
                if idx == 0:
                    color = (0, 255, 0)
                    label = "Main"
                else:
                    color = (0, 0, 255)
                    label = "Intruder"
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(frame, f"{label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # show/hide popup based on intruder_detected
        if intruder_detected:
            show_intruder_popup()
            # also show on-frame alert
            cv2.putText(frame, "INTRUDER DETECTED!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            # close popup if it was shown
            if popup_shown:
                close_intruder_popup()

        # --- Pose-based detections ---
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
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
            if is_waving(left_wrist_x_hist):
                behavior_counts["Waving"] += 1
            if is_waving(right_wrist_x_hist):
                behavior_counts["Waving"] += 1

            left_object = detect_object_in_hand(frame, lw_px)
            right_object = detect_object_in_hand(frame, rw_px)
            if left_object or right_object:
                behavior_counts["Object in Hand"] += 1

            for px in [lw_px, rw_px]:
                x, y = px
                cv2.rectangle(frame, (x - ROI_SIZE, y - ROI_SIZE), (x + ROI_SIZE, y + ROI_SIZE), (255, 120, 0), 1)

            cv2.putText(frame, f"Posture: {posture}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Face/emotion via face_mesh (detailed landmarks)
        if face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing.DrawingSpec((200,200,200), 1, 1))
                emotion_label = detect_emotion(face_landmarks.landmark)
            cv2.putText(frame, f"Emotion: {emotion_label}", (w - 360, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Behavior summary on frame
        y_offset = h - 220
        i = 0
        for label, count in behavior_counts.items():
            cv2.putText(frame, f"{label}: {count}", (10, y_offset + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            i += 1

        # FPS smoothing
        curr_time = time.time()
        inst_fps = 1.0 / (curr_time - prev_time + 1e-6)
        fps = FPS_SMOOTH * fps + (1 - FPS_SMOOTH) * inst_fps
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Interview Behavior & Emotion Analysis (Press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    try:
        pose.close()
    except Exception:
        pass
    try:
        face_mesh.close()
    except Exception:
        pass
    try:
        face_detector.close()
    except Exception:
        pass

    # ensure popup closed
    if popup_shown:
        try:
            popup_event.set()
        except Exception:
            pass