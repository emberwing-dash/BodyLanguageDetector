import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

# ------------------------------
# SETTINGS
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "F:\Body Language Detector\pytorch_setup\cnn_lstm_meld.pth"  # Change to your actual .pth file path
EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]  # Update if needed
SEQUENCE_LENGTH = 16  # Frames per sequence

# ------------------------------
# FIND CAMERA INDEX
# ------------------------------
def find_working_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            print(f"✅ Found working camera at index {i}")
            return i
        cap.release()
    print("❌ No working camera found.")
    return None

# ------------------------------
# MODEL DEFINITION
# ------------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(CNN_LSTM, self).__init__()
        cnn = models.resnet18(weights=None)  # No warning
        modules = list(cnn.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.cnn_out_dim = cnn.fc.in_features

        self.lstm = nn.LSTM(self.cnn_out_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        cnn_features = []
        for t in range(T):
            f = self.cnn(x[:, t])
            f = f.view(B, -1)
            cnn_features.append(f)
        feats = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(feats)
        last_feat = lstm_out[:, -1, :]
        out = self.fc(last_feat)
        return out

# ------------------------------
# LOAD TRAINED MODEL
# ------------------------------
model = CNN_LSTM(hidden_dim=256, num_classes=len(EMOTIONS)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))  # No warning
model.eval()

# ------------------------------
# PREPROCESS FRAME
# ------------------------------
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.tensor(frame / 255.0, dtype=torch.float32).permute(2, 0, 1)
    return frame

# ------------------------------
# RUN REALTIME DETECTION
# ------------------------------
camera_index = find_working_camera()
if camera_index is None:
    exit()

cap = cv2.VideoCapture(camera_index)
frames = []

print("📸 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    processed = preprocess_frame(frame)
    frames.append(processed)

    # Keep only last N frames
    if len(frames) > SEQUENCE_LENGTH:
        frames.pop(0)

    # Predict when we have enough frames
    if len(frames) == SEQUENCE_LENGTH:
        input_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)  # Shape [1, T, C, H, W]
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = EMOTIONS[predicted.item()]

        cv2.putText(frame, f"Emotion: {emotion}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()