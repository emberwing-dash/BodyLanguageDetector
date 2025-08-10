import cv2
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
from collections import deque
import numpy as np

# Config
MODEL_PATH = "cnn_lstm_meld.pth"
N_FRAMES = 8
IMG_SIZE = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace this with your actual emotion labels, in the same order as training
LABELS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# CNN + LSTM Model class (same as training)
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(CNN_LSTM, self).__init__()
        cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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

# Transform for frames
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def main():
    print("Loading model...")
    model = CNN_LSTM(hidden_dim=256, num_classes=len(LABELS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded.")

    # Frame buffer for last N_FRAMES
    frame_buffer = deque(maxlen=N_FRAMES)

    cap = cv2.VideoCapture(0)  # Open webcam 0
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Starting real-time emotion detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess and add to buffer
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_t = transform(frame_rgb)  # [C,H,W]
        frame_buffer.append(frame_t)

        # When buffer filled, run prediction
        if len(frame_buffer) == N_FRAMES:
            input_tensor = torch.stack(list(frame_buffer))  # [T,C,H,W]
            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)  # [1,T,C,H,W]

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
                pred_label = LABELS[pred_idx.item()]
                confidence = conf.item()

            # Display predicted emotion on frame
            text = f"{pred_label}: {confidence*100:.1f}%"
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Real-time Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
