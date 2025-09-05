from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
import cv2
import numpy as np
import base64
from PIL import Image
import io
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Recreate the model architecture (same as in your training file)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(2,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,128,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.adapt_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.adapt_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=4).to(device)

# Load the saved weights
try:
    model.load_state_dict(torch.load(r"D:\XRAY DETECTION\X Ray Detection\XrayDetection.pth", map_location=device))
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Warning: best_model.pth not found. Please ensure the model file is in the same directory.")
except Exception as e:
    print(f"Error loading model: {e}")

# Class names (same order as in training)
CLASS_NAMES = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]

def preprocess_image(image_data, is_base64=True):
    """Preprocess image data for model prediction"""
    try:
        if is_base64:
            # Decode base64 image
            image_data = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
        else:
            image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Resize to model input size
        image_resized = cv2.resize(image_array, (256, 256))
        
        # Normalize
        image_normalized = image_resized / 255.0
        
        return image_normalized
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_images(xray_image, mask_image):
    """Make prediction using the loaded model"""
    try:
        # Preprocess both images
        xray_processed = preprocess_image(xray_image)
        mask_processed = preprocess_image(mask_image)
        
        if xray_processed is None or mask_processed is None:
            return None, "Error processing images"
        
        # Stack channels (same as in training)
        combined = np.stack([xray_processed, mask_processed], axis=0)  # Shape: (2, 256, 256)
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, 2, 256, 256)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': CLASS_NAMES[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {
                CLASS_NAMES[i]: float(probabilities[0][i]) 
                for i in range(len(CLASS_NAMES))
            }
        }, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

@app.route('/')
def index():
    """Serve the main webpage"""
    # Read the HTML file content
    try:
        with open(r'D:\XRAY DETECTION\X Ray Detection\WebPage.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return """
        <h1>Error: WebPage.html not found</h1>
        <p>Please ensure WebPage.html is in the same directory as this Flask app.</p>
        """

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.json
        
        if not data or 'xray_image' not in data or 'mask_image' not in data:
            return jsonify({
                'success': False,
                'error': 'Both xray_image and mask_image are required'
            }), 400
        
        # Get images from request
        xray_image = data['xray_image']
        mask_image = data['mask_image']
        
        # Make prediction
        result, error = predict_images(xray_image, mask_image)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 500
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_loaded = os.path.exists("best_model.pth") and 'model' in globals() and model is not None
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'model_path': 'best_model.pth',
        'device': str(device),
        'classes': CLASS_NAMES,
        'server_info': 'Simple COVID-19 X-ray Classification Server'
    })

if __name__ == '__main__':
    print(f"Starting Flask server on device: {device}")
    print(f"Model classes: {CLASS_NAMES}")
    app.run(debug=True, host='0.0.0.0', port=5000)