import os
from google.cloud import vision

# Correct path format
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"F:\python projects\setup\vital_chiller.json"

# Initialize client
client = vision.ImageAnnotatorClient()

# You can now use the client safely
print("Vision API client initialized successfully.")

