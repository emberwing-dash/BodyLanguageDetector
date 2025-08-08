import os, io 
from google.cloud import vision

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"F:\python projects\setup\vital_chiller.json"

client = vision.ImageAnnotatorClient()