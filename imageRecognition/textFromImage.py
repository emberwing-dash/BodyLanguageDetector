import os, io 
from google.cloud import vision
import pandas as pd

#SETUP OF CLIENT:
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"F:\python projects\setup\vital_chiller.json"

client = vision.ImageAnnotatorClient()

#IMAGE TO TEXT Code:
FILE1 = 'cautionSign.png'
FOLDER_PATH = r'F:\python projects\imageRecognition\img'

with io.open(os.path.join(FOLDER_PATH,FILE1),'rb') as img_file: #join path of image to src folder
    content = img_file.read()

#TEXT DETECTION Code:
image = vision.Image(content=content)
response = client.text_detection(image=image) #.json format
texts = response.text_annotations #extract 


#JSON DATA EXTRACTION (using pandas)
df = pd.DataFrame(columns=['locale', 'description']) #filtering only locale and description columns
for text in texts: #append locale and description in the form of dictionary (tabular form)
    df = df.append(
        dict(locale=text.locale,
             description=text.description),
             ignore_index = True)
    