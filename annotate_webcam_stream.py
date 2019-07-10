import pyttsx3
import engineio
import requests
import cv2
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from invoke import run
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", type=int, default=7777, help="Port of caption generator API")
parser.add_argument("-e", "--engine", type=int, default=0, help="0 if you want to use the 'say' Speech Synthesis Manager engine, 1 if you want to use pyttsx3")
args = parser.parse_args()

url_coco = 'http://localhost:{}/generate_caption_ms_coco'.format(args.port)
url_cc = 'http://localhost:{}/generate_caption_conceptual_captions'.format(args.port)

special_toks = ('xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '-')

if args.engine == 1:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    engine.setProperty('voice', voices[0].id)
    #engine.setProperty('rate', 150)

elif args.engine > 1:
    print("Invalid TTS engine chosen")
    sys.exit()

def remove_special_toks(cap):
    return ' '.join([w for w in cap.split() if w not in special_toks])

def say_caption(caption):
    if args.engine == 1:
        engine.say(caption)
        engine.runAndWait()
    else:
        cmd = "say -v Alex {}".format(caption)
        result = run(cmd, hide=True, warn=True)

def get_caption(img, url):
    imgByteArr = BytesIO()
    img.save(imgByteArr, format='JPEG')
    imgByteArr.seek(0)
    
    response = requests.post(url, files={'input_image': imgByteArr})
    if response.ok:
        return response.text
    else:
        return "Could not connect to API"

cam = cv2.VideoCapture(0)
cam.open(0)
cv2.namedWindow("Window_1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Window_1", 960,600)

while True:
    ret, frame = cam.read()
    cv2.imshow("Window_1", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame, 'RGB')
        caption = get_caption(frame, url_coco)
        say_caption(remove_special_toks(caption))
        print(caption)
    elif k%256 == 13:
        # RET pressed
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame, 'RGB')
        caption = get_caption(frame, url_cc)
        say_caption(remove_special_toks(caption))
        print(caption)
        
cam.release()

cv2.destroyAllWindows()
