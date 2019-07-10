import matplotlib
matplotlib.use("Agg")  # so that import pyplot does not try to pull in a GUI
from flask import Flask, request, send_file
from flasgger import Swagger
#from fastai.text import *
from torchvision import transforms, models
from PIL import Image
from caption_generator_model import *
from utils import *
from BeamSearch import *
import torch
import time

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

app = Flask(__name__)
app.config['SWAGGER'] = {
    "swagger_version": "2.0",
    "title": "Image Caption Generator",
    "description": "Generates image captions using a model based on the article 'Show, Attend and Tell: Neural Image Caption Generation with Visual Attention' by Xu et al. (2016)."}

swagger = Swagger(app)

n_layers, emb_sz = 1, 500
beam_width = 5

vocab_ms_coco = pickle.load(open("vocab_ms_coco.pkl", "rb" ))
vocab_cc = pickle.load(open("vocab_conceptual_captions.pkl", "rb" ))

model_ms_coco = torch.load('model_ms_coco.pth', map_location=device);
model_ms_coco = model_ms_coco.to(device)
model_ms_coco.eval();

model_cc = torch.load('model_conceptual_captions.pth', map_location=device);
model_cc = model_cc.to(device)
model_cc.eval();

model_ms_coco.device = device
model_cc.device = device

beam_search_ms_coco = BeamSearch(model_ms_coco.encode, model_ms_coco.decode_step, beam_width, device=device)
beam_search_cc = BeamSearch(model_cc.encode, model_cc.decode_step, beam_width, device=device)

sz = 224

valid_tfms = transforms.Compose([
    transforms.Resize(sz),
    transforms.CenterCrop(sz),
    transforms.ToTensor(),
    transforms.Normalize([0.5238, 0.5003, 0.4718], [0.3159, 0.3091, 0.3216])
])

inv_normalize = transforms.Normalize(
    mean=[-0.5238/0.3159, -0.5003/0.3091, -0.4718/0.3216],
    std=[1/0.3159, 1/0.3091, 1/0.3216]
)

denorm = transforms.Compose([
    inv_normalize,
    transforms.functional.to_pil_image
])

def generate_caption_helper(img, beam_search_func):
    img_transformed = valid_tfms(img)
    results = beam_search_func(img_transformed)

    return img_transformed, results

def visualize_attention_mechanism_helper(img, beam_search_func, vocab):
    transformed_img, results = generate_caption_helper(img, beam_search_func)
    visualization = visualize_attention(transformed_img, results[0], results[1], denorm, vocab, return_fig_as_PIL_image=True).convert("RGB")

    imgByteArr = io.BytesIO()
    visualization.save(imgByteArr, format='JPEG')

    return send_file(io.BytesIO(imgByteArr.getvalue()),
                     attachment_filename='return.jpeg',
                     mimetype='image/jpeg')

@app.route('/generate_caption_ms_coco', methods=["POST"])
def generate_caption_ms_coco():
    """Generates a caption for the given image using a model trained on the MS COCO dataset
    ---
    tags:
    - Image caption generator
    parameters:                                                      
    - name: input_image                                              
      in: formData                                                   
      type: file                                                     
      required: true                                                 
    responses:                                                       
        200:                                                         
            description: "image"                                     
    """
    try:
        img = Image.open(request.files.get("input_image"))
        _, results = generate_caption_helper(img, beam_search_ms_coco)
        return vocab_ms_coco.textify(results[0])
    except:
        return "Prediction unsuccessful. Please choose a JPEG file."

@app.route('/visualize_attention_mechanism_ms_coco', methods=["POST"])
def visualize_attention_mechanism_ms_coco():
    """Generates a caption and visualizes the attention mechanism for the given image using a model trained on the MS COCO dataset
    ---
    tags:
    - Image caption generator
    parameters:                                                      
    - name: input_image                                              
      in: formData                                                   
      type: file                                                     
      required: true                                                 
    responses:                                                       
        200:                                                         
            description: "image"                                     
    """
    try:
        img = Image.open(request.files.get("input_image"))
        return visualize_attention_mechanism_helper(img, beam_search_ms_coco, vocab_ms_coco)
    except:
        return "Prediction unsuccessful. Please choose a JPEG file."

@app.route('/generate_caption_conceptual_captions', methods=["POST"])
def generate_caption_conceptual_captions():
    """Generates a caption for the given image using a model trained on the Conceptual Captions dataset
    ---
    tags:
    - Image caption generator
    parameters:                                                      
    - name: input_image                                              
      in: formData                                                   
      type: file                                                     
      required: true                                                 
    responses:                                                       
        200:                                                         
            description: "image"                                     
    """
    try:
        img = Image.open(request.files.get("input_image"))
        _, results = generate_caption_helper(img, beam_search_cc)
        return vocab_cc.textify(results[0])
    except:
        return "Prediction unsuccessful. Please choose a JPEG file."

@app.route('/visualize_attention_mechanism_conceptual_captions', methods=["POST"])
def visualize_attention_mechanism_conceptual_captions():
    """Generates a caption and visualizes the attention mechanism for the given image using a model trained on the Conceptual Captions dataset
    ---
    tags:
    - Image caption generator
    parameters:                                                      
    - name: input_image                                              
      in: formData                                                   
      type: file                                                     
      required: true                                                 
    responses:                                                       
        200:                                                         
            description: "image"                                     
    """
    try:
        img = Image.open(request.files.get("input_image"))
        return visualize_attention_mechanism_helper(img, beam_search_cc, vocab_cc)
    except:
        return "Prediction unsuccessful. Please choose a JPEG file."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7777)
