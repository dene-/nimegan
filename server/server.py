from io import BytesIO

from flask.helpers import send_file
from flask_cors import CORS, cross_origin
from numpy.core.fromnumeric import size
import torch
import torchvision
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from flask import Flask, flash, request, redirect, Response
from werkzeug.utils import secure_filename
import mimetypes
import numpy as np
import random
import os

from Models import *

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask("nimegan")
CORS(app)

import torch.quantization

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2", device=device).eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def face2paint(
        model: torch.nn.Module,
        img: Image.Image
    ) -> Image.Image:

        with torch.no_grad():
            input = to_tensor(img).unsqueeze(0) * 2 - 1
            output = model(input.to(device)).cpu()[0]
            output = (output * 0.5 + 0.5).clip(0, 1)

        return to_pil_image(output)

def waifu2x(
        img: Image.Image
    ) -> Image.Image:

        model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
                        
        model_cran_v2 = network_to_half(model_cran_v2)

        model_cran_v2.load_state_dict(torch.load("./CARN_model_checkpoint.pt", 'cpu'))
        # if use GPU, then comment out the next line so it can use fp16. 
        model_cran_v2 = model_cran_v2.float() 

        # origin
        with torch.no_grad():
                input = to_tensor(img).unsqueeze(0)
                output = model_cran_v2(input.to(device)).cpu()[0]
                output = output.clip(0, 1)

        return to_pil_image(output)

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def process_image():
    if request.method == 'POST':

        # Check if the post request has the file part
        if 'file' not in request.files:
            return {'error': 'No file part in the request'}, 400

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '' or file is None:
            return {'error': 'No selected file'}, 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            mimetype = mimetypes.guess_type(filename)[0]

            img = Image.open(BytesIO(file.read())).convert("RGB")

            #img.save(os.path.join('./uploads', str(random.randint(0, 20000)) + '_' + filename))

            out = face2paint(model, img)

            buffer_out = BytesIO()
            format = mimetype.split("/")[1]

            out.save(buffer_out, format=format, quality=100)

            buffer_out.seek(0)

            return send_file(buffer_out, mimetype=mimetype)
        else:
            return {'error': 'File type not allowed'}, 400

if __name__ == '__main__':
    app.secret_key = 'super secret key'

    app.run(host='0.0.0.0', port=1024, debug=True, use_reloader=True)
