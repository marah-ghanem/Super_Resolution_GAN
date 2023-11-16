import os
import torch
import numpy as np
from PIL import Image
from RealESRGAN import RealESRGAN
from keras.preprocessing.image import load_img
from flask import Flask, request, render_template
from flask import Flask, request, render_template, send_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Selected device: {device}")

model = RealESRGAN(device, scale=2)
model.load_weights('weights/RealESRGAN_x2.pth', download=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['Results']='results'

# Load the pre-trained model
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    uploaded_file = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return render_template('draft.html')

        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            uploaded_file = file.filename

            img = load_img(filename)
            sr_image = model.predict(img)

            filename = os.path.join(app.config['Results'], file.filename)
            sr_image.save(filename)

    return render_template('index.html', uploaded_file=uploaded_file)

@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['Results'], filename), as_attachment=True)

@app.route('/index')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(port=3000,debug=True)

