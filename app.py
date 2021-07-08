import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import requests
import json

from werkzeug.utils import secure_filename
import numpy as np
import glob 

  



ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (150, 150)
UPLOAD_FOLDER = 'uploads'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join('static/image/')
MASK_PATH = os.path.join('static/mask/')
image_list = glob.glob(IMAGE_PATH+'*.png')
mask_list = glob.glob(MASK_PATH+'*.png') 


app = Flask(__name__, template_folder='Templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def parse_code(l):
    '''Function to parse lines in a text file, returns separated elements (label codes and names in this case)
    '''
    if len(l.strip().split("\t")) == 2:
        a, b = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), b
    else:
        a, b, c = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), c

label_codes, label_names = zip(*[parse_code(l) for l in open("label.txt")])
label_codes, label_names = list(label_codes), list(label_names)
label_codes[:5], label_names[:5]        

code2id = {v:k for k,v in enumerate(label_codes)}
id2code = {k:v for k,v in enumerate(label_codes)}


def onehot_to_rgb(onehot, colormap = id2code):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)
    
    
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score
 
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def total_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss
    
def model_predict(img):
    
    # Preprocessing the image
    img = image.img_to_array(img)/255
    x = np.expand_dims(img, axis=0)
    test = json.dumps({"data": x.tolist()})
    test = bytes(test, encoding='utf8')

    headers = {'Content-Type':'application/json'}
    api_key = 'duSKvtbltK22bhiVmL3jlThs7bcbOWh9'
    headers = {'Content-Type':'application/json',  'Authorization':('Bearer '+ api_key)} 

    resp = requests.post('http://20.62.213.21:80/api/v1/service/ocr-p8-srv4966/score', test, headers=headers)

    #x = preprocess_input(x, mode='caffe')
    data = json.loads(resp.content)
    z = json.loads(data)['result']
    z = np.squeeze(z)
    z = z.reshape(256, 256, 8)
  
    return z

#MODEL_PATH = 'model.h5'
# Load your trained model
#model = load_model(MODEL_PATH,custom_objects={'dice_coeff':dice_coeff})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS




@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        id_image = int(request.form['id_image'])
        if id_image >= len(image_list):
            return render_template('home.html', label='Out of range', imagesource='file://null')
        file_path = os.path.join(image_list[id_image])
        mask_file_path =os.path.join(mask_list[id_image])
        image_in = cv2.imread(os.path.join(image_list[id_image]))
        z = model_predict(image_in)
        
        image_out = onehot_to_rgb(z, id2code)
        filename = "predict_mask"+request.form['id_image']+".png"
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), image_out)
        predict_mask_file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        output = {'Negative:': 0, 'Positive': 1}
    return render_template("home.html", label=output, imagesource=file_path,masksource=mask_file_path,predict_masksource=predict_mask_file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(threaded=True,debug=True)
