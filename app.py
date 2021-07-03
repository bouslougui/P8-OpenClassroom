import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template

from werkzeug.utils import secure_filename
import numpy as np
import glob 


from keras.models import load_model
from keras.preprocessing import image
import cv2


from azure.common.credentials import ServicePrincipalCredentials
from azureml.core.model import Model
from tensorflow import keras   

from azureml.core.authentication import ServicePrincipalAuthentication

from azureml.core import Workspace,Datastore,Dataset

# Retrieve the IDs and secret to use with ServicePrincipalCredentials
tenant_id = "e6311692-39bf-4a2b-95d1-2636e4e409c7"
client_id = "80b5eb4f-f73f-45ee-8e68-47de95d57340"
client_secret = "DPG~sW85RX8Loh8L4-zc-RCH8GyJq-x5t3"


svc_pr = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=client_id,
    service_principal_password=client_secret)

ws = Workspace(
    subscription_id="9c6e1d8a-238d-4ae2-8b6e-7cb9dbae6faf",
    resource_group="OCR",
    workspace_name="P8_OCR",
    auth=svc_pr
    )

model = Model(ws, 'ocr_p8_voiture_v1')

model.download(target_dir=os.getcwd(), exist_ok=True)

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
    
def model_predict(img, model):
    
    # Preprocessing the image
    img = image.img_to_array(img)/256
    x = np.expand_dims(img, axis=0)
    #x = preprocess_input(x, mode='caffe')
    z = model.predict(x)
    z = np.squeeze(z)
    z = z.reshape(256, 256, 8)
  
    return z

MODEL_PATH = 'model.h5'
# Load your trained model
model = load_model(MODEL_PATH,custom_objects={'dice_coeff':dice_coeff})

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
        z = model_predict(image_in, model)
        
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
