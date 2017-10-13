import os

from flask import Flask
from keras.models import load_model

app = Flask(__name__)
model = load_model(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '../../my_model.h5'))
setattr(app, 'basicNN', model)
from app import views