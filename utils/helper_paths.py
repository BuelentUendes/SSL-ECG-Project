# Helper file for used paths
import os
from os.path import abspath

# Define the common paths
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = abspath(os.path.join(FILE_PATH, './../'))
DATA_PATH = abspath(os.path.join(BASE_PATH, './', "data"))
SAVED_MODELS_PATH = abspath(os.path.join(BASE_PATH, 'saved_models'))
