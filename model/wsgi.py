import json
import os
import pickle
import sys

from io import BytesIO

import requests
from PIL import Image
from annoy import AnnoyIndex
from sklearn.decomposition import PCA
import numpy as np
import urllib
from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib
import yaml
import opennsfw2 as n2
from Detector import Detector

app = Flask(__name__)

__dir__ = os.path.dirname(__file__)
sys.path.append(__dir__)

# load in app user-agent or any other app config
app.config.update(
    yaml.safe_load(open(os.path.join(__dir__, 'flask_config.yaml'))))

# Enable CORS for API endpoints
cors = CORS(app, resources={r'/api/*': {'origins': '*'}})

# efficientNetB3V2 model with PCA(256) for making predictions
ANNOY_INDEX = AnnoyIndex(256, 'angular')
PCA256 = PCA(n_components=256)
IDX_TO_URL = {}
K_MAX = 500  # maximum number of neighbors (even if submitted argument is larger)
FACE_DETECTOR = None

@app.route('/api/v1/similar-images', methods=['GET'])
def get_neighbors_by_name():
    """Returns the Images based on the similarity with the given query image."""
    args = parse_args(src='commons')
    return get_neighbors(args)

@app.route('/api/v1/similar-images-url', methods=['POST'])
def get_neighbors_by_url():
    """Returns the Images based on the similarity with the given query url."""
    args = parse_args(src='url')
    return get_neighbors(args)

@app.route('/api/v1/similar-images-bytes', methods=['POST'])
def get_neighbors_by_bytes():
    """Returns the Images based on the similarity with the given query bytes of image."""
    args = parse_args(src='bytes')
    return get_neighbors(args)

def get_neighbors(args):
    if 'error' in args:
        return jsonify({'Error': args['error']})
    else:
        image = generate_image(args['data'])
        if image is None:
            return jsonify({
                "warning": "This input image seems to contain content that is not suitable for work."
            })
        if if_faces(image*255.0):
            return jsonify({
                "warning": "This input image seems to include faces. This model is not designed to detect face "
                           "similarity. "
            })
        print("Generated Image")
        embeds = generate_embeddings(image)
        print("Generated Embeddings")
        results = []
        for idx, dist in zip(*ANNOY_INDEX.get_nns_by_vector(embeds, args['k'], include_distances=True)):
            sim = 1 - dist
            if sim >= args['threshold']:
                results.append({'url': IDX_TO_URL[idx], 'score': sim})
            else:
                break
        print("Returning Results")
        return jsonify(results)


def parse_args(src='commons'):
    # number of neighbors
    k_default = 10  # default number of neighbors
    k_min = 1
    try:
        k = max(min(int(request.args.get('k')), K_MAX), k_min) + 1
    except Exception:
        k = k_default

    # seed qid
    image_name, url, data = None, None, None
    if src == 'commons':
        image_name = request.args.get('image_name')
        url = generate_url(image_name)
    elif src == 'url':
        url = request.get_json()['url']
    print(url)
    try:
        if src != 'bytes':
            data = urllib.request.urlopen(url).read()
        else:
            data = request.files['image']
    except Exception:
        return {'error': "Error: URL({}) is not valid".format(url)}

    # threshold for similarity to include
    t_default = 0  # default minimum cosine similarity
    t_max = 1  # maximum cosine similarity threshold (even if submitted argument is larger)
    try:
        threshold = min(float(request.args.get('threshold')), t_max)
    except Exception:
        threshold = t_default

    # pass arguments
    args = {
        'url': url,
        'k': k,
        'data': data,
        'threshold': threshold,
    }
    print("Parsed args successfully..")
    return args

def generate_url(image_name):
    m = hashlib.md5()
    m.update(image_name.encode('utf-8'))
    md5sum = m.hexdigest()
    quoted_name = urllib.parse.quote(image_name)
    url = "https://upload.wikimedia.org/wikipedia/commons/{}/{}/{}".format(md5sum[:1], md5sum[:2], quoted_name)
    return url

def generate_image(byte_string):
    image = None
    try:
        image = Image.open(BytesIO(byte_string))
    except Exception:
        image = Image.open(byte_string)
    if nsfw_check(image):
        return None
    image = image.resize((224, 224), Image.BILINEAR)
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.repeat(image[..., np.newaxis], 3, -1)
    elif image.shape[-1] == 2:
        image = np.dstack((image, np.mean(image, axis=(2,))))
    else:
        image = image[:, :, :3]
    image = image / 255.0
    return image

def generate_embeddings(image):
    headers = {"content-type": "application/json"}
    payload = json.dumps({"signature_name": "serving_default", "instances": [image.tolist()]})
    result = requests.post('http://localhost:8501/v1/models/efficient_net_b3_v2:predict',
                           data=payload, headers=headers)
    embeds = json.loads(result.content.decode('utf-8'))['predictions']
    print("Got Embeddings..")
    embeds = PCA256.transform(embeds).reshape(-1)
    print("Applied PCA..")
    return embeds

def nsfw_check(image):
    image = n2.preprocess_image(image, n2.Preprocessing.SIMPLE)
    headers = {"content-type": "application/json"}
    payload = json.dumps({"signature_name": "serving_default", "instances": [image.tolist()]})
    result = requests.post('http://localhost:8502/v1/models/open_nsfw:predict',
                           data=payload, headers=headers)
    preds = json.loads(result.content.decode('utf-8'))['predictions'][0]
    return preds[-1] > 0.8

def if_faces(image):
    def calculate_area_1(face, w, h):
        box = face['face']
        area = box[-1] * box[-2]
        aoi = (area / (w*h))*100
        return aoi

    def calculate_area_max(faces, w, h):
        s_aoi, m_aoi = 0.0, 0.0
        for box in faces:
            t_aoi = calculate_area_1(box, w, h)
            m_aoi = m_aoi if m_aoi > t_aoi else t_aoi
            s_aoi += t_aoi
        return m_aoi, s_aoi
    faces = FACE_DETECTOR.detect(image)
    if len(faces) == 1:
        face_aoi = calculate_area_1(faces[0], 224.0, 224.0)
        return True if face_aoi >= 2.5 else False
    elif len(faces) > 1:
        max_aoi, sum_aoi = calculate_area_max(faces, 224.0, 224.0)
        return True if max_aoi > 3.0 or sum_aoi > 15.0 else False
    return False

def load_similarity_index():
    global IDX_TO_URL
    global PCA256, FACE_DETECTOR
    index_fp = os.path.join('/', 'extrastorage', 'data', 'tree.cnn')
    pca256_fp = os.path.join(__dir__, 'resources', 'pca256.pkl')
    idxmap_fp = os.path.join(__dir__, 'resources', 'idx2url.pkl')

    print("Using pre-built ANNOY index")
    ANNOY_INDEX.load(index_fp)
    with open(pca256_fp, 'rb') as fin:
        PCA256 = pickle.load(fin)
    print("Loaded PCA")
    with open(idxmap_fp, 'rb') as fin:
        IDX_TO_URL = pickle.load(fin)
    print("Loaded IDX_TO_URL")
    FACE_DETECTOR = Detector()
    print("Loaded Face Detector")

    print("{0} IDs in nearset neighbor index.".format(ANNOY_INDEX.get_n_items()))

application = app
load_similarity_index()

if __name__ == '__main__':
    application.run()
