import json
import os
import pickle

from io import BytesIO

import requests
from PIL import Image
from annoy import AnnoyIndex
from sklearn.decomposition import PCA
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import urllib
from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib
import yaml

app = Flask(__name__)

__dir__ = os.path.dirname(__file__)

# load in app user-agent or any other app config
app.config.update(
    yaml.safe_load(open(os.path.join(__dir__, 'flask_config.yaml'))))

# Enable CORS for API endpoints
cors = CORS(app, resources={r'/api/*': {'origins': '*'}})

# efficientNetB3V2 model with PCA(256) for making predictions
ANNOY_INDEX = AnnoyIndex(256, 'angular')
PCA256 = PCA(n_components=256)
MODEL = None
IDX_TO_URL = {}
K_MAX = 500  # maximum number of neighbors (even if submitted argument is larger)
model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2"

@app.route('/api/v1/similar-images', methods=['GET'])
def get_neighbors():
    """Returns the Images based on the similarity with the given query image."""
    args = parse_args()
    if 'error' in args:
        return jsonify({'Error': args['error']})
    else:
        image = generate_image(args['data'])
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

def parse_args():
    # number of neighbors
    k_default = 10  # default number of neighbors
    k_min = 1
    try:
        k = max(min(int(request.args.get('k')), K_MAX), k_min) + 1
    except Exception:
        k = k_default

    # seed qid
    image_name = request.args.get('image_name')
    url = generate_url(image_name)
    print(url)
    try:
        data = urllib.request.urlopen(url).read()
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
    image = Image.open(BytesIO(byte_string))
    image = image.resize((224, 224), Image.ANTIALIAS)
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

def load_similarity_index():
    global IDX_TO_URL
    global PCA256
    # TODO: change the path for server
    index_fp = os.path.join(os.curdir, 'resources', 'embeddings.ann')
    pca256_fp = os.path.join(os.curdir, 'resources', 'pca256.pkl')
    idxmap_fp = os.path.join(os.curdir, 'resources', 'id2url.pkl')

    print("Using pre-built ANNOY index")
    ANNOY_INDEX.load(index_fp)
    with open(pca256_fp, 'rb') as fin:
        PCA256 = pickle.load(fin)
    print("Loaded PCA")
    with open(idxmap_fp, 'rb') as fin:
        IDX_TO_URL = pickle.load(fin)
    print("Loaded IDX_TO_URL")

    print("{0} IDs in nearset neighbor index.".format(ANNOY_INDEX.get_n_items()))

application = app
# FOR DISABLING THE GPU USE (IN CASE ANY)
tf.config.set_visible_devices([], 'GPU')
load_similarity_index()

if __name__ == '__main__':
    application.run()
