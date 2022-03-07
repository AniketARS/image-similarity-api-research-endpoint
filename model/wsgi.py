import bz2
import os
import re
import pickle

from io import BytesIO
from PIL import Image
from annoy import AnnoyIndex
from sklearn.decomposition import PCA
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import urllib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
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
QID_TO_IDX = {}
PCA256 = None
MODEL = None
IDX_TO_QID = {}
K_MAX = 500  # maximum number of neighbors (even if submitted argument is larger)
model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2"

@app.route('/api/v1/outlinks', methods=['POST'])
def get_neighbors():
    """Returns the Images based on the similarity with the given query image."""
    args = parse_args()
    if 'error' in args:
        return jsonify({'Error': args['error']})
    else:
        image = generate_image(args['data'])
        embeds = generate_embeddings(image)
        results = []
        for idx, dist in zip(*ANNOY_INDEX.get_nns_by_vector(embeds, args['k'], include_distances=True)):
            sim = 1 - dist
            if sim >= args['threshold']:
                results.append({'id': idx, 'score': sim})
            else:
                break
        return jsonify(results)

@app.route('/api/v1/outlinks-interactive', methods=['GET'])
def get_neighbors_interactive():
    """Interactive Wikipedia-based topic modeling endpoint. Takes positive/negative constraints on list."""
    args = parse_args_interactive()
    if 'error' in args:
        return jsonify({'Error': args['error']})
    else:
        search_k = int(args['k'] * ANNOY_INDEX.get_n_trees() / min(len(args['pos']) + len(args['neg']), ANNOY_INDEX.get_n_trees()))
        neg = {}
        for qid in args['neg']:
            qid_idx = QID_TO_IDX[qid]
            for rank, idx in enumerate(ANNOY_INDEX.get_nns_by_item(qid_idx, args['k'], search_k=search_k, include_distances=False)):
                if idx == qid_idx:
                    continue
                qid_nei = IDX_TO_QID[idx]
                if qid_nei not in args['pos'] and qid_nei not in args['skip']:
                    if qid_nei not in neg:
                        neg[qid_nei] = []
                    neg[qid_nei].append(args['k'] - rank)
        # average inverse rank -- missing ranks then default to 0 which makes sense
        # more highly ranked items -> larger numbers
        for qid in neg:
            neg[qid] = int(sum(neg[qid]) / len(args['neg']))

        pos = {}
        avg_min_sim = 0
        for qid in args['pos']:
            qid_idx = QID_TO_IDX[qid]
            # I bump args['k'] because in practice I find that the result sets are slightly too small
            # This is due to filters and the impreciseness of the search trees used by Annoy
            indices, distances = ANNOY_INDEX.get_nns_by_item(qid_idx, args['k']+10, search_k=search_k, include_distances=True)
            avg_min_sim += distances[-1]
            for i, idx in enumerate(indices):
                qid_nei = IDX_TO_QID[idx]
                try:
                    sim = 1 - distances[i + neg.get(qid_nei, 0)]
                except IndexError:
                    sim = 1 - distances[-1]
                if qid_nei not in args['neg'] and qid_nei not in args['skip'] and qid_nei not in args['pos']:
                    if qid_nei not in pos:
                        pos[qid_nei] = []
                    pos[qid_nei].append(sim)
        avg_min_sim = avg_min_sim / len(args['pos'])
        for qid in pos:
            pos[qid] = (sum(pos[qid]) + (avg_min_sim * (len(args['pos']) - len(pos[qid])))) / len(args['pos'])

        results = [{'qid':qid, 'score':score} for qid,score in pos.items() if score >= args['threshold']]
        results = sorted(results, key=lambda x:x['score'], reverse=True)[:args['k']]
        add_article_titles(args['lang'], results)
        return jsonify(results)

def parse_args_interactive():
    args = parse_args()
    if 'error' not in args:
        args['k'] += 1
        args['pos'] = [args['qid']] + [qid for qid in request.args.get('pos','').upper().split('|') if validate_qid_model(qid)]
        args['neg'] = [qid for qid in request.args.get('neg', '').upper().split('|') if validate_qid_model(qid)]
        args['skip'] = [qid for qid in request.args.get('skip', '').upper().split('|') if validate_qid_model(qid)]
    return args

def parse_args():
    # number of neighbors
    k_default = 10  # default number of neighbors
    k_min = 1
    try:
        k = max(min(int(request.get_json()['k']), K_MAX), k_min) + 1
    except Exception:
        k = k_default

    # seed qid
    url = request.get_json()['url']
    try:
        data = urllib.request.urlopen(url).read()
    except Exception:
        return {'error': "Error: URL({}) is not valid".format(url)}

    # threshold for similarity to include
    t_default = 0  # default minimum cosine similarity
    t_max = 1  # maximum cosine similarity threshold (even if submitted argument is larger)
    try:
        threshold = min(float(request.get_json()['threshold']), t_max)
    except Exception:
        threshold = t_default

    # pass arguments
    args = {
        'url': url,
        'k': k,
        'data': data,
        'threshold': threshold,
    }
    return args

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
    embeds = MODEL.predict(np.expand_dims(image, axis=0))
    embeds = embeds.reshape(1, -1)
    embeds = PCA256.transform(embeds).reshape(-1)
    return embeds

def load_model():
    global MODEL
    os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.curdir, 'resources', 'model')
    MODEL = tf.keras.Sequential([hub.KerasLayer(model_url, trainable=False)])
    print("Model Created")
    MODEL.build([None, 224, 224, 3])  # Batch input shape.
    # this will preload the model into Memory
    _ = MODEL.predict(np.ones(shape=(1, 224, 224, 3)))
    print("Model Loaded into Memory")

def load_similarity_index():
    global IDX_TO_QID
    global QID_TO_IDX
    global PCA256
    index_fp = os.path.join(os.curdir, 'resources', 'embeddings.ann')
    pca256_fp = os.path.join(os.curdir, 'resources', 'pca256.pkl')
    qidmap_fp = os.path.join(__dir__, 'resources/qid_to_idx.pickle')
    if os.path.exists(index_fp):
        print("Using pre-built ANNOY index")
        ANNOY_INDEX.load(index_fp)
        with open(pca256_fp, 'rb') as fin:
            PCA256 = pickle.load(fin)
        # with open(qidmap_fp, 'rb') as fin:
        #     QID_TO_IDX = pickle.load(fin)
    else:
        print("Builing ANNOY index")
        ANNOY_INDEX.on_disk_build(index_fp)
        with bz2.open(os.path.join(__dir__, 'resources/embeddings.tsv.bz2'), 'rt') as fin:
            for idx, line in enumerate(fin, start=0):
                line = line.strip().split('\t')
                qid = line[0]
                QID_TO_IDX[qid] = idx
                emb = [float(d) for d in line[1].split()]
                ANNOY_INDEX.add_item(idx, emb)
                if idx + 1 % 1000000 == 0:
                    print("{0} embeddings loaded.".format(idx))
        print("Building AnnoyIndex with 25 trees.")
        ANNOY_INDEX.build(100)
        with open(qidmap_fp, 'wb') as fout:
            pickle.dump(QID_TO_IDX, fout)
    IDX_TO_QID = {v: k for k, v in QID_TO_IDX.items()}
    print("{0} QIDs in nearset neighbor index.".format(ANNOY_INDEX.get_n_items()))

application = app
# FOR TESTING LOCALLY - SHOULD BE REMOVED
tf.config.set_visible_devices([], 'GPU')
load_similarity_index()
load_model()

if __name__ == '__main__':
    application.run()
