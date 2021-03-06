#!/usr/bin/env bash
# setup Cloud VPS instance with initial server etc.

# these can be changed but most other variables should be left alone
APP_LBL='api-endpoint'  # descriptive label for endpoint-related directories
REPO_LBL='similaritymodel'  # directory where repo code will go
GIT_CLONE_HTTPS='https://github.com/AniketARS/image-similarity-api-research-endpoint'  # for `git clone`
MODEL_WGET='https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2?tf-hub-format=compressed'
PCA256_WGET='https://drive.google.com/uc?export=download&id=1j-oOy-SdkUQY6xKYqIy0xXTrpRqx_Nyi'
OPENNSFW_WGET='https://drive.google.com/uc?export=download&id=1rnyJFN8nG4oaPumysl9h9cB00CUn5LqQ'
MTCNN_WGET='https://drive.google.com/uc?export=download&id=1-I2HhPverIjcP-vKbPVJmrTZaGfdkDEN'

ETC_PATH="/etc/${APP_LBL}"  # app config info, scripts, ML models, etc.
SRV_PATH="/srv/${APP_LBL}"  # application resources for serving endpoint
TMP_PATH="/tmp/${APP_LBL}"  # store temporary files created as part of setting up app (cleared with every update)
LOG_PATH="/var/log/uwsgi"  # application log data
LIB_PATH="/var/lib/${APP_LBL}"  # where virtualenv will sit

echo "Updating the system..."
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
apt-get update
apt-get install -y build-essential  # gcc (c++ compiler) necessary for fasttext
apt-get install -y nginx  # handles incoming requests, load balances, and passes to uWSGI to be fulfilled
apt-get install -y python3-pip  # install dependencies
apt-get install -y python3-wheel  # make sure dependencies install correctly even when missing wheels
apt-get install -y python3-venv  # for building virtualenv
apt-get install -y python3-dev  # necessary for fasttext
apt-get install -y uwsgi
apt-get install -y uwsgi-plugin-python3
apt-get install -y tensorflow-model-server
apt-get install -y python3-opencv
apt-get install -y libgl1
# potentially add: apt-get install -y git python3 libpython3.7 python3-setuptools

echo "Setting up paths..."
rm -rf ${TMP_PATH}
mkdir -p ${TMP_PATH}
mkdir -p ${SRV_PATH}/sock
mkdir -p ${ETC_PATH}
# Image Features Generator
mkdir -p ${ETC_PATH}/resources
mkdir -p ${ETC_PATH}/resources/efficient_net_b3_v2
# NSFW Classifier
mkdir -p ${ETC_PATH}/resources/efficient_net_b3_v2/1
mkdir -p ${ETC_PATH}/resources/open_nsfw
# MTCNN Face Detector
mkdir -p ${ETC_PATH}/resources/open_nsfw/1

mkdir -p ${LOG_PATH}
mkdir -p ${LIB_PATH}

echo "Setting up virtualenv..."
python3 -m venv ${LIB_PATH}/p3env
source ${LIB_PATH}/p3env/bin/activate

echo "Cloning repositories..."
# NOTE: a more stable install would involve building wheels on an identical instance and then the following:
# NOTE: see (https://gerrit.wikimedia.org/g/research/recommendation-api/wheels/+/refs/heads/master) for an example.
# git clone https://gerrit.wikimedia.org/r/research/recommendation-api/wheels ${TMP_PATH}/wheels
# echo "Making wheel files..."
# cd ${TMP_PATH}/wheels
# rm -rf wheels/*.whl
# make
# git clone ${GIT_CLONE_HTTPS} ${TMP_PATH}/${REPO_LBL}
# echo "Installing repositories..."
# pip3 install --no-deps ${TMP_PATH}/wheels/wheels/*.whl
# pip3 install --no-deps ${TMP_PATH}/recommendation-api

# The simpler process is to just install dependencies per a requirements.txt file
# With updates, however, the packages could change, leading to unexpected behavior or errors
git clone --branch image-similarity ${GIT_CLONE_HTTPS} ${TMP_PATH}/${REPO_LBL}

echo "Installing repositories..."
pip install wheel
pip install Cython==0.29.28
pip install numpy==1.21.5
pip install -r ${TMP_PATH}/${REPO_LBL}/requirements.txt

# If UI included, consider the following for managing JS dependencies:
# echo "Installing front-end resources..."
# mkdir -p ${SRV_PATH}/resources
# cd ${TMP_PATH}
# npm install bower
# cd ${SRV_PATH}/resources
# ${TMP_PATH}/node_modules/bower/bin/bower install --allow-root ${TMP_PATH}/recommendation-api/recommendation/web/static/bower.json

echo "Downloading model and index, hang on..."

wget -O model.tar.gz ${MODEL_WGET}
wget -O open_nsfw.tar.xz ${OPENNSFW_WGET}
wget -O mtcnn.tar.xz ${MTCNN_WGET}
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VDxVg-RSr2BuRO8dsFB3mCaoPMh2VHqf' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VDxVg-RSr2BuRO8dsFB3mCaoPMh2VHqf" -O idx2url.pkl && rm -rf /tmp/cookies.txt
wget -O pca256.pkl ${PCA256_WGET}

tar -xzf model.tar.gz -C ${ETC_PATH}/resources/efficient_net_b3_v2/1
rm model.tar.gz

tar -xf open_nsfw.tar.xz -C ${ETC_PATH}/resources/
rm open_nsfw.tar.xz

tar -xf mtcnn.tar.xz -C ${ETC_PATH}/resources/
rm mtcnn.tar.xz

mv idx2url.pkl ${ETC_PATH}/resources
mv pca256.pkl ${ETC_PATH}/resources

echo "Setting up ownership..."  # makes www-data (how nginx is run) owner + group for all data etc.
chown -R www-data:www-data ${ETC_PATH}
chown -R www-data:www-data ${SRV_PATH}
chown -R www-data:www-data ${LOG_PATH}
chown -R www-data:www-data ${LIB_PATH}

echo "Copying configuration files..."
cp ${TMP_PATH}/${REPO_LBL}/model/config/* ${ETC_PATH}
# TODO: fix this to be more elegant (one directory or not necessary because run as package)
cp ${TMP_PATH}/${REPO_LBL}/model/wsgi.py ${ETC_PATH}
cp ${TMP_PATH}/${REPO_LBL}/model/__init__.py ${ETC_PATH}
cp ${TMP_PATH}/${REPO_LBL}/model/Detector.py ${ETC_PATH}
cp ${TMP_PATH}/${REPO_LBL}/model/flask_config.yaml ${ETC_PATH}
cp ${ETC_PATH}/model.nginx /etc/nginx/sites-available/model
if [[ -f "/etc/nginx/sites-enabled/model" ]]; then
    unlink /etc/nginx/sites-enabled/model
fi
ln -s /etc/nginx/sites-available/model /etc/nginx/sites-enabled/
cp ${ETC_PATH}/model.service /etc/systemd/system/
cp ${ETC_PATH}/tensorflow.service /etc/systemd/system/
cp ${ETC_PATH}/opennsfw.service /etc/systemd/system/
cp ${ETC_PATH}/onet.service /etc/systemd/system/
cp ${ETC_PATH}/pnet.service /etc/systemd/system/
cp ${ETC_PATH}/rnet.service /etc/systemd/system/

echo "Enabling and starting services..."
systemctl enable model.service  # uwsgi starts when server starts up
systemctl enable tensorflow.service # tensorflow serving server starts
systemctl enable opennsfw.service # nsfw serving server starts
systemctl enable onet.service # mtcnn_onet serving server starts
systemctl enable pnet.service # mtcnn_pnet serving server starts
systemctl enable rnet.service # mtcnn_rnet serving server starts
systemctl daemon-reload  # refresh state

systemctl restart model.service  # start up uwsgi
systemctl restart tensorflow.service # start up tensorflow serving
systemctl restart opennsfw.service # start up nsfw serving
systemctl restart onet.service # start up mtcnn_onet serving
systemctl restart pnet.service # start up mtcnn_pnet serving
systemctl restart rnet.service # start up mtcnn_rnet serving
systemctl restart nginx  # start up nginx