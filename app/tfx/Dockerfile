FROM ubuntu:xenial

# install tensorflow-model-server
RUN apt update && \
apt-get install -y curl && \
echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - && \
apt update && \
apt-get install tensorflow-model-server

# copy saved model format over to container
COPY ./ModelSavedFormat /models/nli-bert

# run tensorflow server when container is created.
CMD tensorflow_model_server --port=8500 --rest_api_port=${PORT} --model_base_path=/models/nli-bert --model_name=nli-bert