from tensorflow/tensorflow:latest-gpu-py3

RUN add-apt-repository ppa:deadsnakes/ppa &&  apt-get update && apt-get install -y python3.6 python3.6-dev && apt install -y libsm6 libxext6 libxrender1 libfontconfig1
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6     
RUN python3.6 -m pip install ipykernel
RUN python3.6 -m ipykernel install
RUN python3.6 -m pip install tensorflow-gpu keras sklearn  h5py sklearn Pillow seaborn scikit-learn keras scikit-image opencv-python
RUN rm /usr/bin/python
RUN ln /usr/bin/python3.6 /usr/bin/python
WORKDIR /notebooks
EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/notebooks --ip 0.0.0.0 --no-browser --allow-root"]
