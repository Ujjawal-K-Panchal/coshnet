ARG VERSION=3.8-cuda11.3-runtime

FROM wallies/python-cuda:${VERSION}

COPY ./code /app/coshnet/code/
COPY ./libs /app/coshnet/libs/
COPY ./requirements.txt /app/coshnet/
COPY ./setup.py /app/coshnet/
COPY ./setup_docker.py /app/coshnet/

WORKDIR /app/coshnet/
RUN python -m venv venv4coshnet
RUN venv4coshnet/bin/python -m pip install --upgrade pip
RUN venv4coshnet/bin/python -m pip install wheel
RUN venv4coshnet/bin/python -m pip install -r requirements.txt
RUN ls

WORKDIR ./code/
RUN ls
RUN ../venv4coshnet/bin/python test_fashion.py --epoch 10 --validate 1 --snapshot coshnet-fashion 
RUN ../venv4coshnet/bin/python test_resnet.py --epoch 10

CMD ../venv4coshnet/bin/python test_loadmodels.py --snapshot coshnet-fashion