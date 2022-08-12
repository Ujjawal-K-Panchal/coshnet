ARG VERSION=3.8-cuda11.3-runtime

FROM wallies/python-cuda:${VERSION}

COPY ./code /app/root/code/
COPY ./libs /app/root/libs/
COPY ./requirements.txt /app/root/
COPY ./setup.py /app/root/
COPY ./setup_docker.py /app/root/

WORKDIR /app/root/
RUN python -m venv venv4coshnet
RUN venv4coshnet/bin/python -m pip install --upgrade pip
RUN venv4coshnet/bin/python -m pip install wheel
RUN venv4coshnet/bin/python -m pip install -r requirements.txt
RUN ls

WORKDIR ./code/
RUN ls
RUN ../venv4coshnet/bin/python test_fashion.py
RUN ../venv4coshnet/bin/python test_resnet.py

CMD ["echo", "ran successfully!"]