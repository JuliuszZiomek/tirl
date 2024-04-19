FROM python:3.9

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN apt-get update
RUN apt-get install -y python3-sklearn python3-sklearn-lib python3-scipy gfortran libopenblas-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev libportmidi-dev libjpeg-dev
RUN git clone https://github.com/JuliuszZiomek/tirl
WORKDIR trajectory-information-rl/
ADD requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install -U tensorflow
RUN python3 -m pip install tf-keras
RUN apt-get install -y tmux
ENTRYPOINT ["/bin/bash"]
