FROM ubuntu:latest

RUN apt-get update \
    && apt-get install -y python3-pip \
    && pip3 install --upgrade pip

RUN pip3 install numpy pandas matplotlib seaborn plotly sklearn

COPY project.py .

CMD ["python3","-u","project.py"]



