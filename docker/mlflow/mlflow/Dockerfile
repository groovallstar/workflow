FROM continuumio/miniconda3

# timezone
RUN export DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime
RUN apt-get update && apt-get install -y tzdata
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN pip install mlflow==1.25.1
RUN pip install psycopg2-binary pymysql
#RUN cd /home && git clone https://github.com/mlflow/mlflow-example.git
