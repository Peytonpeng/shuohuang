FROM python:3.12

WORKDIR /app

ADD . /app

#RUN pip install --no-index --find-links=./offline_packages -r ./requirements.txt
RUN pip3 install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

EXPOSE 5001

ENV NAME World

CMD ["python3", "app.py"]
