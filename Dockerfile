FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --upgrade xgboost==2.0.3


CMD ["python", "main.py"]