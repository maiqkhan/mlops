FROM python:3.9.19-slim

RUN pip install -U pip

WORKDIR /app
COPY ["predict.py", "lin_reg.bin", "requirements.txt", "./"]

RUN pip install -r requirements.txt

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]