FROM python:3.11.13-slim

WORKDIR /app 

COPY Pipfile.lock Pipfile scoring.py ./

RUN pip install pipenv && pipenv install --system --deploy --ignore-pipfile

ENTRYPOINT ["python", "scoring.py"]