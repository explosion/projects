FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY ./requirements.txt /app/
RUN pip install spacy-nightly --pre
RUN pip install -r requirements.txt
RUN ./scripts/download_models.sh
COPY ./scripts /app/app
