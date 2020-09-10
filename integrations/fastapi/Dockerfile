FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY ./requirements.txt /app/
RUN pip install -r requirements.txt
# TODO: fix this once we have models
RUN python -c "import spacy;spacy.blank('en').to_disk('en_core_web_sm')"
COPY ./scripts /app/app
