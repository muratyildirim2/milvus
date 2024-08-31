FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV FLASK_ENV=production

VOLUME /app/milvus_data

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "main:app"]
