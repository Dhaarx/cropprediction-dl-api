FROM python:3.13-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt


EXPOSE 10000

CMD ["gunicorn","--bind","0.0.0.0:10000","app:app"]
