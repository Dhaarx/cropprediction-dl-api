FROM python:3.13

WORKDIR /app

# Install git-lfs
RUN apt-get update && apt-get install -y git-lfs && git lfs install

COPY . .

RUN git lfs pull || true
RUN pip install --upgrade pip && pip install -r requirements.txt


EXPOSE 10000

CMD ["gunicorn","--bind","0.0.0.0:10000","app:app"]
