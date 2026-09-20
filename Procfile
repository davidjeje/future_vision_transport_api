web: MODEL_MODE=${MODEL_MODE:-huggingface} gunicorn --chdir api app.main:app --workers 1 --worker-class uvicorn_worker.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
