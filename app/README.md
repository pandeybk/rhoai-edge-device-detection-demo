## Setup virtualenv
```
python3 -m venv venv
source ./venv/bin/activate
```

## Install django
```
pip install django
```

##  Create and Configure Your Django Project
```
django-admin startproject image_predictor
cd image_predictor
python manage.py startapp prediction_app
```

## Run App
```
python manage.py runserver
```

## Podman Build and run
Make sure to create .env file with following content
```
MODEL_SERVER_URL=<model-server-url>
TOKEN=<your-token>
DJANGO_ALLOWED_HOSTS=".apps.demo.ai.ocp.run,localhost,127.0.0.1"
```

Now build and run the code.
```
podman build -t localhost/image_predictor:latest .
podman run --env-file=./.env -p 8090:8000 localhost/image_predictor:latest
```