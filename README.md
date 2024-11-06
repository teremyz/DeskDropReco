# Deskdrop - Internal Communications Platform

Deskdrop is an internal communications platform designed to allow employees of companies to share relevant articles with their peers. The system provides recommendations to users based on their interactions with the platform, such as viewing or sharing articles.

## Getting Started - API

Follow the instructions below to build and run the Deskdrop container locally.

### Prerequisites

Make sure you have Docker installed on your machine. You can follow the official Docker installation guide: https://docs.docker.com/get-docker/

### Build the Container

To build the Docker container from the root of the project, run the following command:

```bash
docker build -t deskdrop .
```

Alternatively, you can pull the pre-built image from Docker Hub:
```bash
docker pull teremyz21/deskdrop
```
### Run the Container
Once the container image is built or pulled, you can run the container with the following command:

Alternatively, you can pull the pre-built image from Docker Hub:
```bash
docker run -d --name deskdrop_container -p 8000:8000 deskdrop
```
## Test the API
Using curl:
```bash
curl -X 'GET' \
  'http://localhost:8000/predict?dateTime=2017-02-17%2016%3A12%3A27&eventType=VIEW&contentId=-5781461435447152359&personId=-9223121837663643404&userRegion=SP&userCountry=BR&lastContentId=-6728844082024523776&lastEventType=VIEW&top_n=10' \
  -H 'accept: application/json'
```
Or open this URL in your browser:
```bash
http://localhost:8000/predict?dateTime=2017-02-17%2016%3A12%3A27&eventType=VIEW&contentId=-5781461435447152359&personId=-9223121837663643404&userRegion=SP&userCountry=BR&lastContentId=-6728844082024523776&lastEventType=VIEW&top_n=10

```

You can access the interactive API documentation at:
```bash
http://localhost:8000/docs
```
This will give you a user-friendly interface to explore all the available endpoints.


## Getting Started - Training pipelines

Clone the repository. Install poetry. Run the following command from the project root folder
```bash
poetry install
```
Run popularity pipeline:
```bash
poetry run popularity_training_pipeline.py --config config.yaml
```
Run matrix factorization model training pipeline:
```bash
poetry run mf_training_pipeline.py --config config.yaml
```
Run XGB training pipeline:
```bash
poetry run popularity_training_pipeline.py --config config.yaml
```
