# Usage
# docker build -t ganjireb/flask_playground /Users/rebecca/Desktop/ZHAW/Frühlingssemester2024/test/flask_playground
# docker run -p 9001:5000 -d ganjireb/flask_playground:latest

FROM python:3.10.11

# Copy Files
WORKDIR /usr/src/app
COPY backend/service.py backend/service.py
COPY templates/index.html templates/index.html

# Install
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Docker Run Command
EXPOSE 80
ENV FLASK_APP=usr/src/app/backend/service.py
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0", "--port=80"]