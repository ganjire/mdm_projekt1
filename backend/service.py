# python -m flask --debug --app service run (works also in PowerShell)

import os
import pickle
from pathlib import Path
import pandas as pd
from azure.storage.blob import BlobServiceClient
from flask import Flask, jsonify, render_template, request, send_file
#from flask_cors import CORS

# init app, load model from storage
print("*** Init and load model ***")
if 'AZURE_STORAGE_CONNECTION_STRING' in os.environ:
    azureStorageConnectionString = os.environ['AZURE_STORAGE_CONNECTION_STRING']
    blob_service_client = BlobServiceClient.from_connection_string(azureStorageConnectionString)

    print("fetching blob containers...")
    containers = blob_service_client.list_containers(include_metadata=True)
    for container in containers:
        existingContainerName = container['name']
        print("checking container " + existingContainerName)
        if existingContainerName.startswith("imdb-model"):
            parts = existingContainerName.split("-")
            print(parts)
            suffix = 1
            if (len(parts) == 3):
                newSuffix = int(parts[-1])
                if (newSuffix > suffix):
                    suffix = newSuffix

    container_client = blob_service_client.get_container_client("imdb-model-" + str(suffix))
    blob_list = container_client.list_blobs()
    for blob in blob_list:
        print("\t" + blob.name)

    # Download the blob to a local file
    Path("../model").mkdir(parents=True, exist_ok=True)
    download_file_path = os.path.join("../model", "RandomForestRegressor.pkl")
    print("\nDownloading blob to \n\t" + download_file_path)

    with open(file=download_file_path, mode="wb") as download_file:
         download_file.write(container_client.download_blob(blob.name).readall())

else:
    print("CANNOT ACCESS AZURE BLOB STORAGE - Please set connection string as env variable")
    print(os.environ)
    print("AZURE_STORAGE_CONNECTION_STRING not set")    

file_path = Path(".", "../model/", "RandomForestRegressor.pkl")
with open(file_path, 'rb') as fid:
    model = pickle.load(fid)



print("*** Init Flask App ***")
app = Flask(__name__, template_folder='/Users/rebecca/Desktop/ZHAW/Frühlingssemester2024/test/flask_playground/templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    year = data['year']
    ranking = data['ranking']
    vote_count = data['vote_count']
    
    
    # Vorhersage und Bewertung
    prediction = model.predict([[year, vote_count, ranking]])
    
    
    return jsonify({'rating_prediction': prediction[0]})


    