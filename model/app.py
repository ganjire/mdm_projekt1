from flask import Flask, jsonify, render_template, request
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle
from pymongo import MongoClient


app = Flask(__name__, template_folder='/Users/rebecca/Desktop/ZHAW/Frühlingssemester2024/test/flask_playground/templates')

def fetch_data():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.59 Safari/537.36'
    }
    url = 'https://www.imdb.com/chart/top'
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        movies = soup.find_all('li', class_='ipc-metadata-list-summary-item')

        data = []

        for movie in movies:
            title_element = movie.find('h3', class_='ipc-title__text')
            if title_element:
                # Zuerst wird der gesamte Text extrahiert und unnötige Leerzeichen entfernt
                full_title_text = title_element.text.strip()
                
                # Trennen des Textes in Ranking und Titel, basierend auf dem ersten Leerzeichen
                split_text = full_title_text.split(' ', 1) # Begrenzt den Split auf das erste Vorkommen
                if len(split_text) > 1:
                    Ranking, Title = split_text[0], split_text[1]
                else:
                    Ranking = "Unknown"
                    Title = full_title_text
            
                # Extrahieren des Jahres aus dem `span`-Element
                year_element = movie.find('span', class_='cli-title-metadata-item')
                if year_element:
                    Year = year_element.text.strip()
                else:
                    Year = "Jahr unbekannt"
            
                rating_element = movie.find('span', class_='ipc-rating-star--imdb')
                if rating_element:
                    # Extrahiere den gesamten Text, der das Rating und die Anzahl der Stimmen enthält
                    rating_vote_text = rating_element.text.strip()
                    
                    # Trenne das Rating von der Anzahl der Stimmen, basierend auf der Annahme,
                    # dass das Rating am Anfang steht und die Anzahl der Stimmen danach kommt
                    rating_text_parts = rating_vote_text.split('(')
                    Rating = rating_text_parts[0].strip()
                    if len(rating_text_parts) > 1:
                        # Entferne die schließende Klammer und extrahiere nur die Anzahl der Stimmen
                        VoteCount = rating_text_parts[1].replace(')', '').strip()
                    else:
                        VoteCount = "Stimmenanzahl unbekannt"
                else:
                    Rating = "Rating unbekannt"
                    VoteCount = "Stimmenanzahl unbekannt"
                    
                # Füge die extrahierten Daten zur Liste hinzu
                data.append([Ranking, Title, Year, Rating, VoteCount])

        # Erstelle das DataFrame
        df = pd.DataFrame(data, columns=['Ranking', 'Title', 'Year', 'Rating', 'Vote Count'])
        print(df)

        # Funktion zur Konvertierung von Vote Count
        def convert_vote_count(vote_str):
            if 'M' in vote_str:
                return float(vote_str.replace('M', '')) * 1000000
            elif 'K' in vote_str:
                return float(vote_str.replace('K', '')) * 1000
            else:
                return float(vote_str)
            
        # Konvertiere die 'Vote Count'-Daten
        df['Vote Count'] = df['Vote Count'].apply(lambda x: convert_vote_count(x) if x != "Stimmenanzahl unbekannt" else np.nan)
        df['Ranking'] = df['Ranking'].str.replace('.', '').astype(int)
        df['Year'] = df['Year'].astype(float)
        df['Rating'] = df['Rating'].astype(float)

        # Aufteilen der Daten
        X = df[['Year', 'Vote Count', 'Ranking']]
        y = df['Rating']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, df
    
    
    return df
    


def insert_data_to_mongodb(df):
    try:
        # Konvertiere das DataFrame in ein Format, das MongoDB versteht (Liste von Dictionaries)
        data_dict = df.to_dict("records")
        
        # Verbindungsstring (Stelle sicher, dass du deinen echten Verbindungsstring hier einfügst)
        client = MongoClient('mongodb+srv://mongodb:RockyRebi12@mdm-projekt.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000')

        # Wähle deine Datenbank
        db = client['imdb']

        # Wähle deine Collection
        collection = db['movies']


        # Füge die Daten ein
        collection.insert_many(data_dict)
        print("Daten erfolgreich eingefügt.")
    except Exception as e:
        print(f"Fehler beim Einfügen der Daten: {e}") 
    
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_data', methods=['GET'])
def get_data():
    X_train, X_test, y_train, y_test, df = fetch_data()
    return jsonify({'train_data': X_train.to_dict(), 'test_data': X_test.to_dict(), 'train_labels': y_train.tolist(), 'test_labels': y_test.tolist(), 'dataframe': df.to_dict()})

@app.route('/predict', methods=['POST'])
def predict():
    global df
    data = request.get_json()
    year = data['year']
    vote_count = data['vote_count']
    ranking = int(data['ranking'])
    
    
    X_train, X_test, y_train, y_test, _ = fetch_data()
    
    # Modell trainieren
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Speichern des Modells
    with open('RandomForestRegressor.pkl', 'wb') as fid:
        pickle.dump(model, fid)
    
    # Vorhersage und Bewertung
    prediction = model.predict([[year, vote_count, ranking]])
    
    # Berechne und drucke MSE und R²
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Wichtigkeit jedes Features 
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importances = pd.Series(importances, index=feature_names)
    feature_importances_dict = feature_importances.sort_values(ascending=False).to_dict()


    # Laden des Modells
    with open('RandomForestRegressor.pkl', 'rb') as fid:
        model_loaded = pickle.load(fid)
        
        
    matched_movie = df[df['Ranking'] == ranking]
    
    if not matched_movie.empty:
        # Wenn ein passender Film gefunden wurde, extrahiere den Titel.
        title = matched_movie.iloc[0]['Title']
    else:
        title = "Film nicht gefunden"
    
    #return jsonify
    return jsonify({
        'rating_prediction': prediction[0],
        'mse': mse,
        'r2': r2,
        'feature_importances': feature_importances_dict,
        'title': title
       
    })
    


if __name__ == '__main__':
    df = pd.DataFrame(['Ranking', 'Title', 'Year', 'Rating', 'Vote Count'])
    _, _, _, _, df = fetch_data()
    insert_data_to_mongodb(df)
    app.run(debug=True)
