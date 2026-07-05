import requests
import json

def emotion_detector(text_to_analyze):
    # Definiujemy parametry połączenia z IBM Watson
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = { "raw_document": { "text": text_to_analyze } }
    
    # Wysyłamy zapytanie do API
    response = requests.post(url, json=myobj, headers=headers)
    
    # 1. Konwertujemy surowy tekst odpowiedzi na słownik Pythona przy użyciu biblioteki json
    formatted_response = json.loads(response.text)
    
    # 2. Wyciągamy zestaw emocji z zagnieżdżonej struktury odpowiedzi IBM Watson
    emotions = formatted_response['emotionPredictions'][0]['emotion']
    
    # Pobieramy punkty dla każdej wymaganej emocji
    anger_score = emotions['anger']
    disgust_score = emotions['disgust']
    fear_score = emotions['fear']
    joy_score = emotions['joy']
    sadness_score = emotions['sadness']
    
    # 3. Logika znajdująca dominującą emocję (szukamy klucza z najwyższą wartością)
    dominant_emotion = max(emotions, key=emotions.get)
    
    # 4. Tworzymy końcowy słownik wyjściowy o dokładnie takim formacie, jakiego żąda instrukcja
    output_format = {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score,
        'dominant_emotion': dominant_emotion
    }
    
    return output_format