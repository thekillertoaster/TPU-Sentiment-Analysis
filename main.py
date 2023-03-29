import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
with open('models/sentiment/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
MODEL_PATH = 'models/sentiment/model.h5'
model = tf.keras.models.load_model(MODEL_PATH)


# Preprocessing function
def preprocess(text):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=100)


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


class Text(BaseModel):
    text: str
    return_labels: bool = False

@app.post('/predict')
async def predict(text: Text):
    preprocessed_text = preprocess(text.text)
    scores = model.predict(preprocessed_text)[0]
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    threshold = 0.75
    if text.return_labels:
        matched_labels = [label for score, label in zip(scores, labels) if score > threshold]
        response = {"matched_labels": matched_labels}
    else:
        response = {'toxicity_scores': scores.tolist()}
    return response

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
