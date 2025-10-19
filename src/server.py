from fastapi import FastAPI
from inference import predict
app = FastAPI()

app.get('/')
def home():
    return 'Hello World!'

@app.post('/predict_sentiment')
async def predict_sentiment(text: str):
    pred = predict(text)
    return {'sentiment': pred}