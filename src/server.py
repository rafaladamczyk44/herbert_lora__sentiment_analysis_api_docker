from fastapi import FastAPI, HTTPException
from inference import predict
from pydantic import BaseModel, Field, field_validator
import uvicorn
"""
http://localhost:6000/docs#/
http://localhost:6000/docs#/Prediction/predict_sentiment_predict_post
"""
app = FastAPI(
    title="HerBERT Sentiment Analysis API",
    description="Polish sentiment analysis using fine-tuned HerBERT model with LoRA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
# Request/Response Models
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")

    @field_validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Ten produkt jest naprawdę świetny! Polecam każdemu."
            }
        }

class SentimentResponse(BaseModel):
    text: str = Field(..., description="Input text")
    label: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    probabilities: dict = Field(..., description="Probability distribution across all labels")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Ten produkt jest naprawdę świetny! Polecam każdemu.",
                "label": "positive",
                "confidence": 0.9524,
                "probabilities": {
                    "positive": 0.9524,
                    "neutral": 0.0321,
                    "negative": 0.0155
                }
            }
        }

@app.get('/', tags=["General"])
def home():
    return {
        "message": "HerBERT Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.post('/predict', response_model=SentimentResponse, tags=["Prediction"])
async def predict_sentiment(sentiment_request: SentimentRequest):
    try:
        result = predict(sentiment_request.text)

        # Check if prediction returned an error
        if "label" not in result:
            raise HTTPException(status_code=400, detail=result.get("text", "Prediction failed"))

        return SentimentResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6000)