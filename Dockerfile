FROM python:3.13-slim
LABEL description="LoRa adapter for HerBERT model for sentiment analysis of Allegro Reviews"

WORKDIR /app
RUN pip install uv

# Copy dependencies
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

COPY . .

EXPOSE 6000

CMD ["uv", "run", "python", "src/server.py"]
