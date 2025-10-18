# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sentiment analysis API using the HerBERT Polish language model (allegro/herbert-base-cased) with LoRA fine-tuning capabilities, deployed via Docker. Built with FastAPI and PyTorch.

## Technology Stack

- **Model**: HerBERT (allegro/herbert-base-cased) - Polish BERT-based language model
- **Fine-tuning**: PEFT/LoRA for efficient parameter tuning
- **Framework**: FastAPI for REST API
- **ML Libraries**: transformers, torch, accelerate, datasets, evaluate
- **Server**: uvicorn with slowapi for rate limiting
- **Python**: >=3.13
- **Package Manager**: uv (based on uv.lock presence)

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
```

### Testing Models
```bash
# Test HerBERT model loading
python test-model.py
```

## Architecture Notes

### Model Architecture
The project uses HerBERT, a Polish language model based on BERT architecture. Fine-tuning will be performed using LoRA (Low-Rank Adaptation) via the PEFT library for memory-efficient training.

### API Structure (Planned)
- FastAPI-based REST endpoints for sentiment analysis
- Rate limiting via slowapi
- Docker containerization for deployment
- Uvicorn ASGI server

### Key Dependencies
- `peft`: Parameter-Efficient Fine-Tuning (LoRA implementation)
- `accelerate`: Distributed training and mixed precision
- `datasets`: Dataset loading and processing
- `evaluate`: Model evaluation metrics
- `sacremoses`: Tokenization utilities
- `protobuf`: Model serialization

## Project Status

This is an early-stage project. The basic model loading test exists (test-model.py) but the API implementation, training scripts, and Docker configuration are yet to be implemented.