# 📈 Autonomous Alpha: Quant Engine & Agentic Trading API

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/Deployment-FastAPI-green)
![TensorFlow](https://img.shields.io/badge/Deep%20Learning-LSTM%20%7C%20GAN-orange)
![Agentic AI](https://img.shields.io/badge/GenAI-LangChain%20%7C%20RAG-purple)

## 📌 Project Overview
An end-to-end autonomous financial machine learning pipeline designed to forecast S&P 500 movements. This system ingests 16 years of historical news alongside live API data, scores sentiment using a dual-NLP pipeline (VADER and FinBERT), and generates alpha signals via XGBoost and LSTMs.

The raw quantitative outputs are fed into a **Retrieval-Augmented Generation (RAG) engine**, where an autonomous LLM committee debates the mathematical signals against live macroeconomic context to generate trading directives. The predictive engine is decoupled and served via a **FastAPI** microservice.

## 🚀 Repository Structure
* `notebooks/` - Contains the 10-Phase research and training pipeline.
* `src/` - Contains the FastAPI application (`api.py`).
* `models/` - Contains the serialized `.json` model weights.
