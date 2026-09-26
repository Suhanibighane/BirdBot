# 🦅 BirdBot – Complete Implementation Plan (DL + Free Retrieval Agent + CC)

This document contains the complete technical architecture and implementation details for **BirdBot**, an intelligent AI ornithologist web application built using **Deep Learning (DL)**, **Free Tool-Based Retrieval Agent (Wikipedia / Web)**, and **Cloud Computing (CC)**.

---

## 📌 Project Overview & Objectives

1. **Vision Classification (Deep Learning)**: Classifies bird images into 25 species using a fine-tuned **MobileNetV3 Large** neural network model (`mobilenetv3_large_bird_classification.pth`).
2. **Zero-Failure Hybrid Agent (XAI & Q&A)**: 
   - **Tier 1 (Cloud LLM)**: Google Gemini API (when online with API key) for natural, conversational reasoning in 12 languages.
   - **Tier 2 (Free Live Tool Retrieval)**: Wikipedia REST API web agent to fetch live facts without API keys or quota limits if the API fails.
   - **Tier 3 (Local 25-Species Knowledge Base)**: Comprehensive offline encyclopedia ensuring 100% uptime with 0ms latency.
3. **Cloud Computing (Deployment & Scalability)**: 
   - Option for separate cloud inference microservice (Google Cloud Run / AWS Lambda).
   - Lightweight web application deployment via Render / Railway / Heroku.

---

## 🏗️ System Architecture

```
                                    User Browser
                             (Upload Image / Chat / Lang)
                                          │
                                          ▼
                            Flask Web App (the_app.py)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
      [DEEP LEARNING LAYER]                          [AI AGENT LAYER (bird_agent.py)]
     MobileNetV3 Classifier                                       │
  (mobilenetv3_large_bird.pth)           ┌────────────────────────┼────────────────────────┐
                  │                      │                        │                        │
                  ▼                      ▼                        ▼                        ▼
       Predict Class & Conf %      Tier 1: Gemini API       Tier 2: Wikipedia Agent  Tier 3: 25-Bird Local KB
                                (Natural LLM & Multi-lang)   (Live Free Web Fetch)   (100% Offline Database)
```

---

## 📁 Project Structure

```
birds_25/
├── the_app.py                                  ← Flask web server & prediction endpoints
├── bird_agent.py                               ← Free tool-based agent & 25-species knowledge base
├── mobilenetv3_large_bird_classification.pth   ← Fine-tuned MobileNetV3 weights (25 classes)
├── best_model_epoch_19.pth                     ← Backup model checkpoint
├── requirements.txt                            ← Cleaned project dependencies
├── Procfile                                    ← Cloud deployment command (gunicorn)
├── Dockerfile                                  ← Container build recipe
├── .env                                        ← GEMINI_API_KEY (optional)
├── templates/
│   └── index.html                              ← Dark-mode chat UI
└── static/
    └── images/                                 ← Reference bird images
```

---

## 🔬 Component Breakdown

### 1. Deep Learning (DL)
- **Model**: `MobileNetV3 Large` (1280 input features, customized classifier with 25 output logits).
- **Preprocessing Pipeline**: 
  - Resize to `(224, 224)`
  - Convert to Tensor
  - ImageNet normalization: Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`
- **Inference**: Softmax probability calculation returning the predicted bird species and confidence percentage.

### 2. Free Tool-Based Retrieval Agent (`bird_agent.py`)
- **Wikipedia REST API**: Uses the Wikimedia REST API (`/api/rest_v1/page/summary/{title}`) to fetch species facts in real time. Free, no API key required, and resilient against API quota limits.
- **Local 25-Bird Knowledge Store**: Pre-seeded with identification marks, habitat, diet, and fun facts for all 25 bird species in the dataset.
- **Explainable AI (XAI)**: Breaks down the prediction into:
  1. 🔍 Visual markers (beak shape, plumage, color patterns)
  2. 🐦 Species overview
  3. 🌿 Habitat & distribution
  4. 🐛 Diet & feeding behavior
  5. ⭐ Fun trivia fact

### 3. Cloud Computing (CC)
- **Production Server**: Gunicorn WSGI (`web: gunicorn the_app:app`)
- **Cloud Microservice Support**: `CLOUD_INFERENCE_URL` support to route heavy PyTorch inferences to Google Cloud Run or AWS Lambda endpoints.
- **Free Web Hosting**: Deployable on **Render.com**, **Railway**, or **Google Cloud Run**.

---

## 🚀 How to Run Locally

1. **Activate your environment**:
   ```powershell
   cd "c:\Users\bigha\OneDrive\Desktop\birds_25"
   .\venv\Scripts\activate
   ```

2. **Run the Flask application**:
   ```powershell
   python the_app.py
   ```

3. **Open in browser**:
   Navigate to `http://127.0.0.1:5000`

---

## ☁️ Cloud Deployment (Free Tier on Render)

1. Push your repository to GitHub.
2. Log into [Render.com](https://render.com) and click **New Web Service**.
3. Link your repository.
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn the_app:app`
5. Click **Deploy**.
