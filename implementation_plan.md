# BirdBot – Full Implementation Plan (Completed)

A complete record of the architecture, changes made, and how all components fit together.

---

## 📌 Project Goal

Extend the existing **MobileNetV3-based bird classification Flask app** into a **ChatGPT-like conversational AI** that:
1. Classifies a bird image using the fine-tuned deep learning model
2. Explains the classification in **natural language** (Explainable AI)
3. Supports **12 languages** for responses
4. Allows users to **follow-up chat** with an AI ornithologist about the identified bird

---

## 🏗️ Architecture Overview

```
User Browser
     │
     │  POST /predict (image + language)
     ▼
Flask Backend (the_app.py)
     │
     ├─► MobileNetV3 Large (fine-tuned, 25 classes)
     │        └── Returns: class_name, confidence %
     │
     └─► Google Gemini 3.6 Flash (LLM)
              └── Returns: XAI explanation in chosen language

     │  POST /chat (bird_name + user_message + language)
     ▼
Google Gemini 3.6 Flash
     └── Returns: conversational follow-up answer
```

---

## ✅ Completed Changes

---

### 1. Backend – [`the_app.py`](file:///c:/Users/bigha/OneDrive/Desktop/birds_25/the_app.py)

#### What changed
| # | Change | Reason |
|---|--------|--------|
| 1 | Fixed hardcoded absolute model path → `os.path.dirname(__file__)` | Portability |
| 2 | Added `map_location=device` to `torch.load()` | CPU/GPU compatibility |
| 3 | Changed `pretrained=False` → `weights=None` | Remove deprecation warning |
| 4 | Added `torch.softmax()` to get confidence % | Show prediction certainty |
| 5 | Integrated `google.genai` (new SDK) | LLM explanations |
| 6 | Added `load_dotenv()` | Load `GEMINI_API_KEY` from `.env` |
| 7 | Added `generate_explanation()` function | XAI prompt sent to Gemini |
| 8 | Added `/chat` route | Follow-up Q&A with the LLM |
| 9 | Updated Gemini model → `gemini-3.6-flash` | Fix 404 on deprecated model |

#### Key functions

**`generate_explanation(bird_name, scientific_name, language, confidence)`**
- Sends a structured prompt to Gemini as an expert ornithologist
- Asks the LLM to explain:
  1. Physical features that caused the model's decision (XAI)
  2. About the bird (description)
  3. Habitat & range
  4. One fun fact
- Entire response is generated in the user's selected language

**`/predict` route (POST)**
- Accepts: `file` (image) + `language` (string)
- Returns JSON: `prediction`, `scientific_name`, `image`, `confidence`, `explanation`

**`/chat` route (POST)**
- Accepts: `bird_name`, `scientific_name`, `message`, `language`
- Returns JSON: `reply` from Gemini

---

### 2. Frontend – [`templates/index.html`](file:///c:/Users/bigha/OneDrive/Desktop/birds_25/templates/index.html)

Completely redesigned from a basic form into a **premium dark-mode chat application**.

#### UI Components
| Component | Description |
|-----------|-------------|
| **Header** | BirdBot logo with gradient text and model badge |
| **Upload Panel** (left) | Drag-and-drop zone with live image preview |
| **Language Selector** | Dropdown with 12 languages (English, Hindi, Spanish, French, German, Arabic, Chinese, Japanese, Portuguese, Bengali, Tamil, Marathi) |
| **Identify Button** | Triggers `/predict`; shows loading spinner while waiting |
| **Result Card** | Shows bird name, scientific name, animated confidence bar |
| **Chat Panel** (right) | ChatGPT-style message bubbles with typing indicator |
| **Input Bar** | Auto-resizing textarea, send on Enter or button click |

#### Design System
- **Font:** Inter (body) + Playfair Display (logo)
- **Theme:** Dark mode (`#0b1120` background)
- **Accent:** `#4ade80` (green) + `#22d3ee` (cyan)
- **Animations:** `fadeUp` for messages, `bounce` for typing dots, animated confidence bar

---

### 3. Environment – [`.env`](file:///c:/Users/bigha/OneDrive/Desktop/birds_25/.env)

```
GEMINI_API_KEY=your_key_here
```

Loaded automatically by `load_dotenv()` at startup. No manual `set` commands needed.

---

### 4. New Dependencies Installed

| Package | Purpose |
|---------|---------|
| `google-genai==2.24.0` | New Google Gemini Python SDK |
| `peft` | (for chatbot_project fine-tuning scripts) |
| `bitsandbytes` | 4-bit quantization for fine-tuning |
| `trl` | SFT training wrapper |
| `chromadb` | Vector DB for RAG (chatbot_project) |

---

### 5. Cloud Computing (Deployment)

To prepare the application for cloud deployment (e.g., Heroku, AWS, Render):
- **WSGI Server:** Added `gunicorn` to dependencies for robust production serving.
- **Procfile:** Created a `Procfile` with the entrypoint `web: gunicorn the_app:app`.

---

## 📁 Final Project Structure

```
birds_25/
├── the_app.py              ← Main Flask app (MobileNet + Gemini)
├── best_model_epoch_19.pth ← Fine-tuned MobileNetV3 weights
├── .env                    ← GEMINI_API_KEY (do not commit to git!)
├── templates/
│   └── index.html          ← Premium dark-mode chat UI
├── static/
│   └── images/             ← Reference bird images
└── chatbot_project/        ← Standalone LLM fine-tuning boilerplate
    ├── train.py            ← LoRA/PEFT fine-tuning script
    ├── inference.py        ← RAG + LLM chatbot script
    └── requirements.txt    ← Missing deps only
```

---

## 🔄 Request → Response Flow

```
1. User drags image into the browser
2. User selects language (e.g., "Hindi")
3. User clicks "Identify Bird"

4. Browser POSTs {file, language} → /predict

5. Flask:
   a. Preprocesses image (Resize 224x224, Normalize)
   b. Runs MobileNetV3 → gets predicted class + confidence
   c. Looks up bird name + scientific name from bird_info dict
   d. Sends XAI prompt to Gemini 3.6 Flash
   e. Returns JSON {prediction, scientific_name, confidence, explanation}

6. Browser displays:
   - Result card (name + confidence bar)
   - Gemini explanation as a BirdBot chat bubble

7. User types follow-up question → /chat
8. Gemini answers in chosen language → displayed as new bubble
```

---

## 🚀 How to Run

```powershell
cd "c:\Users\bigha\OneDrive\Desktop\birds_25"
python the_app.py
# Open http://127.0.0.1:5000
```

> [!IMPORTANT]
> The `.env` file must contain a valid `GEMINI_API_KEY`. Get one free at https://aistudio.google.com/app/apikey

> [!TIP]
> To add more bird classes in the future: add entries to `bird_info` in `the_app.py`, retrain MobileNetV3 with `out_features` matching the new class count, and replace `best_model_epoch_19.pth`.
