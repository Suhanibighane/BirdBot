from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image
import io
import os
import requests
from bird_agent import BirdBotAgent

app = Flask(__name__)

# ─────────────────────────────────────────────
# Configure paths
# ─────────────────────────────────────────────
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), "static")
IMAGES_FOLDER = os.path.join(STATIC_FOLDER, "images")

# ─────────────────────────────────────────────
# Cloud Inference Configuration
# If CLOUD_INFERENCE_URL is set (e.g. AWS SageMaker / Google Cloud Run),
# the app bypasses loading PyTorch locally and uses the Cloud API.
# ─────────────────────────────────────────────
CLOUD_INFERENCE_URL = os.environ.get("CLOUD_INFERENCE_URL", "")

if not CLOUD_INFERENCE_URL:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v3_large

    print("Loading PyTorch model locally...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobilenet = mobilenet_v3_large(weights=None)
    mobilenet.classifier[3] = torch.nn.Linear(in_features=1280, out_features=25)
    
    # Priority: mobilenetv3_large_bird_classification.pth -> best_model_epoch_19.pth
    primary_model = os.path.join(os.path.dirname(__file__), "mobilenetv3_large_bird_classification.pth")
    fallback_model = os.path.join(os.path.dirname(__file__), "best_model_epoch_19.pth")
    model_path = primary_model if os.path.exists(primary_model) else fallback_model
    
    print(f"Loading weights from: {os.path.basename(model_path)}")
    mobilenet.load_state_dict(torch.load(model_path, map_location=device))
    mobilenet.to(device)
    mobilenet.eval()

    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
else:
    print(f"Using Cloud Inference Endpoint: {CLOUD_INFERENCE_URL}")

# ─────────────────────────────────────────────
# Bird class names and static info (used as context for LLM)
# ─────────────────────────────────────────────
bird_info = {
    'Asian-Green-Bee-Eater':      {'name': 'Asian Green Bee-Eater',       'scientific_name': 'Merops orientalis',        'image': 'asian_green_bee_eater.jpg'},
    'Brown-Headed-Barbet':        {'name': 'Brown-Headed Barbet',          'scientific_name': 'Psilopogon zeylanicus',    'image': 'brown_headed_barbet.jpg'},
    'Cattle-Egret':               {'name': 'Cattle Egret',                 'scientific_name': 'Bubulcus ibis',            'image': 'cattle_egret.jpg'},
    'Common-Kingfisher':          {'name': 'Common Kingfisher',            'scientific_name': 'Alcedo atthis',            'image': 'common_kingfisher.jpg'},
    'Common-Myna':                {'name': 'Common Myna',                  'scientific_name': 'Acridotheres tristis',     'image': 'common_myna.jpg'},
    'Common-Rosefinch':           {'name': 'Common Rosefinch',             'scientific_name': 'Carpodacus erythrinus',    'image': 'common_rosefinch.jpg'},
    'Common-Tailorbird':          {'name': 'Common Tailorbird',            'scientific_name': 'Orthotomus sutorius',      'image': 'common_tailorbird.jpg'},
    'Coppersmith-Barbet':         {'name': 'Coppersmith Barbet',           'scientific_name': 'Psilopogon haemacephalus','image': 'coppersmith_barbet.jpg'},
    'Forest-Wagtail':             {'name': 'Forest Wagtail',               'scientific_name': 'Dendronanthus indicus',    'image': 'forest_wagtail.jpg'},
    'Gray-Wagtail':               {'name': 'Gray Wagtail',                 'scientific_name': 'Motacilla cinerea',        'image': 'gray_wagtail.jpg'},
    'Hoopoe':                     {'name': 'Hoopoe',                       'scientific_name': 'Upupa epops',              'image': 'hoopoe.jpg'},
    'House-Crow':                 {'name': 'House Crow',                   'scientific_name': 'Corvus splendens',         'image': 'house_crow.jpg'},
    'Indian-Grey-Hornbill':       {'name': 'Indian Grey Hornbill',         'scientific_name': 'Ocyceros birostris',       'image': 'indian_grey_hornbill.jpg'},
    'Indian-Peacock':             {'name': 'Indian Peacock',               'scientific_name': 'Pavo cristatus',           'image': 'indian_peacock.jpg'},
    'Indian-Pitta':               {'name': 'Indian Pitta',                 'scientific_name': 'Pitta brachyura',          'image': 'indian_pitta.jpg'},
    'Indian-Roller':              {'name': 'Indian Roller',                'scientific_name': 'Coracias benghalensis',    'image': 'indian_roller.jpg'},
    'Jungle-Babbler':             {'name': 'Jungle Babbler',               'scientific_name': 'Turdoides striata',        'image': 'jungle_babbler.jpg'},
    'Northern-Lapwing':           {'name': 'Northern Lapwing',             'scientific_name': 'Vanellus vanellus',        'image': 'northern_lapwing.jpg'},
    'Red-Wattled-Lapwing':        {'name': 'Red-Wattled Lapwing',          'scientific_name': 'Vanellus indicus',         'image': 'red_wattled_lapwing.jpg'},
    'Ruddy-Shelduck':             {'name': 'Ruddy Shelduck',               'scientific_name': 'Tadorna ferruginea',       'image': 'ruddy_shelduck.jpg'},
    'Rufous-Treepie':             {'name': 'Rufous Treepie',               'scientific_name': 'Dendrocitta vagabunda',    'image': 'rufous_treepie.jpg'},
    'Sarus-Crane':                {'name': 'Sarus Crane',                  'scientific_name': 'Antigone antigone',        'image': 'sarus_crane.jpg'},
    'White-Breasted-Kingfisher':  {'name': 'White-Breasted Kingfisher',    'scientific_name': 'Halcyon smyrnensis',       'image': 'white_breasted_kingfisher.jpg'},
    'White-Breasted-Waterhen':    {'name': 'White-Breasted Waterhen',      'scientific_name': 'Amaurornis phoenicurus',   'image': 'white_breasted_waterhen.jpg'},
    'White-Wagtail':              {'name': 'White Wagtail',                'scientific_name': 'Motacilla alba',           'image': 'white_wagtail.jpg'},
}

class_names = list(bird_info.keys())

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    language = request.form.get('language', 'English')

    try:
        # 1. Run MobileNetV3 classification
        img_bytes = file.read()
        
        if CLOUD_INFERENCE_URL:
            # Bypass local PyTorch and send image to Cloud API (AWS / Google Cloud Run)
            response = requests.post(CLOUD_INFERENCE_URL, files={'file': img_bytes})
            response.raise_for_status()
            api_result = response.json()
            class_index = api_result.get('class_index', 0)
            confidence_pct = api_result.get('confidence', 95.0)
        else:
            # Run local PyTorch model
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = mobilenet(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            class_index = predicted.item()
            confidence_pct = confidence.item() * 100

        if class_index >= len(class_names):
            return jsonify({"error": "Predicted class index out of range"}), 500

        predicted_class = class_names[class_index]
        bird_details = bird_info[predicted_class]

        # 2. Generate Explanation via BirdBotAgent (Tier 1 Gemini -> Tier 2 Wikipedia -> Tier 3 Local KB)
        explanation = BirdBotAgent.generate_explanation(
            bird_key=predicted_class,
            bird_name=bird_details['name'],
            scientific_name=bird_details['scientific_name'],
            language=language,
            confidence=confidence_pct
        )

        return jsonify({
            "prediction":       bird_details['name'],
            "scientific_name":  bird_details['scientific_name'],
            "image":            bird_details['image'],
            "confidence":       round(confidence_pct, 2),
            "language":         language,
            "explanation":      explanation,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Allows follow-up Q&A about the identified bird powered by BirdBotAgent."""
    data = request.get_json() or {}
    bird_name = data.get('bird_name', '')
    scientific_name = data.get('scientific_name', '')
    user_message = data.get('message', '')
    language = data.get('language', 'English')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    reply = BirdBotAgent.answer_chat(
        bird_name=bird_name,
        scientific_name=scientific_name,
        message=user_message,
        language=language
    )

    return jsonify({"reply": reply})


@app.route('/static/images/<filename>')
def static_images(filename):
    return send_from_directory(IMAGES_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
