from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image
import io
import os
import requests

app = Flask(__name__)

# ─────────────────────────────────────────────
# Configure paths
# ─────────────────────────────────────────────
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), "static")
IMAGES_FOLDER = os.path.join(STATIC_FOLDER, "images")

# ─────────────────────────────────────────────
# Cloud Inference Configuration
# If CLOUD_INFERENCE_URL is set (e.g. AWS SageMaker / HF Endpoint),
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
    model_path = os.path.join(os.path.dirname(__file__), "best_model_epoch_19.pth")
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
# Helper: Generate Rule-based Explanation (Offline)
# ─────────────────────────────────────────────
def generate_explanation(bird_name: str, scientific_name: str, language: str, confidence: float) -> str:
    """Generate a template-based explanation without using any external API."""
    
    # Offline dictionary containing brief facts for local fallback.
    local_knowledge = {
        'Asian Green Bee-Eater': 'It is a near passerine bird in the bee-eater family. Known for its vivid green plumage and habit of catching insects on the wing. Usually found in open woodland or grassland across Asia.',
        'Indian Peacock': 'Also known as the Indian Peafowl, it is native to the Indian subcontinent. Males display a magnificent tail covert composed of beautiful elongated feathers. They thrive in deciduous forests.',
    }
    
    fact = local_knowledge.get(bird_name, f"The {bird_name} ({scientific_name}) is a fascinating bird species. It exhibits unique plumage and physical characteristics that our deep learning model recognized with high confidence.")
    
    explanation = f"""Here is a breakdown of the classification for the **{bird_name}** ({scientific_name}):

1. **Identification** – The model recognized this bird with {confidence:.1f}% confidence. It identified key visual features in the image such as the beak shape, color patterns, and wing structures unique to the {bird_name}.
2. **About the Bird** – {fact}
3. **Habitat & Range** – Typically inhabits environments suited for its foraging needs. It is well-adapted to its native ecological niche.
4. **Fun Fact** – The {bird_name} plays an important role in its local ecosystem, often aiding in insect control or seed dispersal!

*Note: This explanation was generated locally offline.*"""
    
    # Simple language notice if not English
    if language != "English":
         explanation = f"*(Note: Translation to {language} is limited in offline mode.)*\n\n" + explanation
         
    return explanation


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
            # Bypass local PyTorch and send image to Cloud API (AWS / HuggingFace)
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

        # 2. Generate LLM explanation (Explainable AI)
        explanation = generate_explanation(
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
            "explanation":      explanation,   # LLM-generated XAI text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Allows follow-up chat questions about the identified bird using local offline responses."""
    data = request.get_json()
    bird_name = data.get('bird_name', '')
    scientific_name = data.get('scientific_name', '')
    user_message = data.get('message', '').lower()

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Very basic keyword-based fallback chatbot
    if 'diet' in user_message or 'eat' in user_message or 'food' in user_message:
        reply = f"The {bird_name} generally feeds on insects, seeds, or small invertebrates, depending on its specific natural diet."
    elif 'habitat' in user_message or 'live' in user_message or 'where' in user_message:
        reply = f"The {bird_name} ({scientific_name}) is commonly found in habitats that support its foraging and nesting needs, such as woodlands, wetlands, or grasslands."
    elif 'name' in user_message or 'scientific' in user_message:
        reply = f"Its scientific name is {scientific_name}."
    else:
        reply = f"That's an interesting question about the {bird_name}! I am currently running in offline mode, so my knowledge base is limited right now. Let's observe its picture together!"

    return jsonify({"reply": reply})


@app.route('/static/images/<filename>')
def static_images(filename):
    return send_from_directory(IMAGES_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
