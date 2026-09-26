import os
import re
import json
import requests
from typing import Dict, Any, Optional

LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Chinese": "zh-CN",
    "Japanese": "ja",
    "Portuguese": "pt",
    "Bengali": "bn",
    "Tamil": "ta",
    "Marathi": "mr",
}


def _translate_response(text: str, language: str) -> str:
    if not text or language == "English":
        return text

    target_code = LANGUAGE_CODES.get(language)
    if not target_code:
        return f"{text}\n\n[Translation unavailable for {language}; showing English.]"

    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source="auto", target=target_code).translate(text)
        if translated:
            return translated
    except Exception as error:
        print(f"[Translation Warning] Could not translate to {language}: {error}")

    return f"{text}\n\n[Translation unavailable for {language}; showing English.]"


def _translate_to_english(text: str, language: str) -> str:
    if not text or language == "English":
        return text

    try:
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source="auto", target="en").translate(text) or text
    except Exception as error:
        print(f"[Translation Warning] Could not translate question from {language}: {error}")
        return text

# Load environment variables if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =====================================================================
# TIER 3: Comprehensive Local 25-Bird Knowledge Base (100% Offline)
# =====================================================================
BIRD_LOCAL_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "Asian-Green-Bee-Eater": {
        "name": "Asian Green Bee-Eater",
        "scientific_name": "Merops orientalis",
        "description": "A slender, richly colored bird with bright bronze-green plumage, a black throat band (gorget), and elongated central tail feathers.",
        "identification": "Recognized by its vivid lime-green feathers, rufous crown, black eye-stripe, and long, curved black beak designed for catching flying insects.",
        "habitat": "Open grasslands, agricultural fields, scrub forests, and semi-arid plains across South and Southeast Asia.",
        "diet": "Strictly insectivorous; specializes in bees, wasps, dragonflies, and beetles captured during agile aerial sorties.",
        "fun_fact": "Before eating a bee or wasp, it repeatedly rubs and batters the insect against a perch to remove the sting and venom sac."
    },
    "Brown-Headed-Barbet": {
        "name": "Brown-Headed Barbet",
        "scientific_name": "Psilopogon zeylanicus",
        "description": "A medium-sized Asian barbet with a streaked brown head, bright orange-yellow eye patches, and a vibrant green body.",
        "identification": "Distinctive yellow patch of bare skin around each eye, heavy reddish-orange bill, and a loud, repetitive 'kutroo-kutroo' call.",
        "habitat": "Tropical and subtropical moist forests, fruit orchards, gardens, and urban parks throughout the Indian subcontinent.",
        "diet": "Frugivorous (figs, berries, mangoes, papayas) and occasionally takes insects, flowers, and nectar.",
        "fun_fact": "They excavate their own nesting cavities in rotten tree trunks using their powerful, heavy beaks."
    },
    "Cattle-Egret": {
        "name": "Cattle Egret",
        "scientific_name": "Bubulcus ibis",
        "description": "A compact white heron often seen following grazing livestock, developing buff-orange plumes during breeding season.",
        "identification": "Stocky white plumage, stout yellow bill, and dark/yellow legs. Shorter neck compared to other egret species.",
        "habitat": "Pastures, farmlands, marshes, paddy fields, and flooded meadows worldwide.",
        "diet": "Grasshoppers, ticks, flies, frogs, earthworms, and small reptiles stirred up by grazing cattle.",
        "fun_fact": "It enjoys a symbiotic relationship with grazing animals, feeding on ticks and insects that bother the mammals."
    },
    "Common-Kingfisher": {
        "name": "Common Kingfisher",
        "scientific_name": "Alcedo atthis",
        "description": "A sparrow-sized, jewel-like kingfisher with brilliant metallic cyan-blue upperparts and rich copper-orange underparts.",
        "identification": "Dazzling iridescent blue back stripe, bright orange belly, short tail, and long dagger-like black bill.",
        "habitat": "Clear, slow-moving streams, rivers, freshwater lakes, ponds, and mangrove swamps.",
        "diet": "Small fish (minnows, sticklebacks), aquatic insects, freshwater shrimp, and tadpoles.",
        "fun_fact": "It has specialized binocular vision that corrects for water refraction, allowing pinpoint diving accuracy."
    },
    "Common-Myna": {
        "name": "Common Myna",
        "scientific_name": "Acridotheres tristis",
        "description": "An assertive, highly adaptable member of the starling family with rich brown body plumage and a glossy black hood.",
        "identification": "Bright yellow bill, yellow legs, and bare yellow skin around the eyes, with white wing patches visible in flight.",
        "habitat": "Urban areas, agricultural land, roadsides, and open scrub across Asia (widely introduced globally).",
        "diet": "Omnivorous; feeds on insects, fruits, grains, kitchen scraps, and small lizards.",
        "fun_fact": "They are exceptional vocal mimics and can replicate human speech and industrial sounds with high clarity."
    },
    "Common-Rosefinch": {
        "name": "Common Rosefinch",
        "scientific_name": "Carpodacus erythrinus",
        "description": "A passerine bird where adult males display brilliant raspberry-red heads, breasts, and rumps.",
        "identification": "Male has vibrant rose-red head and breast; females are olive-brown with streaked underparts and a stout conical bill.",
        "habitat": "Open scrublands, willow thickets near rivers, forest edges, and agricultural fields.",
        "diet": "Seeds, buds, berries, fruits, and small insects during the breeding season.",
        "fun_fact": "Their cheerful song sounds distinctively like the phrase 'pleased to meet you!'."
    },
    "Common-Tailorbird": {
        "name": "Common Tailorbird",
        "scientific_name": "Orthotomus sutorius",
        "description": "A tiny, active songbird famous for constructing cradle-like nests by stitching leaves together.",
        "identification": "Bright olive-green upperparts, creamy-white belly, a rufous-chestnut cap, and an upright perky tail.",
        "habitat": "Gardens, forest edges, scrublands, and urban vegetation across tropical Asia.",
        "diet": "Insects, larvae, tiny spiders, and nectar.",
        "fun_fact": "It pierces the edges of large leaves with its beak and sews them with plant fibers or spider silk to make a concealed nest."
    },
    "Coppersmith-Barbet": {
        "name": "Coppersmith Barbet",
        "scientific_name": "Psilopogon haemacephalus",
        "description": "A small barbet with vibrant crimson forehead and throat patches and yellow eye bands.",
        "identification": "Vivid red forehead and breast patch, green body, yellowish eye ring, and a repetitive metallic 'tuk-tuk-tuk' call.",
        "habitat": "Urban gardens, roadside trees, orchards, and open woodland across the Indian subcontinent.",
        "diet": "Primarily banyan and peepal figs, small fruits, and nectar.",
        "fun_fact": "Its name comes from its rhythmic call, which sounds exactly like a coppersmith striking metal with a hammer."
    },
    "Forest-Wagtail": {
        "name": "Forest Wagtail",
        "scientific_name": "Dendronanthus indicus",
        "description": "A forest-dwelling passerine distinct for swinging its tail sideways rather than up and down.",
        "identification": "Olive-brown upperparts, white belly, a distinctive double black breast band, and white wing bars.",
        "habitat": "Dense forest floors, shaded woodland trails, and bamboo groves in South and East Asia.",
        "diet": "Insects, spiders, worms, and small invertebrates found among fallen leaf litter.",
        "fun_fact": "Unlike all other wagtails that bob their tails vertically, the Forest Wagtail wags its tail horizontally side-to-side."
    },
    "Gray-Wagtail": {
        "name": "Gray Wagtail",
        "scientific_name": "Motacilla cinerea",
        "description": "A slender, graceful bird with slate-gray upperparts, a remarkably long black-and-white tail, and bright yellow undertail coverts.",
        "identification": "Bluish-gray back, lemon-yellow belly and vent, slender pinkish legs, and constant vertical tail bobbing.",
        "habitat": "Fast-flowing rocky mountain streams, torrents, water cascades, and rocky riverbanks.",
        "diet": "Aquatic flies, mayflies, midges, beetles, and small crustaceans.",
        "fun_fact": "It has the longest tail of all European and Asian wagtail species."
    },
    "Hoopoe": {
        "name": "Hoopoe",
        "scientific_name": "Upupa epops",
        "description": "An unmistakable bird known for its flamboyant erectile feather crest and dramatic zebra-striped black and white wings.",
        "identification": "Pinkish-cinnamon body, long decurved bill, and a fan-shaped crest tipped in black that raises when excited.",
        "habitat": "Dry open countryside, semi-deserts, grasslands, orchards, and parklands across Afro-Eurasia.",
        "diet": "Insects, beetle larvae, crickets, pupae, and subterranean worms probed from the soil.",
        "fun_fact": "Female hoopoes and nestlings secrete a foul-smelling chemical from their uropygial gland to deter predators."
    },
    "House-Crow": {
        "name": "House Crow",
        "scientific_name": "Corvus splendens",
        "description": "A highly intelligent, adaptable urban corvid with a two-toned gray and black body.",
        "identification": "Glossy black forehead, crown, throat, and wings with a contrasting smoky-gray neck, mantle, and breast.",
        "habitat": "Towns, cities, agricultural settlements, and coastal fishing ports across South Asia.",
        "diet": "Opportunistic omnivore: seeds, human food refuse, small reptiles, insects, eggs, and carrion.",
        "fun_fact": "Recognized by scientists as one of the most intelligent bird species, capable of tool use and facial recognition."
    },
    "Indian-Grey-Hornbill": {
        "name": "Indian Grey Hornbill",
        "scientific_name": "Ocyceros birostris",
        "description": "A medium-sized arboreal hornbill with silvery-grey plumage and a curved bill with a pointed horn (casque).",
        "identification": "Grey body plumage, long graduated tail with a black subterminal band, and a dark bill topped with a sharp casque.",
        "habitat": "Deciduous forests, roadside avenues, university campuses, and city gardens with mature fruiting trees.",
        "diet": "Figs, berries, flowers, tree bark insects, and small lizards.",
        "fun_fact": "During breeding, the female seals herself inside a tree hollow with mud and droppings, leaving only a slit for the male to feed her."
    },
    "Indian-Peacock": {
        "name": "Indian Peacock",
        "scientific_name": "Pavo cristatus",
        "description": "The national bird of India, celebrated for the male's iridescent sapphire-blue neck and massive train of eye-spotted feathers.",
        "identification": "Males feature metallic blue neck/breast and an ornate train; females (peahens) are mottled brown with greenish neck plumage.",
        "habitat": "Dry deciduous forests, scrub jungles, agricultural edges, and riverbanks across the Indian subcontinent.",
        "diet": "Omnivorous: seeds, berries, tender shoots, insects, small rodents, and venomous snakes.",
        "fun_fact": "They are famous for their ability to fight and eat venomous cobras and vipers without suffering harm."
    },
    "Indian-Pitta": {
        "name": "Indian Pitta",
        "scientific_name": "Pitta brachyura",
        "description": "A stubby-tailed, jewel-toned ground bird colloquially called the 'nine-colored bird' (Navrang).",
        "identification": "Green mantle, blue wing patches, yellow-buff belly with a scarlet vent, black eye-stripe, and white throat.",
        "habitat": "Dense undergrowth, deciduous scrub, moist forest floors, and shaded ravines.",
        "diet": "Insects, grubs, snails, millipedes, and earthworms foraged from moist leaf litter.",
        "fun_fact": "Their local name 'Navrang' literally translates to 'nine colors' due to their spectacular multihued plumage."
    },
    "Indian-Roller": {
        "name": "Indian Roller",
        "scientific_name": "Coracias benghalensis",
        "description": "A striking bird that explodes into brilliant shades of electric turquoise and deep ultramarine blue in flight.",
        "identification": "Brownish back and lilac-brown breast at rest; transforms into vibrant electric blue wings and tail in flight.",
        "habitat": "Open grasslands, agricultural fields, roadside power lines, and open scrublands.",
        "diet": "Beetles, grasshoppers, crickets, scorpions, spiders, and small amphibians.",
        "fun_fact": "State bird of multiple Indian states; famous for its acrobatic mid-air rolling dive displays during courtship."
    },
    "Jungle-Babbler": {
        "name": "Jungle Babbler",
        "scientific_name": "Turdoides striata",
        "description": "A gregarious, noisy passerine that constantly moves in tightly-knit family groups of six to ten birds.",
        "identification": "Ashy-brown mottled plumage, pale piercing white/yellow eyes, and yellowish bill and legs.",
        "habitat": "Forest margins, gardens, suburban scrub, and bamboo clumps.",
        "diet": "Insects, larvae, fallen fruit, seeds, and nectar.",
        "fun_fact": "Popularly nicknamed the 'Seven Sisters' in India because they are almost always observed in noisy groups of seven."
    },
    "Northern-Lapwing": {
        "name": "Northern Lapwing",
        "scientific_name": "Vanellus vanellus",
        "description": "A distinctive wading bird with iridescent dark green/bronze upperparts, a black chest band, and a long wispy crest.",
        "identification": "Long slender crest curled upwards, iridescent green-purple back, crisp black chest, and rounded paddle-shaped wings.",
        "habitat": "Wetlands, marshes, flooded agricultural fields, and mudflats.",
        "diet": "Earthworms, insects, small snails, spiders, and seeds.",
        "fun_fact": "Its unique 'peewit' call and floppy, erratic wing flapping makes it one of the easiest waders to spot in flight."
    },
    "Red-Wattled-Lapwing": {
        "name": "Red-Wattled Lapwing",
        "scientific_name": "Vanellus indicus",
        "description": "A ground-dwelling plover noted for its bright red fleshy facial wattles and alarm call.",
        "identification": "Bright red fleshy wattle in front of each eye, black head and chest, bronze-brown back, and yellow legs.",
        "habitat": "Lakeshores, marshes, open farmlands, ploughed fields, and gravel beds.",
        "diet": "Beetles, ants, larvae, mollusks, and weed seeds.",
        "fun_fact": "Its loud alarm call sounds like 'Did-he-do-it? Pity-to-do-it!', alerting all surrounding animals to incoming predators."
    },
    "Ruddy-Shelduck": {
        "name": "Ruddy Shelduck",
        "scientific_name": "Tadorna ferruginea",
        "description": "A large, stately waterfowl with striking cinnamon-orange plumage and a pale buff-white head.",
        "identification": "Deep orange-brown body, pale cream head, black tail, and white wing patches with iridescent green speculum.",
        "habitat": "Large inland water bodies, high-altitude lakes, wide rivers, and reservoir margins.",
        "diet": "Aquatic plants, seeds, shoots, mollusks, crustaceans, and small fish.",
        "fun_fact": "Known as the 'Brahminy Duck' in India and revered in Buddhist folklore as a symbol of fidelity and lifelong pairing."
    },
    "Rufous-Treepie": {
        "name": "Rufous Treepie",
        "scientific_name": "Dendrocitta vagabunda",
        "description": "A long-tailed member of the crow family with rich rufous, silver-grey, and black plumage.",
        "identification": "Long graduated silvery-grey tail with a black tip, black hood, cinnamon-rufous back, and grey wing patches.",
        "habitat": "Open deciduous forest, scrub country, plantations, and city gardens.",
        "diet": "Fruits, seeds, nectar, invertebrates, small reptiles, bird eggs, and carrion.",
        "fun_fact": "They have a mutualistic bond with tigers and deer, often grooming parasites off deer coats in national parks."
    },
    "Sarus-Crane": {
        "name": "Sarus Crane",
        "scientific_name": "Antigone antigone",
        "description": "The tallest flying bird in the world, standing up to 1.8 meters tall with grey plumage and a vibrant red head.",
        "identification": "Towering stature, uniform ash-grey body, bare crimson-red head and upper neck, and long greenish-pink legs.",
        "habitat": "Wetlands, marshes, flooded paddy fields, and natural shallow swamps.",
        "diet": "Aquatic plants, tubers, seeds, insects, crustaceans, frogs, and small fish.",
        "fun_fact": "They mate for life and engage in magnificent unison calling displays and synchronized courtship dances."
    },
    "White-Breasted-Kingfisher": {
        "name": "White-Breasted Kingfisher",
        "scientific_name": "Halcyon smyrnensis",
        "description": "A bold tree kingfisher with a brilliant turquoise-blue back, chocolate-brown head/belly, and a clean white bib.",
        "identification": "Bright white throat and breast, rich chestnut brown head and lower body, glowing electric-blue back and wings, and thick red bill.",
        "habitat": "Agricultural fields, canals, roadside wires, ponds, gardens, and forest edges (often far from water).",
        "diet": "Crabs, frogs, lizards, rodents, grasshoppers, beetles, and fish.",
        "fun_fact": "Unlike common kingfishers, it hunts primarily on land for lizards and rodents rather than strictly for fish."
    },
    "White-Breasted-Waterhen": {
        "name": "White-Breasted Waterhen",
        "scientific_name": "Amaurornis phoenicurus",
        "description": "A dark slate-grey rail with a clean white face, breast, and belly, and a rusty-red patch under the short tail.",
        "identification": "Clean white face and underparts contrasting sharply with dark slaty-grey upperparts, long yellow legs and toes.",
        "habitat": "Marshes, ponds, water hyacinth beds, wet paddy fields, and roadside ditches.",
        "diet": "Insects, small fish, aquatic invertebrates, weed seeds, and tender plant shoots.",
        "fun_fact": "They walk with a characteristic jerky motion, flicking their short tail upright to reveal a cinnamon-rufous patch."
    },
    "White-Wagtail": {
        "name": "White Wagtail",
        "scientific_name": "Motacilla alba",
        "description": "A slender black, white, and grey passerine bird renowned for its energetic walking and constant tail pumping.",
        "identification": "Crisp monochrome appearance: black bib, white face, grey or black mantle, white belly, and long wagging tail.",
        "habitat": "Open country, riverbanks, lake margins, lawns, paved plazas, and agricultural land.",
        "diet": "Small beetles, flies, midges, caterpillars, and spiders caught on foot.",
        "fun_fact": "It runs in fast spurts, stopping abruptly to bob its tail, and uses an undulating wave-like flight pattern."
    }
}

# =====================================================================
# TIER 2: Free Tool-Based Retrieval (Wikipedia & Web Agent)
# =====================================================================
def search_wikipedia_summary(query_title: str) -> Optional[str]:
    """
    Fetches the live Wikipedia summary using the free Wikimedia REST API.
    Does not require any API keys or paid tiers.
    """
    try:
        clean_title = query_title.strip().replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_title}"
        headers = {"User-Agent": "BirdBot-AI/2.0 (contact: support@birdbot.local)"}
        resp = requests.get(url, headers=headers, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract")
            if extract and len(extract) > 40:
                return extract
    except Exception as e:
        print(f"[Wikipedia Tool] Error fetching for '{query_title}': {e}")
    return None

def web_agent_search(bird_name: str, scientific_name: str) -> Optional[str]:
    """
    Searches Wikipedia for the bird by common name or scientific name.
    """
    summary = search_wikipedia_summary(bird_name)
    if summary:
        return summary
    
    if scientific_name:
        summary = search_wikipedia_summary(scientific_name)
        if summary:
            return summary
            
    return None

# =====================================================================
# TIER 1: Google Gemini LLM API (if available)
# =====================================================================
def get_gemini_model():
    """Initializes Google Gemini client if GEMINI_API_KEY is present."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client
    except Exception:
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=api_key)
            return legacy_genai.GenerativeModel("gemini-1.5-flash")
        except Exception:
            return None

# =====================================================================
# AGENT CORE: Unified Explain & Q&A Router
# =====================================================================
class BirdBotAgent:
    """
    Multi-Tier AI Agent:
    - Tier 1: Gemini LLM (Full natural language & multi-language)
    - Tier 2: Free Wikipedia Tool Retrieval (Dynamic facts without API keys)
    - Tier 3: Local 25-Bird Encyclopedia (Guaranteed 100% offline uptime)
    """

    @classmethod
    def get_bird_data(cls, bird_key: str) -> Dict[str, Any]:
        """Retrieves structured local knowledge for a given bird class."""
        if bird_key in BIRD_LOCAL_KNOWLEDGE:
            return BIRD_LOCAL_KNOWLEDGE[bird_key]
        for key, data in BIRD_LOCAL_KNOWLEDGE.items():
            if key.lower().replace("-", " ") == bird_key.lower().replace("-", " "):
                return data
            if data["name"].lower() == bird_key.lower():
                return data
        # Fallback default
        return {
            "name": bird_key.replace("-", " "),
            "scientific_name": "Aves",
            "description": f"A distinctive avian species recognized by the model.",
            "identification": "Classified based on visible morphological markers in the uploaded image.",
            "habitat": "Suitable environmental habitats and ecosystems.",
            "diet": "Natural wild diet appropriate for this species.",
            "fun_fact": "Birds are integral pollinators, pest controllers, and seed dispersers in global ecosystems."
        }

    @classmethod
    def generate_explanation(cls, bird_key: str, bird_name: str, scientific_name: str, language: str, confidence: float) -> str:
        """
        Generates an Explainable AI (XAI) breakdown.
        Attempts Gemini -> Falls back to Wikipedia Tool -> Falls back to Local Knowledge.
        """
        bird_data = cls.get_bird_data(bird_key)
        
        # 1. Check Tier 1: Gemini API
        client = get_gemini_model()
        if client is not None:
            try:
                prompt = f"""
You are BirdBot, an expert ornithologist and Explainable AI (XAI) assistant.
The fine-tuned MobileNetV3 deep learning model classified the uploaded image as:
- Bird Name: {bird_name}
- Scientific Name: {scientific_name}
- Confidence: {confidence:.1f}%

Generate a comprehensive, structured response in {language} containing:
1. 🔍 **Species Field Marks**: Describe field marks associated with {bird_name}, clearly labeling them as reference information, not evidence from this particular image.
2. 🐦 **About the Bird**: Concise overview of its physical traits and behavior.
3. 🌿 **Habitat & Range**: Where this bird lives and geographic distribution.
4. 🐛 **Diet & Feeding**: What it feeds on in the wild.
5. ⭐ **Fun Fact**: One fascinating, unique trivia fact.

The classifier provides a class score but no saliency map or pixel-level attribution. Do not claim that any feature caused the prediction or that you inspected specific image regions. Describe confidence as a model score, not a calibrated probability. Keep formatting clean and answer completely in {language}.
"""
                # Try google.genai (new SDK)
                if hasattr(client, "models"):
                    resp = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=prompt
                    )
                    if resp and resp.text:
                        return resp.text
                # Try legacy google.generativeai
                elif hasattr(client, "generate_content"):
                    resp = client.generate_content(prompt)
                    if resp and resp.text:
                        return resp.text
            except Exception as e:
                print(f"[Agent Warning] Gemini API failed: {e}. Falling back to Tier 2 (Wikipedia) + Tier 3 (Local Knowledge).")

        # 2. Tier 2 & 3: Free Tool Retrieval (Wikipedia) + Local Knowledge Base
        wiki_extract = web_agent_search(bird_name, scientific_name)
        about_text = wiki_extract if wiki_extract else bird_data.get("description", "")
        
        explanation = f"""### 🔍 Classification Report: **{bird_name}** (*{scientific_name}*)

1. **Model Score**: {confidence:.1f}% (not a calibrated probability)
    - **Species field marks** (reference information, not a pixel-level explanation): {bird_data.get('identification', '')}

2. **About the Species**:
   - {about_text}

3. **Habitat & Distribution**:
   - {bird_data.get('habitat', 'Found in diverse terrestrial and wetland ecosystems.')}

4. **Diet & Feeding Behavior**:
   - {bird_data.get('diet', 'Insects, seeds, and local natural food sources.')}

5. **⭐ Fun Fact**:
   - {bird_data.get('fun_fact', 'A wonderful specimen of avian biodiversity!')}

---
*🛡️ Powered by BirdBot Free Hybrid Agent (Live Tool Retrieval & Local Knowledge Base)*"""

        return _translate_response(explanation, language)

    @classmethod
    def answer_chat(cls, bird_name: str, scientific_name: str, message: str, language: str = "English") -> str:
        """
        Answers user follow-up questions intelligently using Tier 1 (Gemini),
        Tier 2 (Wikipedia Tool), or Tier 3 (Smart Local Query Matcher).
        """
        bird_data = cls.get_bird_data(bird_name)
        
        # Tier 1: Try Gemini API
        client = get_gemini_model()
        if client is not None:
            try:
                prompt = f"""
You are BirdBot, a helpful AI ornithologist assistant.
The conversation is about the bird: {bird_name} ({scientific_name}).
Known reference facts:
- Description: {bird_data.get('description', '')}
- Field marks: {bird_data.get('identification', '')}
- Habitat: {bird_data.get('habitat', '')}
- Diet: {bird_data.get('diet', '')}
- Fun fact: {bird_data.get('fun_fact', '')}
The user asks: "{message}"

Answer accurately, engagingly, and concisely in {language}.
Use the supplied facts when relevant. Do not claim to know what visual features caused the classifier's prediction.
"""
                if hasattr(client, "models"):
                    resp = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=prompt
                    )
                    if resp and resp.text:
                        return resp.text
                elif hasattr(client, "generate_content"):
                    resp = client.generate_content(prompt)
                    if resp and resp.text:
                        return resp.text
            except Exception as e:
                print(f"[Agent Warning] Gemini Chat failed: {e}. Falling back to Tool & Local Agent.")

        # Tier 2 & 3: question-aware local facts, then a clearly labeled overview.
        user_q = _translate_to_english(message, language).casefold().strip()
        local_answers = (
            (r"\b(diet|eat|food|feed|prey|hunting)\b", f"**Diet of the {bird_name}:** {bird_data.get('diet')}"),
            (r"\b(habitat|live|where|found|country|location|place|nest|range)\b", f"**Habitat & Range of the {bird_name}:** {bird_data.get('habitat')}"),
            (r"\b(fact|trivia|special|unique|interesting)\b", f"**Did you know?** {bird_data.get('fun_fact')}"),
            (r"\b(look|color|beak|feather|identify|size|wing|shape|markings)\b", f"**Field marks for {bird_name}:** {bird_data.get('identification')}"),
            (r"\b(name|scientific|called|species)\b", f"The common name is **{bird_name}** and its scientific classification is ***{scientific_name}***."),
            (r"\b(about|describe|description|overview|behavior)\b", f"**About the {bird_name}:** {bird_data.get('description')}"),
        )
        for pattern, answer in local_answers:
            if re.search(pattern, user_q):
                return _translate_response(answer, language)
        
        # A summary may provide context, but is not guaranteed to answer the exact question.
        wiki_text = web_agent_search(bird_name, scientific_name)
        if wiki_text:
            answer = f"I don't have a specific local answer for that question. Here is a general overview of the {bird_name}:\n{wiki_text}"
        else:
            answer = f"I don't have a specific local answer for that question. About the **{bird_name}** (*{scientific_name}*): {bird_data.get('description')} You can ask about its diet, habitat, field marks, or a fun fact."
        return _translate_response(answer, language)
