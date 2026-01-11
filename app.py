"""
Magic Card Organizer
A tool to photograph Magic: The Gathering cards, parse their names via VLM,
validate against Scryfall API, and export to CSV.
"""

import base64
import csv
import io
import json
import logging
import os
import re
import sys
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
SCRYFALL_API = "https://api.scryfall.com"
VLM_PROVIDER = os.environ.get("VLM_PROVIDER", "openai").lower()  # openai, anthropic, or deepseek
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# VLM Prompt for card extraction
CARD_EXTRACTION_PROMPT = """You are a Magic: The Gathering card recognition expert.

Look at this image and identify ALL Magic: The Gathering cards visible in the photo.

For each card you can see, extract ONLY the card name (the text at the very top of each card, to the left of the mana cost).

Rules:
- Return ONLY the card names, one per line
- Do NOT include mana costs, card types, rules text, or any other text
- If you can partially see a card but can read the name, include it
- If a card name is unclear or unreadable, skip it
- If there are no Magic cards in the image, respond with "NO_CARDS_FOUND"

Example output format:
Lightning Bolt
Counterspell
Black Lotus

Now analyze the image and list all card names you can see:"""


def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    # Convert to RGB if necessary (handles RGBA, P mode, etc.)
    if image.mode in ('RGBA', 'P', 'LA'):
        image = image.convert('RGB')
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def extract_card_names_openai(image_base64):
    """Extract card names using OpenAI GPT-4 Vision."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    logger.info("Using OpenAI GPT-4 Vision for card extraction")
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "o4-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CARD_EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
        },
        timeout=60
    )
    
    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
        raise Exception(f"OpenAI API error: {response.status_code}")
    
    result = response.json()
    return result["choices"][0]["message"]["content"]


def extract_card_names_anthropic(image_base64):
    """Extract card names using Anthropic Claude Vision."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    logger.info("Using Anthropic Claude for card extraction")
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": CARD_EXTRACTION_PROMPT
                        }
                    ]
                }
            ]
        },
        timeout=60
    )
    
    if response.status_code != 200:
        logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
        raise Exception(f"Anthropic API error: {response.status_code}")
    
    result = response.json()
    return result["content"][0]["text"]


def extract_card_names_deepseek(image_base64):
    """Extract card names using DeepSeek Vision."""
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
    
    logger.info("Using DeepSeek Vision for card extraction")
    
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CARD_EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        },
        timeout=60
    )
    
    if response.status_code != 200:
        logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
        raise Exception(f"DeepSeek API error: {response.status_code}")
    
    result = response.json()
    return result["choices"][0]["message"]["content"]


def extract_card_names_from_image(image):
    """
    Extract card names from an image using the configured VLM provider.
    Returns a list of card names.
    """
    logger.info(f"Extracting card names using VLM provider: {VLM_PROVIDER}")
    
    # Convert image to base64
    image_base64 = image_to_base64(image)
    logger.debug(f"Image converted to base64 ({len(image_base64)} chars)")
    
    # Call the appropriate VLM provider
    if VLM_PROVIDER == "openai":
        raw_response = extract_card_names_openai(image_base64)
    elif VLM_PROVIDER == "anthropic":
        raw_response = extract_card_names_anthropic(image_base64)
    elif VLM_PROVIDER == "deepseek":
        raw_response = extract_card_names_deepseek(image_base64)
    else:
        raise ValueError(f"Unknown VLM provider: {VLM_PROVIDER}")
    
    logger.info(f"VLM response received")
    logger.debug(f"Raw VLM response: {raw_response}")
    
    # Parse the response into card names
    if "NO_CARDS_FOUND" in raw_response:
        logger.info("VLM detected no Magic cards in the image")
        return []
    
    # Extract card names (one per line, clean up)
    card_names = []
    for line in raw_response.strip().split('\n'):
        name = line.strip()
        # Skip empty lines and common non-card-name patterns
        if not name:
            continue
        if name.lower().startswith(('here', 'the card', 'i can see', 'card name', '-', '•', '*', '1.', '2.')):
            continue
        # Remove leading numbers/bullets if present
        name = re.sub(r'^[\d\.\-\*\•\–]+\s*', '', name)
        # Remove trailing punctuation
        name = name.rstrip('.,;:')
        if len(name) >= 2:
            card_names.append(name)
    
    logger.info(f"Parsed {len(card_names)} card names from VLM response")
    for name in card_names:
        logger.debug(f"  Card name: '{name}'")
    
    return card_names


def validate_card_with_scryfall(card_name):
    """
    Validate a card name against Scryfall's API.
    Returns the official card data if found, None otherwise.
    """
    logger.info(f"Validating card against Scryfall: '{card_name}'")
    
    try:
        # Use fuzzy search to be more forgiving of OCR errors
        response = requests.get(
            f"{SCRYFALL_API}/cards/named",
            params={"fuzzy": card_name},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            official_name = data.get("name")
            logger.info(f"✓ Card validated: '{card_name}' → '{official_name}'")
            return {
                "name": official_name,
                "set": data.get("set_name"),
                "set_code": data.get("set"),
                "type": data.get("type_line"),
                "mana_cost": data.get("mana_cost", ""),
                "rarity": data.get("rarity"),
                "oracle_text": data.get("oracle_text", ""),
                "image_url": data.get("image_uris", {}).get("normal", ""),
                "scryfall_uri": data.get("scryfall_uri", ""),
                "validated": True
            }
        elif response.status_code == 404:
            logger.debug(f"Card not found directly, trying autocomplete for '{card_name}'")
            # Try autocomplete for partial matches
            autocomplete_response = requests.get(
                f"{SCRYFALL_API}/cards/autocomplete",
                params={"q": card_name},
                timeout=10
            )
            if autocomplete_response.status_code == 200:
                suggestions = autocomplete_response.json().get("data", [])
                if suggestions:
                    logger.debug(f"Autocomplete suggestion: '{suggestions[0]}'")
                    # Try the first suggestion
                    return validate_card_with_scryfall(suggestions[0])
            logger.warning(f"✗ Card not found: '{card_name}'")
            return None
        else:
            logger.warning(f"Scryfall API returned status {response.status_code} for '{card_name}'")
            return None
            
    except requests.RequestException as e:
        logger.error(f"Scryfall API error for '{card_name}': {e}")
        return None


def process_cards(image):
    """
    Main processing pipeline: VLM -> Parse -> Validate
    """
    logger.info("=" * 50)
    logger.info("Starting card processing pipeline")
    
    # Extract card names using VLM
    card_names = extract_card_names_from_image(image)
    
    if not card_names:
        logger.info("No card names extracted from image")
        return {
            "raw_names": [],
            "validated_cards": [],
            "unvalidated_names": []
        }
    
    # Validate each card name against Scryfall
    validated_cards = []
    unvalidated_names = []
    
    logger.info(f"Validating {len(card_names)} card names against Scryfall...")
    for name in card_names:
        card_data = validate_card_with_scryfall(name)
        if card_data:
            validated_cards.append(card_data)
        else:
            unvalidated_names.append(name)
    
    logger.info("=" * 50)
    logger.info(f"Pipeline complete: {len(validated_cards)} validated, {len(unvalidated_names)} unrecognized")
    
    return {
        "raw_names": card_names,
        "validated_cards": validated_cards,
        "unvalidated_names": unvalidated_names
    }


def generate_csv(cards):
    """
    Generate a CSV file from validated card data.
    """
    output = io.StringIO()
    fieldnames = ['name', 'set', 'set_code', 'type', 'mana_cost', 'rarity']
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for card in cards:
        writer.writerow({
            'name': card.get('name', ''),
            'set': card.get('set', ''),
            'set_code': card.get('set_code', ''),
            'type': card.get('type', ''),
            'mana_cost': card.get('mana_cost', ''),
            'rarity': card.get('rarity', '')
        })
    
    output.seek(0)
    return output.getvalue()


# Store processed results temporarily (in production, use a proper session/cache)
processed_results = {}


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Return current VLM provider configuration."""
    return jsonify({
        "vlm_provider": VLM_PROVIDER,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_anthropic_key": bool(ANTHROPIC_API_KEY),
        "has_deepseek_key": bool(DEEPSEEK_API_KEY)
    })


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    Handle image upload and process cards.
    """
    logger.info(f"POST /api/upload - Image upload request from {request.remote_addr}")
    
    if 'image' not in request.files:
        logger.warning("Upload rejected: No image in request")
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        logger.warning("Upload rejected: Empty filename")
        return jsonify({"error": "No image selected"}), 400
    
    logger.info(f"Processing uploaded file: {file.filename}")
    
    try:
        # Open and process the image
        image = Image.open(file.stream)
        logger.info(f"Image opened: {image.size[0]}x{image.size[1]} {image.mode}")
        
        # Process cards
        results = process_cards(image)
        
        # Store results for CSV export
        session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        processed_results[session_id] = results
        logger.info(f"Session created: {session_id}")
        
        return jsonify({
            "session_id": session_id,
            "raw_names": results["raw_names"],
            "validated_cards": results["validated_cards"],
            "unvalidated_names": results["unvalidated_names"]
        })
        
    except ValueError as e:
        # API key not set
        logger.error(f"Configuration error: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error processing upload: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/manual-add', methods=['POST'])
def manual_add_card():
    """
    Manually add a card by name and validate against Scryfall.
    """
    data = request.get_json()
    
    if not data or 'card_name' not in data:
        logger.warning("Manual add rejected: No card name provided")
        return jsonify({"error": "No card name provided"}), 400
    
    card_name = data['card_name']
    session_id = data.get('session_id')
    
    logger.info(f"POST /api/manual-add - Adding card: '{card_name}'")
    
    card_data = validate_card_with_scryfall(card_name)
    
    if card_data:
        # Add to session if provided
        if session_id and session_id in processed_results:
            processed_results[session_id]["validated_cards"].append(card_data)
            logger.info(f"Card added to session {session_id}")
        
        return jsonify({"card": card_data})
    else:
        logger.warning(f"Manual add failed: Card '{card_name}' not found")
        return jsonify({"error": f"Card '{card_name}' not found"}), 404


@app.route('/api/export/<session_id>', methods=['GET'])
def export_csv(session_id):
    """
    Export validated cards as CSV.
    """
    logger.info(f"GET /api/export/{session_id} - CSV export request")
    
    if session_id not in processed_results:
        logger.warning(f"Export failed: Session {session_id} not found")
        return jsonify({"error": "Session not found"}), 404
    
    cards = processed_results[session_id]["validated_cards"]
    logger.info(f"Exporting {len(cards)} cards to CSV")
    
    csv_content = generate_csv(cards)
    
    # Create response with CSV file
    output = io.BytesIO()
    output.write(csv_content.encode('utf-8'))
    output.seek(0)
    
    logger.info(f"CSV export complete: magic_cards_{session_id}.csv")
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'magic_cards_{session_id}.csv'
    )


@app.route('/api/search', methods=['GET'])
def search_card():
    """
    Search for a card by name using Scryfall autocomplete.
    """
    query = request.args.get('q', '')
    
    if len(query) < 2:
        return jsonify({"suggestions": []})
    
    logger.debug(f"GET /api/search - Query: '{query}'")
    
    try:
        response = requests.get(
            f"{SCRYFALL_API}/cards/autocomplete",
            params={"q": query},
            timeout=10
        )
        
        if response.status_code == 200:
            suggestions = response.json().get("data", [])
            logger.debug(f"Search returned {len(suggestions)} suggestions")
            return jsonify({"suggestions": suggestions[:10]})
        else:
            logger.warning(f"Scryfall search returned status {response.status_code}")
            return jsonify({"suggestions": []})
            
    except requests.RequestException as e:
        logger.error(f"Search error: {e}")
        return jsonify({"suggestions": []})


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("  Arcane Registry - MTG Card Scanner")
    logger.info("=" * 50)
    logger.info(f"VLM Provider: {VLM_PROVIDER}")
    logger.info(f"OpenAI API Key: {'configured' if OPENAI_API_KEY else 'NOT SET'}")
    logger.info(f"Anthropic API Key: {'configured' if ANTHROPIC_API_KEY else 'NOT SET'}")
    logger.info(f"DeepSeek API Key: {'configured' if DEEPSEEK_API_KEY else 'NOT SET'}")
    logger.info("")
    logger.info("Starting server on http://0.0.0.0:5050")
    logger.info("Press Ctrl+C to stop")
    logger.info("")
    
    app.run(debug=True, host='0.0.0.0', port=5050)
