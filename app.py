# app.py
from flask import Flask, render_template, request, jsonify, session
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch
import json
import logging
from dotenv import load_dotenv
import re

# Local imports
from mantis_api import MantisBTAPI
from intent_classifier import IntentClassifier

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default-secret-key')

# Load the trained model
try:
    tokenizer = AutoTokenizer.from_pretrained("./helpdesk_bot_model")
    model = AutoModelForSeq2SeqLM.from_pretrained("./helpdesk_bot_model")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    # Fallback to a smaller model if the custom model fails to load
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    # model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    tokenizer.pad_token = tokenizer.eos_token

# Initialize Mantis BT API
mantis_api = MantisBTAPI(
    api_url=os.environ.get('MANTIS_API_URL', 'https://your-mantis-instance/api/rest'),
    api_token=os.environ.get('MANTIS_API_TOKEN', '')
)

# Initialize Intent Classifier
intent_classifier = IntentClassifier("./intent_classifier_model.pkl")

# Function to generate response from the model
def generate_response(query):
    try:
        inputs = tokenizer.encode(query, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I'm sorry, I couldn't process your request. Please try again or contact the IT department directly."

# Function to handle ticket creation
def handle_ticket_creation(text, user_id=None):
    # Extract entities from the query
    entities = intent_classifier.extract_entities(text)

    # Default values
    category = entities.get('category', 'General')
    priority = entities.get('priority', 'normal')

    # Extract summary and description
    # Remove common phrases used to request ticket creation
    summary = re.sub(r'(create|log|open|submit|raise)(\s+a)?\s+ticket(\s+for)?', '', text, flags=re.IGNORECASE).strip()
    if not summary:
        summary = "IT Support Request"

    description = f"Ticket created by the IT Helpdesk chatbot.\n\nOriginal query: {text}"
    if user_id:
        description += f"\n\nRequested by: {user_id}"

    # Create ticket via Mantis BT API
    result = mantis_api.create_ticket(
        summary=summary,
        description=description,
        category=category,
        priority=priority
    )

    if result.get('success'):
        ticket_id = result['issue']['issue']['id']
        return f"I've created ticket #{ticket_id} for you. The IT team will look into it shortly. You can check the status of your ticket anytime by asking me 'What's the status of ticket #{ticket_id}?'"
    else:
        logger.error(f"Failed to create ticket: {result.get('error')}")
        return f"I couldn't create a ticket due to a technical issue. Please try again later or contact the IT helpdesk directly at extension 5555."

# Function to handle ticket status queries
def handle_ticket_status(text):
    # Extract ticket ID from the query
    ticket_match = re.search(r'ticket (?:number|#)?\s*(\d+)', text.lower())

    if not ticket_match:
        return "I couldn't find a ticket number in your query. Please provide a ticket number, for example: 'What's the status of ticket #123?'"

    ticket_id = ticket_match.group(1)

    # Get ticket status via Mantis BT API
    result = mantis_api.get_ticket(ticket_id)

    if result.get('success'):
        ticket = result['ticket']
        response = f"Ticket #{ticket['id']} status: {ticket['status']}.\n"
        response += f"Summary: {ticket['summary']}.\n"
        response += f"Category: {ticket['category']}.\n"
        response += f"Priority: {ticket['priority']}.\n"
        response += f"Assigned to: {ticket['assigned_to']}.\n"
        response += f"Last updated: {ticket['updated_at']}."
        return response
    else:
        logger.error(f"Failed to get ticket status: {result.get('error')}")
        return f"I couldn't retrieve information for ticket #{ticket_id}. It may not exist or there might be a technical issue. Please try again later or contact the IT helpdesk directly."

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('message', '')
        user_id = session.get('user_id', None)

        # Log incoming request
        logger.info(f"Received query: {query}")

        # Classify the intent of the query
        intent = intent_classifier.classify_intent(query)
        logger.info(f"Classified intent: {intent}")

        # Handle intent accordingly
        if intent == 'create_ticket':
            response = handle_ticket_creation(query, user_id)
        elif intent == 'check_ticket_status':
            response = handle_ticket_status(query)
        else:
            # For general queries, use the trained model
            response = generate_response(query)

        # Log the response
        logger.info(f"Response: {response}")

        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"response": "I encountered an error while processing your request. Please try again later."})

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

# API documentation endpoint
@app.route('/api/docs', methods=['GET'])
def api_docs():
    return render_template('api_docs.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)