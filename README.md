# IT Helpdesk Chatbot

An intelligent chatbot application designed to assist with IT helpdesk operations. The chatbot can understand user queries, create and track tickets in MantisBT, and provide automated assistance for common IT issues.

## Features

- **Intent Classification**: Automatically categorizes user queries into intents such as ticket creation, status checking, password resets, etc.
- **Entity Extraction**: Identifies key information in user messages like ticket numbers, issue categories, and priorities.
- **Ticket Management**: Creates and monitors tickets in MantisBT ticketing system.
- **NLP-Powered Responses**: Uses a fine-tuned language model to generate contextually appropriate responses.
- **Docker Support**: Easy deployment with Docker and Docker Compose.

## Project Structure

```
it-helpdesk-chatbot/
├── app.py                      # Main Flask application
├── intent_classifier.py        # Intent classification module
├── mantis_api.py               # MantisBT API integration
├── train_intent.py             # Script to train intent classifier
├── train_model.py              # Script to train response generator model
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── .env                        # Environment variables (create this)
├── templates/                  # HTML templates (create this)
│   ├── index.html              # Chat interface
│   └── api_docs.html           # API documentation
├── logs/                       # Log files
└── helpdesk_bot_model/         # Trained model files
```

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- MantisBT instance with API access
- GPU (optional, for faster model training and inference)

## Installation

### Option 1: Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/it-helpdesk-chatbot.git
   cd it-helpdesk-chatbot
   ```

2. Create a `.env` file with your configuration:
   ```
   MANTIS_API_URL=https://your-mantis-instance/api/rest
   MANTIS_API_TOKEN=your_api_token
   SECRET_KEY=your_flask_secret_key
   FLASK_DEBUG=False
   ```

3. Build and start the container:
   ```bash
   docker-compose up -d
   ```

4. Access the application at http://localhost:5000

### Option 2: Manual Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/it-helpdesk-chatbot.git
   cd it-helpdesk-chatbot
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your configuration (same as above).

4. Run the application:
   ```bash
   python app.py
   ```

## Missing Components

Before running the application, you need to:

1. Create HTML templates:
    - Create a `templates` directory
    - Add `index.html` with your chat interface
    - Add `api_docs.html` with API documentation

2. Train or provide models:
    - Follow the training instructions below, or
    - Place pre-trained models in the `helpdesk_bot_model` directory

## Training Models

### Intent Classifier

1. Prepare a JSONL dataset with intents and example queries:
   ```json
   {"intent": "create_ticket", "prompt": "I need to create a ticket for my broken monitor"},
   {"intent": "check_ticket_status", "prompt": "What's the status of ticket #123?"}
   ```

2. Train the intent classifier:
   ```bash
   python train_intent.py --data_path your_dataset.jsonl --output_path intent_classifier_model.pkl
   ```

### Response Generator

1. Prepare a JSONL dataset with example conversations:
   ```json
   {"body": "My computer is running slow", "answer": "I can help you with that. Let's check a few things. First..."},
   {"body": "I need to reset my password", "answer": "I'd be happy to help you reset your password. Please..."}
   ```

2. Train the response model:
   ```bash
   python train_model.py --data_path your_dataset.jsonl --output_dir helpdesk_bot_model
   ```

## API Endpoints

- **GET /**: Main chat interface
- **POST /api/chat**: Send messages to the chatbot
    - Request body: `{"message": "Your message here"}`
    - Response: `{"response": "Bot response"}`
- **GET /health**: Health check endpoint
- **GET /api/docs**: API documentation

## Configuration Options

The application is configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| MANTIS_API_URL | MantisBT API URL | https://your-mantis-instance/api/rest |
| MANTIS_API_TOKEN | MantisBT API token | - |
| SECRET_KEY | Flask secret key | default-secret-key |
| FLASK_DEBUG | Debug mode | False |
| PORT | HTTP port | 5000 |

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (add tests for your features)
5. Submit a pull request

## License

[Specify your license here]

## Acknowledgements

- Hugging Face Transformers library
- Flask framework
- MantisBT team