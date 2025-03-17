
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

class IntentClassifier:
    def __init__(self, model_path=None):
        """
        Initialize the intent classifier

        Args:
            model_path (str, optional): Path to the trained model file
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Intent patterns (fallback if no model is provided)
        self.intent_patterns = {
            'create_ticket': [
                r'create.*ticket', r'log.*ticket', r'report.*issue',
                r'raise.*ticket', r'open.*ticket', r'submit.*ticket',
                r'new.*ticket', r'file.*ticket', r'register.*issue'
            ],
            'check_ticket_status': [
                r'status.*ticket', r'ticket.*status', r'check.*ticket',
                r'update.*ticket', r'progress.*ticket', r'ticket.*progress'
            ],
            'password_reset': [
                r'reset.*password', r'change.*password', r'forgot.*password',
                r'new.*password', r'password.*change', r'password.*reset'
            ],
            'access_issues': [
                r'access.*denied', r'can\'t.*login', r'unable.*login',
                r'login.*problem', r'account.*locked', r'access.*issue'
            ],
            'software_issues': [
                r'install.*software', r'update.*software', r'software.*problem',
                r'application.*error', r'program.*crash', r'software.*crash'
            ],
            'hardware_issues': [
                r'computer.*problem', r'hardware.*issue', r'printer.*issue',
                r'monitor.*problem', r'keyboard.*not working', r'mouse.*not working'
            ],
            'network_issues': [
                r'internet.*down', r'network.*problem', r'wifi.*issue',
                r'connection.*lost', r'can\'t.*connect', r'slow.*internet'
            ],
            'email_issues': [
                r'email.*problem', r'can\'t.*send.*email', r'email.*not working',
                r'outlook.*issue', r'missing.*email', r'email.*attachment'
            ],
            'general_info': [
                r'how.*to', r'what.*is', r'where.*find', r'info.*about',
                r'explain.*', r'tell.*me'
            ]
        }

        # Load model if provided
        self.model = None
        self.vectorizer = None
        if model_path:
            try:
                with open(model_path, 'rb') as f:
                    self.model, self.vectorizer = pickle.load(f)
            except Exception as e:
                print(f"Error loading model: {str(e)}")

    def preprocess_text(self, text):
        """
        Preprocess text for classification

        Args:
            text (str): Text to preprocess

        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        filtered_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]

        # Join tokens back into string
        return ' '.join(filtered_tokens)

    def classify_intent(self, text):
        """
        Classify the intent of the user's query

        Args:
            text (str): User's query

        Returns:
            str: Classified intent
        """
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)

        # Use trained model if available
        if self.model and self.vectorizer:
            # Transform text into feature vector
            features = self.vectorizer.transform([preprocessed_text])

            # Predict intent
            intent_idx = self.model.predict(features)[0]

            # Map index to intent label
            intent_labels = list(self.intent_patterns.keys())
            return intent_labels[intent_idx]

        # Fallback to pattern matching
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    return intent

        # Default intent
        return 'general_query'

    def extract_entities(self, text):
        """
        Extract entities from the user's query

        Args:
            text (str): User's query

        Returns:
            dict: Extracted entities
        """
        entities = {}

        # Extract ticket number
        ticket_match = re.search(r'ticket (?:number|#)?\s*(\d+)', text.lower())
        if ticket_match:
            entities['ticket_id'] = ticket_match.group(1)

        # Extract category
        category_patterns = {
            'hardware': [r'hardware', r'computer', r'printer', r'monitor', r'keyboard', r'mouse'],
            'software': [r'software', r'application', r'program', r'install', r'update'],
            'network': [r'network', r'internet', r'wifi', r'connection'],
            'email': [r'email', r'outlook', r'mail'],
            'access': [r'access', r'login', r'password', r'account']
        }

        for category, patterns in category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    entities['category'] = category
                    break
            if 'category' in entities:
                break

        # Extract priority
        priority_patterns = {
            'critical': [r'urgent', r'critical', r'emergency', r'asap'],
            'high': [r'high', r'important'],
            'normal': [r'normal', r'regular'],
            'low': [r'low', r'minor']
        }

        for priority, patterns in priority_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    entities['priority'] = priority
                    break
            if 'priority' in entities:
                break

        return entities