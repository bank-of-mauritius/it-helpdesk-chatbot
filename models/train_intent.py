#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train an intent classifier for IT helpdesk queries using scikit-learn.
"""

import argparse
import json
import logging
import os
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Preprocess text for intent classification."""
    # Convert to lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Remove stopwords and lemmatize
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    # Join tokens back into string
    return ' '.join(filtered_tokens)

def prepare_data(data_path, intent_field, text_field):
    """Prepare data for intent classification."""
    intents = []
    texts = []

    logger.info(f"Loading dataset from {data_path}")
    with open(data_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                intents.append(item[intent_field])
                texts.append(item[text_field])
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line: {line}")
            except KeyError as e:
                logger.warning(f"Missing key in JSON: {e}")

    logger.info(f"Loaded {len(texts)} examples")

    # Preprocess texts
    preprocessed_texts = [preprocess_text(text) for text in texts]

    return preprocessed_texts, intents

def train_intent_classifier(args):
    """Train an intent classifier."""
    # Prepare data
    X, y = prepare_data(args.data_path, args.intent_field, args.text_field)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=2
    )

    # Create SVM classifier
    classifier = LinearSVC(C=args.C, random_state=args.seed)

    # Fit classifier
    logger.info("Training intent classifier...")
    X_train_vec = vectorizer.fit_transform(X_train)
    classifier.fit(X_train_vec, y_train)

    # Evaluate classifier
    X_test_vec = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_vec)

    # Print classification report
    logger.info("Classification report:")
    report = classification_report(y_test, y_pred)
    logger.info("\n" + report)

    # Save classifier and vectorizer
    logger.info(f"Saving model to {args.output_path}")
    with open(args.output_path, 'wb') as f:
        pickle.dump((classifier, vectorizer), f)

    logger.info("Training complete")

def main():
    parser = argparse.ArgumentParser(description="Train an intent classifier for helpdesk queries")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL dataset")
    parser.add_argument("--intent_field", type=str, default="intent", help="Field name for intent in the dataset")
    parser.add_argument("--text_field", type=str, default="prompt", help="Field name for text in the dataset")
    parser.add_argument("--output_path", type=str, default="./intent_classifier_model.pkl", help="Output path for the model")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--max_features", type=int, default=5000, help="Maximum number of features for TF-IDF")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter for SVM")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_intent_classifier(args)

if __name__ == "__main__":
    main()