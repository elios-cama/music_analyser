# firebase_config.py
import firebase_admin
from firebase_admin import credentials, firestore
import os

def initialize_firebase():
    # Get the absolute path to the service account key
    current_dir = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(os.path.dirname(current_dir), 'service_account_key.json')
    
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)
    return firestore.client()

db = initialize_firebase()