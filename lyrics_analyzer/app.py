from flask import Flask, request, jsonify, send_file
from analyzer import LyricsAnalyzer
import os
import traceback
from firebase_config import db
from firebase_admin import firestore
from collections import Counter

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Lyrics Analyzer API!"

@app.route('/analyze', methods=['POST'])
def start_analysis():
    try:
        data = request.get_json()
        artist_name = data.get('artist_name')
        genius_token = data.get('genius_token')
        
        if not all([artist_name, genius_token]):
            return {'error': 'Missing required parameters'}, 400
        
        # Initialize and run analysis
        analyzer = LyricsAnalyzer(artist_name, genius_token)
        analyzer.analyze()  # This now saves to Firebase instead of generating PDF
        
        return {
            'status': 'success',
            'message': f'Analysis complete for {artist_name}. Data saved to Firebase.'
        }, 200
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")  # Log the actual error
        return {
            'error': 'Analysis failed',
            'details': str(e)
        }, 500

@app.route('/artist/<artist_name>', methods=['GET'])
def get_artist_data(artist_name):
    try:
        # Get artist document
        artist_doc = db.collection('artists').document(artist_name).get()
        
        if not artist_doc.exists:
            return jsonify({"error": "Artist not found"}), 404
            
        # Get all albums for this artist
        albums = db.collection('albums')\
            .where('artist_id', '==', artist_name)\
            .stream()
            
        # Get word stats for each album
        artist_data = {
            'artist': artist_doc.to_dict(),
            'albums': {}
        }
        
        for album in albums:
            album_data = album.to_dict()
            # Get word stats for this album
            word_stats = db.collection('word_stats')\
                .document(f"{artist_name}_{album_data['name']}")\
                .get()
                
            artist_data['albums'][album.id] = {
                **album_data,
                'word_stats': word_stats.to_dict() if word_stats.exists else {}
            }
            
        return jsonify(artist_data), 200
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return jsonify({"error": "Failed to fetch data"}), 500

# You might also want specific endpoints for albums or word stats
@app.route('/artist/<artist_name>/albums', methods=['GET'])
def get_artist_albums(artist_name):
    try:
        albums = db.collection('albums')\
            .where('artist_id', '==', artist_name)\
            .stream()
            
        albums_data = {album.id: album.to_dict() for album in albums}
        
        if not albums_data:
            return jsonify({"error": "No albums found"}), 404
            
        return jsonify(albums_data), 200
        
    except Exception as e:
        print(f"Error fetching albums: {str(e)}")
        return jsonify({"error": "Failed to fetch albums"}), 500

@app.route('/album/<artist_name>/<album_name>/words', methods=['GET'])
def get_album_words(artist_name, album_name):
    try:
        word_stats = db.collection('word_stats')\
            .document(f"{artist_name}_{album_name}")\
            .get()
            
        if not word_stats.exists:
            return jsonify({"error": "Word stats not found"}), 404
            
        return jsonify(word_stats.to_dict()), 200
        
    except Exception as e:
        print(f"Error fetching word stats: {str(e)}")
        return jsonify({"error": "Failed to fetch word stats"}), 500

if __name__ == '__main__':
    app.run(debug=True)