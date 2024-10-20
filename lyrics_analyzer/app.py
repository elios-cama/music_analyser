from flask import Flask, request, jsonify, send_file
from analyzer import LyricsAnalyzer
import os
import traceback

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Lyrics Analyzer API!"

@app.route('/analyze', methods=['POST'])
def start_analysis():
    try:
        data = request.json
        if not data or 'artist_name' not in data or 'genius_token' not in data:
            return jsonify({"error": "Missing artist_name or genius_token"}), 400

        artist_name = data['artist_name']
        genius_token = data['genius_token']

        analyzer = LyricsAnalyzer(artist_name, genius_token)
        analyzer.analyze()

        pdf_filename = f'wordcloud_plots_{artist_name}.pdf'
        if os.path.exists(pdf_filename):
            return jsonify({"message": "Analysis complete", "pdf_available": True}), 200
        else:
            return jsonify({"error": "Analysis failed to generate PDF"}), 500

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/result/<artist_name>', methods=['GET'])
def get_result(artist_name):
    pdf_filename = f'wordcloud_plots_{artist_name}.pdf'
    if os.path.exists(pdf_filename):
        return send_file(pdf_filename, as_attachment=True)
    else:
        return jsonify({"error": "PDF file not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)