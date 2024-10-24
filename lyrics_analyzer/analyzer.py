import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np
from PIL import Image
from collections import Counter
from colorthief import ColorThief
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import matplotlib.backends.backend_pdf as pdf_backend
import string
import re
from unidecode import unidecode
from tqdm import tqdm
import os
import pickle
import matplotlib
matplotlib.use('Agg')  
import io
from firebase_config import db
from firebase_admin import firestore
from collections import Counter

class LyricsAnalyzer:
    def __init__(self, artist_name, client_access_token):
        self.artist_name = artist_name.strip()
        self.client_access_token = client_access_token
        self.base_url = "https://api.genius.com"
        self.headers = {"Authorization": f"Bearer {self.client_access_token}"}
        self.df = None
        self.checkpoint_file = f"{self.artist_name.lower().replace(' ', '_')}_lyrics_data.pkl"
        self._initialize_nltk()

    def clean_text(self, text):
        """
        Clean and process lyrics text:
        - Remove punctuation and special characters
        - Convert to lowercase
        - Remove stopwords
        - Remove single characters and numbers
        """
        if not text:
            return ""
            
        # Convert to lowercase and remove newlines
        text = text.lower()
        text = text.replace('\n', ' ')
        
        # Remove text within brackets and parentheses
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize and remove stopwords
        words = word_tokenize(text)
        words = [word for word in words 
                if word not in self.all_stopwords  # Remove stopwords
                and len(word) > 1  # Remove single characters
                and not word.isnumeric()]  # Remove numbers
        
        return ' '.join(words)
    


    def _initialize_nltk(self):
        print("Initializing NLTK resources...")
        for resource in ['punkt', 'stopwords', 'punkt_tab']:
            nltk.download(resource, quiet=True)
        
        self.french_stopwords = set(stopwords.words('french'))
        self.english_stopwords = set(stopwords.words('english'))
        
        # Expanded custom stopwords
        self.custom_stopwords = {
            # Your existing custom stopwords
            "ouai", "oui", "no", "nan", "non", "jsais", "ca", "ça", "jai", "cest", "jsuis", 
            "si", "jme", "tas", "ni", "jte", "ya", "eh", "oh", "comme", "plus", "tant", 
            "rien", "tout", "quand", "ouais", "trop", "là", "va", "dla", "où", "san", 
            "quon", "quil", "quelle", "qujdois", "davoir", "skandalize", "dun", "sen", 
            "car", "faire", "fais", "jirai", "faut", "ai", "as", "a", "avons", "avez", 
            "ont", "suis", "es", "est", "sommes", "êtes", "sont", "fait", "faisons", 
            "faites", "font", "peux", "peut", "pouvons", "pouvez", "peuvent", "dois", 
            "cette", "tous", "doit", "jveux", "jmets", "devons", "jfais", "yeah", "devez", 
            "doivent", "vais", "vas", "allons", "allez", "vont", "estce", "dit", "quelque", 
            "jetais",

            # Additional French contractions and informal words
            "j'ai", "t'as", "c'est", "d'un", "qu'on", "qu'il", "s'en", "d'une", "l'on",
            "n'est", "m'en", "c'était", "qu'elle", "d'être", "j'suis", "j'vais", "j'peux",
            "chuis", "jen", "jpeux", "menfou", "truc", "mec", "meuf", "chai", "chui", 
            "ben", "ptit", "ptite", "ptin", "jsais", "jvais",

            # French interjections and filler words
            "ah", "euh", "bah", "hein", "pfft", "pff", "rah", "wesh", "han", "huh", 
            "beh", "meh", "genre", "style", "grave", "vachement", "carrément", "direct",
            "bref", "voilà", "quoi", "enfin", "bon", "puis",

            # Additional English words
            "the", "and", "to", "of", "in", "it", "is", "that", "for", "on", "with", 
            "as", "at", "by", "from", "up", "about", "into", "over", "after",
            
            # English contractions and informal
            "ain't", "gonna", "gotta", "wanna", "ima", "imma", "lemme", "gimme",
            "dunno", "nah", "nope", "yep", "yup", "sup", "tryna",

            # Common song filler words
            "like", "just", "know", "get", "got", "feel", "think", "way",
        }
        
        self.all_stopwords = self.french_stopwords.union(self.english_stopwords).union(self.custom_stopwords)
        print("Initialization complete.")


    def analyze(self):
        checkpoint_data = self.load_checkpoint()
        if checkpoint_data:
            print("Loading data from checkpoint...")
            data = checkpoint_data
        else:
            data = self._fetch_and_process_data()
            self.save_checkpoint(data)

        self.df = self._create_dataframe(data)
        self._save_to_firebase()

    def _fetch_and_process_data(self):
        artist_id = self.get_artist_id()
        if not artist_id:
            raise ValueError(f"Could not find artist: {self.artist_name}")

        songs = self.get_artist_songs(artist_id)
        data = []
        print("Fetching song information and lyrics...")
        for song in tqdm(songs, desc="Processing songs"):
            song_info = self.get_song_info(song['id'])
            lyrics = self.scrape_lyrics(self.artist_name, song['title'])
            if lyrics:
                data.append({
                    "Song Name": song['title'],
                    "Song ID": song['id'],
                    "Album Name": song_info['album']['name'] if song_info['album'] else "<single>",
                    "Album ID": song_info['album']['id'] if song_info['album'] else None,
                    "Release Date": song_info['release_date'],
                    "Lyrics":self.clean_text(lyrics),
                    "url": song_info['song_art_image_url']
                })
        return data

    def _create_dataframe(self, data):
        print("Creating DataFrame and cleaning data...")
        df = pd.DataFrame(data)
        df = df.dropna(subset=['Lyrics', 'Album ID'])
        df = df.drop_duplicates(subset=['Song Name'])
        
        print("Cleaning lyrics and removing stopwords...")
        df['LyricsClean'] = df['Lyrics'].apply(self.text_cleansing).apply(self.clean_and_tokenize).apply(' '.join)
        return df

    # def _generate_visualizations(self):
    #     figures = []
    #     for album_name, group in self.df.groupby("Album Name"):
    #         lyrics = " ".join(group["LyricsClean"])
    #         album_cover_url = group["url"].iloc[0]
    #         figure = self.create_wordcloud_and_frequency_graph(lyrics, album_name, album_cover_url)
    #         figures.append((album_name, figure))
    #         plt.close(figure)  # Close the figure to free up memory

    #     pdf_filename = f'wordcloud_plots_{self.artist_name}.pdf'
    #     with pdf_backend.PdfPages(pdf_filename) as pdf:
    #         for album_name, fig in figures:
    #             pdf.savefig(fig)
    #             plt.close(fig)  # Close the figure after saving

    #     print(f"Analysis complete. PDF generated: {pdf_filename}")

    def get_artist_id(self):
        print(f"Searching for artist: {self.artist_name}")
        search_url = f"{self.base_url}/search"
        params = {'q': self.artist_name}
        try:
            response = requests.get(search_url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if 'response' not in data:
                print(f"Unexpected API response structure. Keys found: {data.keys()}")
                return None
            
            # Print all artists found for debugging
            print("\nArtists found in search:")
            for hit in data["response"]["hits"]:
                artist_name = hit["result"]["primary_artist"]["name"]
                artist_id = hit["result"]["primary_artist"]["id"]
                print(f"Name: {artist_name}, ID: {artist_id}")
            
            # More flexible matching
            for hit in data["response"]["hits"]:
                genius_name = hit["result"]["primary_artist"]["name"].lower().replace(' ', '')
                search_name = self.artist_name.lower().replace(' ', '')
                if genius_name == search_name:
                    return hit["result"]["primary_artist"]["id"]
            
            print(f"Artist '{self.artist_name}' not found in search results.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error making request to Genius API: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected data structure in API response: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def get_artist_songs(self, artist_id):
        print(f"Fetching songs for artist ID: {artist_id}")
        songs = []
        next_page = 1
        with tqdm(desc="Fetching songs", unit="page") as pbar:
            while next_page:
                url = f"{self.base_url}/artists/{artist_id}/songs"
                params = {'page': next_page, 'per_page': 50}
                response = requests.get(url, params=params, headers=self.headers)
                data = response.json()
                page_songs = data['response']['songs']
                songs.extend([song for song in page_songs if song["primary_artist"]["id"] == artist_id])
                next_page = data['response']['next_page']
                pbar.update(1)
        print(f"Found {len(songs)} songs.")
        return songs

    def get_song_info(self, song_id):
        url = f"{self.base_url}/songs/{song_id}"
        response = requests.get(url, headers=self.headers)
        return response.json()["response"]["song"]

    def scrape_lyrics(self, artist, track):
        print(f"Scraping lyrics for: {artist} - {track}")
        artist = self.clean_url_string(artist)
        track = self.clean_url_string(track)
        url = f"https://genius.com/{artist}-{track}-lyrics"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            lyrics_div = soup.find('div', class_='Lyrics__Container-sc-1ynbvzw-1')
            if lyrics_div:
                return lyrics_div.get_text('\n').strip()
        return None
    
    def _save_to_firebase(self):
        """Process and save data to Firebase"""
        try:
            # First create/update artist document
            artists_ref = db.collection('artists')
            artist_doc = artists_ref.document(self.artist_name)
            artist_doc.set({
                'name': self.artist_name,
                'last_updated': firestore.SERVER_TIMESTAMP
            }, merge=True)
            
            # Process each album
            for album_name, group in self.df.groupby("Album Name"):
                # Get album data
                album_data = {
                    'album': album_name,
                    'album_id': group['Album ID'].iloc[0],
                    'release_date': group['Release Date'].iloc[0]
                }
                
                # Get album cover and color
                album_cover_url = group['url'].iloc[0]
                color_thief = ColorThief(requests.get(album_cover_url, stream=True).raw)
                dominant_color = color_thief.get_color(quality=1)
                dominant_color_hex = self.rgb2hex(dominant_color[0], dominant_color[1], dominant_color[2])
                
                # Calculate word frequencies
                lyrics = " ".join(group["LyricsClean"])
                word_counts = Counter(lyrics.split())
                
                # Save to Firebase collections
                self._save_album_to_firebase(
                    album_data=album_data,
                    word_counts=word_counts,
                    album_cover_url=album_cover_url,
                    dominant_color=dominant_color_hex
                )
                
            print(f"All data processed and saved for {self.artist_name}")
            
        except Exception as e:
            print(f"Error in saving to Firebase: {str(e)}")

    def _save_album_to_firebase(self, album_data, word_counts, album_cover_url, dominant_color):
        """Save album data to Firebase collections"""
        try:
            # Reference to collections
            albums_ref = db.collection('albums')
            word_stats_ref = db.collection('word_stats')
            
            # Create album document
            album_doc = albums_ref.document(f"{self.artist_name}_{album_data['album']}")
            album_doc.set({
                'name': album_data['album'],
                'artist_id': self.artist_name,  # Reference to artist
                'genius_album_id': album_data['album_id'],
                'cover_url': album_cover_url,
                'dominant_color': dominant_color,
                'release_date': album_data['release_date']
            })
            
            # Save word statistics
            word_stats_doc = word_stats_ref.document(f"{self.artist_name}_{album_data['album']}")
            word_stats_doc.set({
                'album_id': album_doc.id,
                'word_frequencies': dict(word_counts.most_common(50)),  # Save top 50 words
                'total_words': sum(word_counts.values()),
                'unique_words': len(word_counts)
            })
            
            print(f"Data saved for album: {album_data['album']}")
            
        except Exception as e:
            print(f"Error saving to Firebase: {str(e)}")

    @staticmethod
    def rgb2hex(r, g, b):
        """Convert RGB to hex color"""
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def clean_url_string(s):
        s = unidecode(s.lower().replace(' ', '-'))
        s = re.sub(r"[^\w\s-]", '', s)
        return s.strip('-')

    @staticmethod
    def text_cleansing(text):
        text = re.sub(r"\[.*?\]", "", text)
        text = text.replace("\n", " ").lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def clean_and_tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        cleaned_tokens = [word for word in tokens if word not in self.all_stopwords and len(word) > 1]
        return cleaned_tokens

    @staticmethod
    def rgb2hex(r, g, b):
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    # def create_wordcloud_and_frequency_graph(self, text, album_name, album_cover_url):
    #     plt.figure(figsize=(12, 5))
        
    #     # Create word cloud
    #     album_cover = Image.open(requests.get(album_cover_url, stream=True).raw).convert('RGB')
    #     color_thief = ColorThief(io.BytesIO(requests.get(album_cover_url, stream=True).content))
    #     dominant_color = color_thief.get_color(quality=1)
    #     dominant_color = self.rgb2hex(*dominant_color)
    #     mask = np.array(album_cover)
    #     image_colors = ImageColorGenerator(mask)
    #     wordcloud = WordCloud(width=400, height=400, background_color="white", mask=mask, collocations=False).generate(text)
        
    #     # Calculate word frequencies
    #     word_counts = Counter(text.split())
    #     most_common_words = word_counts.most_common(15)
    #     words, counts = zip(*most_common_words)
        
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(mask, cmap=plt.cm.gray, interpolation="bilinear")
    #     plt.axis("off")
    #     plt.title(f"Album Cover for {album_name}")
        
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
    #     plt.axis("off")
    #     plt.title(f"Word Cloud for {album_name}")
        
    #     plt.subplot(1, 3, 3)
    #     plt.barh(words, counts, color=dominant_color)
    #     plt.xlabel("Frequency")
    #     plt.title(f"Top 15 Most Frequent Words for {album_name}")
        
    #     plt.tight_layout()
        
    #     return plt.gcf()

    def save_checkpoint(self, data):
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Checkpoint saved to {self.checkpoint_file}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None

if __name__ == "__main__":
    ARTIST_NAME = input("Enter the artist name: ")
    CLIENT_ACCESS_TOKEN = input("Enter your Genius API client access token: ")
    
    analyzer = LyricsAnalyzer(ARTIST_NAME, CLIENT_ACCESS_TOKEN)
    analyzer.analyze()