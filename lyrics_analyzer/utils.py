import re
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords

def clean_url_string(s):
    s = unidecode(s.lower().replace(' ', '-'))
    s = re.sub(r"[^\w\s-]", '', s)
    return s.strip('-')

def text_cleansing(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("\n", " ").lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def remove_stopwords(text, language='english'):
    stop_words = set(stopwords.words(language))
    words = nltk.word_tokenize(text)
    return " ".join([word for word in words if word.lower() not in stop_words and len(word) > 1])

def rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)