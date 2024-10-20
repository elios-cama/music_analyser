### setup.py
python
from setuptools import setup, find_packages

setup(
    name="lyrics-analyzer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'requests',
        'beautifulsoup4',
        'pandas',
        'nltk',
        'wordcloud',
        'numpy',
        'Pillow',
        'colorthief',
        'matplotlib',
        'unidecode',
    ],
)