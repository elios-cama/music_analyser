from lyrics_analyzer import LyricsAnalyzer

def main():
    artist_name = input("Enter the artist name: ")
    client_access_token = input("Enter your Genius API client access token: ")
    
    analyzer = LyricsAnalyzer(artist_name, client_access_token)
    analyzer.analyze()

if __name__ == "__main__":
    main()