import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load environment variables
load_dotenv()

# Spotify API setup
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.getenv("SPOTIPY_CLIENT_ID"),
                                                             client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")))

# List of known tracks to test
test_tracks = [
    {"track_name": "Shape of You", "artist": "Ed Sheeran"},  # Shape of You by Ed Sheeran
    {"track_name": "Blinding Lights", "artist": "The Weeknd"},  # Blinding Lights by The Weeknd
    {"track_name": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars"}  # Uptown Funk by Mark Ronson
]

# Fetch genre information from Spotify API using artist's ID
def get_genre_from_spotify(artist_name):
    # Search for the artist on Spotify
    result = sp.search(q=artist_name, type='artist', limit=1)
    
    if result['artists']['items']:
        artist = result['artists']['items'][0]
        genres = artist.get('genres', [])
        
        if genres:
            return genres
        else:
            print(f"No genre found for {artist_name} on Spotify.")
            return None
    else:
        print(f"Artist {artist_name} not found on Spotify.")
        return None

# Main execution
if __name__ == "__main__":
    for track in test_tracks:
        print(f"Fetching genres for {track['track_name']} by {track['artist']}")
        genres = get_genre_from_spotify(track['artist'])
        
        if genres:
            print(f"Genres for {track['track_name']} by {track['artist']}: {', '.join(genres)}")
        else:
            print(f"No genres found for {track['track_name']} by {track['artist']}")