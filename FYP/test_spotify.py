import os
import requests
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Load environment variables
load_dotenv()

# Spotify API setup
sp = spotipy.Spotify(auth_manager=spotipy.SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope="user-read-playback-state user-read-recently-played"
))

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

def get_recent_tracks(limit=10):
    recent = sp.current_user_recently_played(limit=limit)
    tracks = []
    for item in recent["items"]:
        track = item["track"]
        track_name = track["name"]
        artist_name = track["artists"][0]["name"]
        tracks.append((track_name, artist_name))
    return tracks

# Main execution
if __name__ == "__main__":
    recent_tracks = get_recent_tracks()

    if recent_tracks:
        print("Last 10 tracks played:")
        for track_name, artist_name in recent_tracks:
            print(f"- {track_name} by {artist_name}")
            genres = get_genre_from_spotify(artist_name)
            if genres:
                print(f"  Genres: {', '.join(genres)}")
            else:
                print(f"  No genres found for {track_name} by {artist_name}")
    else:
        print("No recent tracks found.")