import os
import requests
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Global Spotify Client Setup
load_dotenv()
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope="user-read-recently-played"
))

# Fetch artist genre from Spotify
def get_artist_genre_from_spotify(artist_id):
    try:
        artist = sp.artist(artist_id)
        genres = artist['genres']
        if genres:
            return genres[0]  
        else:
            return "No genre found"
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error fetching genre for artist ID {artist_id}: {e}")
        return "Error fetching genre"

# Fetch recent tracks from Spotify
def get_recent_tracks(limit=10):
    results = sp.current_user_recently_played(limit=limit)
    tracks = []
    for item in results['items']:
        track = item['track']
        track_info = {
            "name": track['name'],
            "artist": track['artists'][0]['name'],
            "artist_id": track['artists'][0]['id'],
            "id": track['id']
        }
        tracks.append(track_info)

    return tracks

# Main execution
if __name__ == "__main__":
    recent_tracks = get_recent_tracks()
    print(f"Retrieved {len(recent_tracks)} tracks:")

    for track in recent_tracks:
        print(f"Fetching genre for {track['name']} by {track['artist']}")
        genre = get_artist_genre_from_spotify(track['artist_id'])
        print(f"Genre for {track['name']} by {track['artist']}: {genre}")