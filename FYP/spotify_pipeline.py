from sklearn.preprocessing import StandardScaler
import joblib  
import pandas as pd

def load_spotify(csv_path):
    df = pd.read_csv(csv_path)
    print("\nAll columns in Spotify dataset:")
    print(df.columns.tolist())
    genre_columns = [col for col in df.columns if col.startswith("Frequency [")]
    freq_map = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Usually": 3,
    "Always": 4
}
    available_columns = [col for col in genre_columns if col in df.columns]
    for col in available_columns:
        if df[col].dtype == object:
            df[col] = df[col].map(freq_map)
    
    df = df.dropna(subset = available_columns + ['Depression'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[available_columns])
    joblib.dump(scaler, "models/spotify_scaler.pkl")
    print("Scaler saved to models/spotify_scaler.pkl")
    labels = df['Depression'].values
    print(f"Processed {scaled_features.shape[0]} samples with {scaled_features.shape[1]} genre features.")
    return scaled_features, labels

if __name__ == "__main__":
    X, y = load_spotify("/datasets/mxmh_survey_results.csv")
    print("Scaled Spotify Features:", X[:5])
    print("Depression Labels:", y[:5])




