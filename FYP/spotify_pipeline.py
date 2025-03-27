from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_spotify(csv_path):
    df = pd.read_csv(csv_path)
    print("\n🎵 All columns in Spotify dataset:")
    print(df.columns.tolist())
    feature_columns = [
    'BPM',
    'Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]',
    'Frequency [Folk]', 'Frequency [Gospel]', 'Frequency [Hip hop]',
    'Frequency [Jazz]', 'Frequency [K pop]', 'Frequency [Latin]',
    'Frequency [Lofi]', 'Frequency [Metal]', 'Frequency [Pop]',
    'Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]',
    'Frequency [Video game music]'
    ]
    freq_map = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Usually": 3,
    "Always": 4
}
    available_columns = [col for col in feature_columns if col in df.columns]
    for col in available_columns:
        if df[col].dtype == object:
            df[col] = df[col].map(freq_map)
    
    df = df.dropna(subset = available_columns + ['Depression'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[available_columns])
    labels = df['Depression'].values
    return scaled_features, labels

if __name__ == "__main__":
    X, y = load_spotify("/datasets/mxmh_survey_results.csv")
    print("Scaled Spotify Features:", X[:5])
    print("Depression Labels:", y[:5])




