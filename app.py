from flask import Flask, jsonify, request
from flask_cors import CORS
from spotipy import Spotify
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import io
import base64

app = Flask(__name__)
CORS(app)

@app.route("/run-model", methods=["POST"])
def run_model():
    data = request.get_json()
    token = data.get("access_token")

    if not token:
        return jsonify({"error": "Access token required"}), 400

    sp = Spotify(auth=token)
    user_profile = sp.current_user()
    print("Logged in as:", user_profile['display_name'])

    # Get liked song IDs
    liked_song_ids = []
    results = sp.current_user_saved_tracks()
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    for track in tracks:
        liked_song_ids.append(track['track']['id'])

    # Load dataset
    df = pd.read_csv("data/data.csv")
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                'duration_ms', 'explicit', 'key', 'mode']
    df['liked'] = df['id'].isin(liked_song_ids).astype(int)

    # KNN Scoring
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features])
    liked_X = X[df['liked'] == 1]
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(liked_X)
    distances, _ = knn.kneighbors(X)
    scores = 1 - (distances.mean(axis=1) / distances.max())
    df['user_score'] = scores

    user_liked_threshold = 0.85
    df['liked_predicted'] = (df['user_score'] >= user_liked_threshold).astype(int)

    # Create bar chart of liked songs' average features
    liked_songs_df = df[df['id'].isin(liked_song_ids)][features]
    df_normalized = pd.DataFrame(scaler.fit_transform(liked_songs_df), columns=features)
    mean_features = df_normalized.mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    mean_features.sort_values().plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title("Average Audio Feature Values of Liked Songs")
    ax.set_xlabel("Average (Normalized) Value")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    top_songs = df[df['liked_predicted'] == 1][['name', 'artists', 'user_score']].head(10).to_dict(orient='records')

    return jsonify({
        "message": "Model run complete!",
        "recommendations": top_songs,
        "figure_base64": f"data:image/png;base64,{img_base64}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)