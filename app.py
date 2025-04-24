from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import os
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

CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI") or "https://your-frontend.web.app/callback"

@app.route("/authorize", methods=["POST"])
def authorize():
    data = request.get_json()
    code = data.get("code")
    code_verifier = data.get("code_verifier")

    if not code or not code_verifier:
        return jsonify({"error": "Missing code or verifier"}), 400

    token_url = "https://accounts.spotify.com/api/token"
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "code_verifier": code_verifier
    }

    response = requests.post(token_url, data=payload)
    token_info = response.json()
    access_token = token_info.get("access_token")

    if not access_token:
        return jsonify({"error": "Failed to exchange token", "details": token_info}), 400

    # Continue from here: same as /run-model logic
    headers = {"Authorization": f"Bearer {access_token}"}
    user_profile = requests.get("https://api.spotify.com/v1/me", headers=headers).json()
    print("Logged in as:", user_profile.get("display_name", "Unknown"))

    liked_song_ids = []
    tracks_url = "https://api.spotify.com/v1/me/tracks?limit=50"
    while tracks_url:
        res = requests.get(tracks_url, headers=headers).json()
        for item in res.get("items", []):
            liked_song_ids.append(item['track']['id'])
        tracks_url = res.get("next")

    df = pd.read_csv("data/data.csv")
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                'duration_ms', 'explicit', 'key', 'mode']
    df['liked'] = df['id'].isin(liked_song_ids).astype(int)

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