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

# %%
# importing spotipy to get user and song data
import spotipy

app = Flask(__name__)
CORS(app)

CLIENT_ID = os.getenv("ddd1727c389c4438a214f8b617e63f3d")
CLIENT_SECRET = os.getenv("18cb85f3f93b467a97be3f935b22e492")
REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI") or "https://music-classifier-7cee1.web.app/callback"

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
        "redirect_uri": 'https://music-classifier-7cee1.web.app/callback.html',
        "client_id": 'ddd1727c389c4438a214f8b617e63f3d',
        "code_verifier": code_verifier
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # 🚨 Make sure you send headers manually without Authorization
    response = requests.post(token_url, data=payload, headers=headers)
    token_info = response.json()
    
    # Assuming you already have token_info from Spotify
    access_token = token_info['access_token']

    # Initialize Spotipy using ONLY the access token
    sp = spotipy.Spotify(auth=access_token)

    # Step 5: Now you can access user data! Let's get the current user's profile
    user_profile = sp.current_user()
    # %% [markdown]
    # **Authenticating the user:**
    # 
    # 
    # %%
    liked_song_ids = []

    results = sp.current_user_saved_tracks()
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    for track in tracks:
        liked_song_ids.append(track['track']['id'])

    # %% [markdown]
    # **Getting and storing all of the users liked songs ids in a list:**
    # 
    #     liked_song_ids


    # %%
    data = pd.read_csv('data/data.csv')
    data_w_genre  = pd.read_csv('data/data_w_genre.csv')
    
    # %%
    liked_songs_df = data[data['id'].isin(liked_song_ids)]

    # %% [markdown]
    # **Flitering the data df to only include the liked songd and storing in:**
    # 
    #     liked_song_df

    # %%
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','explicit','key','mode']

    liked_songs_df = liked_songs_df[features]

    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(liked_songs_df),
        columns=features
    )

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

    # %% [markdown]
    # **Visualizing the Normalized Audio Features**

    # %%
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data[features])

    data['liked'] = data['id'].isin(liked_song_ids).astype(int)

    # Only use liked songs for the "center points"
    liked_X = X[data['liked'] == 1]

    # Fit Nearest Neighbors
    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')  # or 'cosine'
    knn.fit(liked_X)

    distances, indices = knn.kneighbors(X)  # Find nearest liked songs for all songs

    # Avoid divide-by-zero issues
    max_distance = distances.max()
    scores = 1 - (distances.mean(axis=1) / max_distance)

    data['user_score'] = scores

    # %% [markdown]
    # **Adding a Column to the dateset called user_score**
    # 
    #     Represents the amount the user likes the song from 1 to 0
    # 
    #     Acheaved with k-nearest neighbor and finds the closer songs as more liked

    # %%
    user_liked_threshold = 0.85  # Value To demermin liked vs disliked songs

    # %% [markdown]
    # **This value represents the value the user_score must be over to be considered "liked"**

    # %%

    data['liked'] = (data['user_score'] >= user_liked_threshold).astype(int)

    # %% [markdown]
    # **Adding a column that called liked that represents wether or not a user does or does not like a song with a 1 or a 0**

    return jsonify({
        "message": "Model run complete!",
        "recommendations": liked_song_ids,
        "figure_base64": f"data:image/png;base64,{img_base64}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)