from flask import Flask, jsonify
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

@app.route("/run-model", methods=["GET"])
def run_model():
    
    # %%
    import numpy as np
    import pandas as pd

    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    import plotly.express as px

    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import euclidean_distances
    from scipy.spatial.distance import cdist

    import warnings

    # %% [markdown]
    # **Importing the nessisary libraries for**
    # 
    #     Handalling math functions : numpy 
    # 
    #     Handeling CSV files : pandas
    # 
    #     Plotting data : seaborn, matplotlib, plotly
    # 
    #     Acessing machine learning modles : sklearn
    # 
    #     Technical computations : scipy

    # %%
    data = pd.read_csv('data/data.csv')
    data_by_artist = pd.read_csv('data/data_by_artist.csv')
    data_by_year = pd.read_csv('data/data_by_year.csv')
    data_by_genre  = pd.read_csv('data/data_by_genre.csv')
    data_w_genre  = pd.read_csv('data/data_w_genre.csv')

    # %% [markdown]
    # **Using pandas to read the CSV files into three variables**
    # 
    #     data.csv INTO data
    # 
    #     data_by_artist.csv INTO data_by_artist
    # 
    #     data_by_year.csv INTO data_by_year
    # 
    #     data_by_genre.csv INTO data_by_genre
    # 
    #     data_w_genre.csv INTO data_w_genre


    # %%
    # importing spotipy to get user and song data
    import spotipy

    # importing 0Auth to Authenticate user credietials
    from spotipy.oauth2 import SpotifyOAuth

    # importing collections to find songs in the spotify defaultdict
    from collections import defaultdict

    # %% [markdown]
    # **Imorting the nessisary Spotipy libraries and tools to:**
    # 
    #     Authenticate and Log In Users : SpotifyOAuth
    # 
    #     Compare and use the collected songs on spotify : collections

    # %%
    # Step 1: Define your Spotify Developer credentials
    CLIENT_ID = 'ddd1727c389c4438a214f8b617e63f3d'
    CLIENT_SECRET = '18cb85f3f93b467a97be3f935b22e492'
    REDIRECT_URI = 'http://127.0.0.1:8888/callback'

    # Step 2: Define the permissions you need (called scopes)
    # You can find full list of scopes here: https://developer.spotify.com/documentation/web-api/concepts/scopes
    SCOPE = 'user-library-read user-top-read playlist-modify-public'

    # Step 3: Create a SpotifyOAuth object to handle the OAuth flow
    sp_oauth = SpotifyOAuth(client_id=CLIENT_ID,
                            client_secret=CLIENT_SECRET,
                            redirect_uri=REDIRECT_URI,
                            scope=SCOPE)

    # Step 4: Create a Spotipy client with authenticated user token
    sp = spotipy.Spotify(auth_manager=sp_oauth)

    # Step 5: Now you can access user data! Let's get the current user's profile
    user_profile = sp.current_user()

    # Step 6: Print their display name
    print("Logged in as:", user_profile['display_name'])


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
    
    import io
    import base64

    # ✅ Step 1: Create a figure + axis manually (NO GUI backend triggered)
    fig, ax = plt.subplots(figsize=(10, 6))

    # ✅ Step 2: Do all plotting using the axis object
    mean_features.sort_values().plot(kind='barh', color='skyblue', ax=ax)

    # ✅ Step 3: Customize plot safely
    ax.set_title("Average Audio Feature Values of Liked Songs")
    ax.set_xlabel("Average (Normalized) Value")
    fig.tight_layout()

    # ✅ Step 4: Convert to base64 for Flask response
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)  # Clean up

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

    data.to_csv("data/data_with_user_scores.csv", index=False)

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
    df = pd.read_csv("data/data_with_user_scores.csv")

    df['liked'] = (df['user_score'] >= user_liked_threshold).astype(int)

    df.to_csv("data/data_scored_with_liked.csv", index=False)

    # %% [markdown]
    # **Adding a column that called liked that represents wether or not a user does or does not like a song with a 1 or a 0**


    result = {
            
        "message": "Model run complete!",
        
        "recommendations": sp.current_user_saved_tracks()['items']
            
    }
        
    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000)
