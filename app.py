# app.py (Flask backend)

from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# === Load and prepare data once when app starts ===
ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv(
    'u.item',
    sep='|',
    encoding='latin-1',
    header=None,
    usecols=[0, 1] + list(range(5, 24)),
    names=['movie_id', 'title'] + [f'genre_{i}' for i in range(19)]
)

data = pd.merge(ratings, movies, on='movie_id')
data['user_avg'] = data.groupby('user_id')['rating'].transform('mean')
data['movie_avg'] = data.groupby('movie_id')['rating'].transform('mean')

# Training
genre_cols = [f'genre_{i}' for i in range(19)]
features = ['user_avg', 'movie_avg'] + genre_cols
X = data[features]
y = data['rating']

scaler = StandardScaler()
X.loc[:, ['user_avg', 'movie_avg']] = scaler.fit_transform(X[['user_avg', 'movie_avg']])

model = LinearRegression()
model.fit(X, y)

movie_avg_map = data[['movie_id', 'movie_avg']].drop_duplicates().set_index('movie_id')

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        user_data = data[data['user_id'] == user_id]
        rated_ids = user_data['movie_id'].unique()
        unrated = movies[~movies['movie_id'].isin(rated_ids)].copy()

        genre_pref = user_data[genre_cols].multiply(user_data['rating'], axis=0).sum()
        genre_freq = user_data[genre_cols].sum()
        user_genre_score = genre_pref / genre_freq.replace(0, 1)

        unrated['genre_score'] = unrated[genre_cols].dot(user_genre_score)

        user_avg = user_data['user_avg'].iloc[0]
        unrated['user_avg'] = user_avg
        unrated['movie_avg'] = unrated['movie_id'].map(movie_avg_map['movie_avg'])
        unrated['movie_avg'].fillna(user_avg, inplace=True)

        unrated.loc[:, ['user_avg', 'movie_avg']] = scaler.transform(unrated[['user_avg', 'movie_avg']])

        X_unrated = unrated[['user_avg', 'movie_avg'] + genre_cols]
        unrated['predicted_rating'] = model.predict(X_unrated)
        unrated['final_score'] = unrated['predicted_rating'] + 0.3 * unrated['genre_score']

        top5 = unrated.sort_values(by='final_score', ascending=False).head(5)
        recommendations = list(top5['title'])

    user_ids = sorted(data['user_id'].unique())
    return render_template('index.html', user_ids=user_ids, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
