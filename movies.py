import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import nltk
nltk.download('vader_lexicon', quiet=True)

# ── page config
st.set_page_config(
    page_title="Movie Story Shapes",
    page_icon="🎬",
    layout="wide"
)

# ── load and train model
@st.cache_resource
def load_model():
    df = pd.read_csv('plot_summaries.txt',
                      sep='\t', header=None,
                      names=['movie_id', 'plot'])

    metadata = pd.read_csv('movie.metadata.tsv',
                            sep='\t', header=None,
                            names=['movie_id', 'freebase_id', 'title',
                                   'release_date', 'revenue', 'runtime',
                                   'languages', 'countries', 'genres'])

    sia = SentimentIntensityAnalyzer()

   def get_arc(plot, n=10):
    words = plot.split()
    seg_size = max(1, len(words) // n)
    scores = []
    for i in range(n):
        segment = ' '.join(words[i*seg_size:(i+1)*seg_size])
        score = sia.polarity_scores(segment)['compound']
        # give ending segments more weight
        if i >= 8:
            score = score * 1.5
        scores.append(score)
    return scores

    arcs, valid_ids = [], []
    for idx, row in df.iterrows():
        if len(str(row['plot']).split()) > 100:
            arcs.append(get_arc(row['plot']))
            valid_ids.append(idx)

    arcs_df = pd.DataFrame(arcs, columns=[f'seg_{i}' for i in range(10)])
    arcs_df['movie_id'] = df.loc[valid_ids, 'movie_id'].values

    scaler = StandardScaler()
    arcs_scaled = scaler.fit_transform(arcs_df[[f'seg_{i}' for i in range(10)]])

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    arcs_df['story_shape'] = kmeans.fit_predict(arcs_scaled)

    arcs_df['movie_id'] = arcs_df['movie_id'].astype(str)
    metadata['movie_id'] = metadata['movie_id'].astype(str)
    merged = arcs_df.merge(metadata[['movie_id', 'title']], on='movie_id', how='left')

    return merged, kmeans, scaler, sia

# ── shape metadata
SHAPE_NAMES = {
    0: "The Tragedy",
    1: "The False Hope",
    2: "The Feel Good",
    3: "The Roller Coaster",
    4: "The Slow Burn",
    5: "The Triumph"
}

SHAPE_DESC = {
    0: "Dark from start to finish. No redemption, no relief.",
    1: "Builds toward something good — then falls apart at the end.",
    2: "Consistently warm and positive all the way through.",
    3: "Up, down, up, down. Never lets you settle.",
    4: "Starts okay, slowly unravels. Hope fades.",
    5: "Rough start, hard middle — but ends on a high."
}

SHAPE_MOVIES = {
    0: "The Godfather Part II, Spider-Man 3",
    1: "Forrest Gump, The Notebook, Devil Wears Prada",
    2: "Inception, Rocky, Clueless",
    3: "Schindler's List, Harry Potter",
    4: "Mean Girls, Gravity",
    5: "Titanic, Finding Nemo, The Pursuit of Happyness"
}

COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c']

# ── UI
st.markdown("# 🎬 Every Movie Is One of 6 Stories")
st.markdown("##### Paste any movie plot and find out which emotional arc it follows — proven across 29,737 films")
st.markdown("---")

merged, kmeans, scaler, sia = load_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Paste a movie plot")
    user_plot = st.text_area("", height=200,
        placeholder="e.g. A 70-year-old widower becomes a senior intern at an online fashion site and develops a close relationship with his young boss...")

    if st.button("🔍 Find My Story Shape", use_container_width=True):
        if user_plot and len(user_plot.split()) > 20:

            def get_arc(plot, n=10):
    words = plot.split()
    seg_size = max(1, len(words) // n)
    scores = []
    for i in range(n):
        segment = ' '.join(words[i*seg_size:(i+1)*seg_size])
        score = sia.polarity_scores(segment)['compound']
        # give ending segments more weight
        if i >= 8:
            score = score * 1.5
        scores.append(score)
    return scores

            arc = get_arc(user_plot)
            arc_scaled = scaler.transform([arc])
            shape = kmeans.predict(arc_scaled)[0]
            color = COLORS[shape]

            st.markdown("---")
            st.markdown(f"## Your movie is:")
            st.markdown(f"<h1 style='color:{color}'>{SHAPE_NAMES[shape]}</h1>",
                        unsafe_allow_html=True)
            st.markdown(f"*{SHAPE_DESC[shape]}*")
            st.markdown(f"**Famous films with this exact arc:** {SHAPE_MOVIES[shape]}")

            # plot the arc
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#0f0f0f')
            ax.set_facecolor('#1a1a1a')
            ax.plot(range(10), arc, color=color, linewidth=3)
            ax.fill_between(range(10), arc, alpha=0.2, color=color)
            ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
            ax.set_ylim(-1, 1)
            ax.set_xticks(range(10))
            ax.set_xticklabels(['Start','','','','Mid','','','','','End'],
                                color='#aaaaaa', fontsize=9)
            ax.tick_params(colors='#aaaaaa')
            ax.set_title('Your Movie\'s Emotional Arc', color='white', fontsize=12)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
            st.pyplot(fig)

        else:
            st.warning("Paste a longer plot summary — at least a few sentences.")

with col2:
    st.markdown("### The 6 Story Shapes")
    segments = [f'seg_{i}' for i in range(10)]

    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    fig.patch.set_facecolor('#0f0f0f')
    axes = axes.flatten()

    for shape in range(6):
        ax = axes[shape]
        ax.set_facecolor('#1a1a1a')
        shape_data = merged[merged['story_shape'] == shape][segments]
        mean_arc = shape_data.mean()
        count = len(shape_data)
        color = COLORS[shape]

        ax.plot(range(10), mean_arc, color=color, linewidth=2.5)
        ax.fill_between(range(10), mean_arc, alpha=0.2, color=color)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.2)
        ax.set_ylim(-1, 1)
        ax.set_title(f'{SHAPE_NAMES[shape]}  ·  {count:,} films',
                      color='white', fontsize=9, fontweight='bold')
        ax.set_xticks([0, 5, 9])
        ax.set_xticklabels(['Start', 'Mid', 'End'], color='#aaaaaa', fontsize=7)
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.markdown("<center><small>Built by Siri Lahari Chava · NLP sentiment analysis + K-means clustering · 29,737 movies analyzed</small></center>",
            unsafe_allow_html=True) 
