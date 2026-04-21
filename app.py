import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Netflix Recommendation Demo", layout="wide")
st.title("Netflix Recommendation Demo 🎬")
st.markdown("Select 3 shows from the search bar and rate them. Recommendations update live!")

# List of shows
all_shows = [
    "Stranger Things", "The Witcher", "The Crown", "Money Heist", "Dark",
    "Bridgerton", "Lupin", "Narcos", "The Queen's Gambit", "Squid Game"
]

# Multi-select search for 3 shows
selected_shows = st.multiselect("Pick up to 3 shows to rate", all_shows, default=[])

if len(selected_shows) > 3:
    st.warning("Please select only 3 shows.")
    selected_shows = selected_shows[:3]

ratings = {}
# Show sliders for each selected show
for show in selected_shows:
    rating = st.slider(f"Rate '{show}'", 1, 5, 3)
    ratings[show] = rating

if len(ratings) < 3:
    st.info("Select and rate 3 shows to get recommendations.")
else:
    st.subheader("Your Ratings")
    st.write(ratings)

    # Example dataset of other users
    other_users = pd.DataFrame({
        "Stranger Things": [5, 4, 2],
        "The Witcher": [3, 5, 1],
        "The Crown": [4, 2, 5],
        "Money Heist": [5, 3, 4],
        "Dark": [2, 5, 3],
        "Bridgerton": [4, 3, 5],
        "Lupin": [5, 4, 3],
        "Narcos": [3, 5, 2],
        "The Queen's Gambit": [4, 2, 5],
        "Squid Game": [5, 4, 2]
    })

    # Simple collaborative filtering recommendation
    recommendations = {}
    for show in other_users.columns:
        if show not in ratings:
            sim_scores = []
            for rated_show in ratings.keys():
                sim = 1 - abs(other_users[show] - ratings[rated_show]) / 4
                sim_scores.append(np.mean(sim))
            recommendations[show] = np.mean(sim_scores)

    st.subheader("Recommended Shows For You")
    for title, _ in sorted(recommendations.items(), key=lambda x: x[1], reverse=True):
        st.write(f"• {title}")
