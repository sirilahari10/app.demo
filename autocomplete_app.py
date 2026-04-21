import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="How AI Thinks", layout="centered")

st.title("The Ultimate Autocomplete 🧠")
st.markdown("Select a starting phrase and watch how the AI calculates the mathematical probability of the next word.")

# Simulated LLM probabilities for educational purposes
scenarios = {
    "The quick brown fox jumps over the lazy ": {"dog": 95.2, "cat": 3.1, "fence": 1.5, "turtle": 0.2},
    "I need to plug in my phone, the battery is ": {"dead": 55.4, "low": 40.1, "dying": 4.0, "hot": 0.5},
    "The data scientist wrote a Python ": {"script": 65.0, "code": 25.5, "program": 8.0, "snake": 1.5},
    "I'd like to order a large pepperoni ": {"pizza": 98.0, "slice": 1.5, "calzone": 0.4, "dog": 0.1}
}

selected_phrase = st.selectbox("Type or select a sentence:", list(scenarios.keys()))

if st.button("Predict Next Word"):
    with st.spinner("Calculating billions of parameters..."):
        time.sleep(1) # Adding a slight pause for dramatic effect
        
        data = scenarios[selected_phrase]
        # Sort data for the chart
        df = pd.DataFrame(list(data.items()), columns=["Word", "Probability"]).sort_values("Probability", ascending=False)
        
        # Display the winning word
        top_word = df.iloc[0]["Word"]
        st.subheader(f'The AI guesses: "{top_word}"')
        
        # Display the chart
        st.markdown("### The Probability Breakdown")
        st.bar_chart(df.set_index("Word"))
        
        st.info("💡 **Notice:** The AI doesn't 'understand' what a pizza or a data scientist is. It just knows that mathematically, the word 'pizza' follows 'pepperoni' 98% of the time in the data it read!")
