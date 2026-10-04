# ☕ Doon Cafe Finder

ML-powered cafe recommender for **Dehradun**. Pick an area or college (GEU, UPES, DIT, Uttaranchal), budget, purpose, ambience and rating, and get your top 3 matches with a match percentage and reasons.

🔗 **Live app:** https://YOUR-APP-NAME.streamlit.app

## Features
- Login / sign up / guest mode
- 100+ Dehradun cafes across 20 localities, including Buddha Temple & New Basti
- "Best cafe for" coffee, studying, dates, friends, working, birthdays, peaceful vibes
- Gradient Boosting model trained on simulated user/cafe preference pairs
- Cozy pastel UI

## Data
`cafes.csv`: names/areas from public listings. `verified=1` rows have cost (and sometimes rating) from public listings; others show "(est.)". Wi-Fi, noise and purpose scores are estimates. Contributions welcome.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py
