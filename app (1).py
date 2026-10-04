import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Doon Cafe Finder", page_icon="☕", layout="centered")

# ---------- COZY PASTEL THEME ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Quicksand:wght@400;600&display=swap');
:root{--cream:#FFF8E7;--yellow:#FFE9A0;--brown:#6B4A3A;--turq:#8ED9CF;--lav:#DCCDF2;}
.stApp{background:var(--cream);color:var(--brown);font-family:'Quicksand',sans-serif;}
h1,h2,h3{font-family:'Fraunces',serif!important;color:var(--brown)!important;}
label,p,span,div{color:var(--brown);}
.hero{background:linear-gradient(135deg,var(--lav),var(--turq));padding:26px;border-radius:24px;
      text-align:center;margin-bottom:18px;box-shadow:0 6px 0 var(--brown)22;}
.hero h1{margin:0;font-size:2.2rem;} .hero p{margin:6px 0 0;font-size:1.05rem;}
.card{background:#fff;border:2px solid var(--yellow);border-radius:20px;padding:16px 18px;
      margin:12px 0;box-shadow:0 4px 0 var(--yellow);}
.card h3{margin:0 0 4px;} .score{float:right;background:var(--turq);color:var(--brown);
      padding:4px 14px;border-radius:999px;font-weight:700;}
.tag{display:inline-block;background:var(--lav);border-radius:999px;padding:2px 10px;
     margin:6px 6px 0 0;font-size:.82rem;}
.tag.y{background:var(--yellow);}
div.stButton>button{background:var(--brown);color:var(--cream);border:none;border-radius:999px;
     padding:.6rem 1.6rem;font-weight:700;width:100%;}
div.stButton>button:hover{background:#8a6350;color:#fff;}
footer,#MainMenu{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- CAFE DATA ----------
AREAS = ["Rajpur Road", "Pacific Mall area", "Jakhan", "Ballupur", "Vasant Vihar",
         "Clock Tower", "Prem Nagar", "Clement Town", "Sahastradhara Road"]
PURPOSES = {"☕ Coffee": "coffee", "📚 Studying": "study", "❤️ Date": "date", "👯 Friends": "friends",
            "💻 Working": "work", "🎂 Birthday": "birthday", "🌿 Peaceful ambience": "peaceful"}

@st.cache_data
def load_cafes():
    try:
        return pd.read_csv("cafes.csv")  # replace with your real Dehradun data
    except FileNotFoundError:
        pass
    # SAMPLE (fictional) cafes -- swap with real data in cafes.csv
    cols = ["name","area","price","rating","wifi","noise","coffee","study","date","friends","work","birthday","peaceful"]
    rows = [
     ["The Doon Brew Co.","Rajpur Road",2,4.5,1,1,5,5,3,3,5,2,4],
     ["Lavender Lane Cafe","Rajpur Road",3,4.3,1,1,4,3,5,3,3,3,5],
     ["Chai & Chapters","Jakhan",1,4.2,1,1,4,5,2,3,4,1,5],
     ["Mallow Mug","Pacific Mall area",2,4.1,1,3,3,2,3,5,2,5,1],
     ["Turquoise Table","Vasant Vihar",3,4.4,1,2,4,3,5,4,3,4,4],
     ["Clock Tower Cuppa","Clock Tower",1,3.9,0,3,3,1,2,5,1,3,1],
     ["Bean There Doon","Ballupur",2,4.0,1,2,4,4,3,4,4,3,3],
     ["Honeycomb Hideout","Prem Nagar",1,4.3,1,2,3,4,3,4,4,3,3],
     ["Hilltop Roasters","Sahastradhara Road",2,4.6,0,1,5,2,5,3,1,3,5],
     ["Sunday Slow Cafe","Clement Town",2,4.2,1,1,4,4,4,3,4,2,5],
     ["Biscotti Bay","Pacific Mall area",3,4.0,1,3,3,2,4,5,2,5,2],
     ["Pine & Pour","Rajpur Road",2,4.7,1,2,5,4,4,4,4,3,4],
     ["Cozy Corner Doon","Jakhan",1,3.8,1,2,3,3,2,4,3,2,3],
     ["Marigold Cafe","Vasant Vihar",2,4.4,1,1,4,5,3,3,5,2,5],
     ["Rainy Day Roastery","Ballupur",2,4.1,0,2,5,2,3,3,2,2,4],
     ["Jam Jar Cafe","Prem Nagar",1,4.0,1,3,3,2,2,5,2,5,1],
    ]
    return pd.DataFrame(rows, columns=cols)

cafes = load_cafes()

# ---------- ML MODEL ----------
FEATS = ["area_match","price_gap","purpose_fit","wifi_ok","noise_gap","rating","rating_ok"]

def make_features(user, df):
    X = pd.DataFrame(index=df.index)
    X["area_match"] = (df["area"] == user["area"]).astype(int)
    X["price_gap"] = (df["price"] - user["budget"]).abs()
    X["purpose_fit"] = df[user["purpose"]]
    X["wifi_ok"] = ((user["wifi"] == 0) | (df["wifi"] == 1)).astype(int)
    X["noise_gap"] = (df["noise"] - user["noise"]).abs()
    X["rating"] = df["rating"]
    X["rating_ok"] = (df["rating"] >= user["min_rating"]).astype(int)
    return X[FEATS]

@st.cache_resource
def train_model():
    """Learns preference -> match score from simulated user/cafe pairs."""
    rng = np.random.default_rng(42)
    n = 6000
    X = pd.DataFrame({
        "area_match": rng.integers(0, 2, n), "price_gap": rng.integers(0, 3, n),
        "purpose_fit": rng.integers(1, 6, n), "wifi_ok": rng.integers(0, 2, n),
        "noise_gap": rng.integers(0, 3, n), "rating": rng.uniform(3.2, 5, n).round(1),
        "rating_ok": rng.integers(0, 2, n)})
    y = (20*X.purpose_fit/5 + 15*X.area_match + 15*X.wifi_ok + 12*(1-X.price_gap/2)
         + 12*(1-X.noise_gap/2) + 14*(X.rating-3.2)/1.8 + 12*X.rating_ok
         + rng.normal(0, 3, n)).clip(0, 100)
    return RandomForestRegressor(n_estimators=150, random_state=0).fit(X, y)

model = train_model()

# ---------- UI ----------
st.markdown('<div class="hero"><h1>☕ Doon Cafe Finder</h1>'
            '<p>Tell us your vibe — our ML model finds your perfect cafe in Dehradun 🏔️</p></div>',
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
area = c1.selectbox("📍 Area", AREAS)
purpose_label = c2.selectbox("✨ Best cafe for…", list(PURPOSES))
budget = c1.select_slider("💰 Budget", [1, 2, 3], value=2, format_func=lambda x: "₹" * x)
noise = c2.select_slider("🤫 Ambience", [1, 2, 3], value=1,
                         format_func=lambda x: {1: "Quiet", 2: "Moderate", 3: "Lively"}[x])
min_rating = c1.slider("⭐ Minimum rating", 3.0, 5.0, 4.0, 0.1)
wifi = c2.toggle("📶 Need Wi-Fi", value=True)

if st.button("Find my cafe 🔍"):
    user = dict(area=area, budget=budget, purpose=PURPOSES[purpose_label],
                wifi=int(wifi), noise=noise, min_rating=min_rating)
    scores = model.predict(make_features(user, cafes))
    res = cafes.assign(match=scores.round(0).astype(int)).sort_values("match", ascending=False).head(3)
    medals = ["🥇 Best Match", "🥈 Runner-up", "🥉 Also great"]
    st.subheader("Your top picks")
    for medal, (_, r) in zip(medals, res.iterrows()):
        tags = (f'<span class="tag y">📍 {r.area}</span><span class="tag y">{"₹"*int(r.price)}</span>'
                f'<span class="tag y">⭐ {r.rating}</span>'
                f'<span class="tag">{"📶 Wi-Fi" if r.wifi else "No Wi-Fi"}</span>'
                f'<span class="tag">{["","Quiet","Moderate","Lively"][int(r.noise)]}</span>')
        st.markdown(f'<div class="card"><span class="score">{r.match}%</span>'
                    f'<small>{medal}</small><h3>{r["name"]}</h3>{tags}</div>', unsafe_allow_html=True)
    st.progress(int(res.match.iloc[0]) / 100, text="Top match confidence")

st.caption("Built with Streamlit + scikit-learn (Random Forest) · Sample cafe data — replace cafes.csv with real listings.")
