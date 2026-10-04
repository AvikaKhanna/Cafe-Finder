import hashlib, hmac, json, os
import numpy as np, pandas as pd, streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Doon Cafe Finder", page_icon="☕", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Quicksand:wght@400;600;700&display=swap');
:root{--cream:#FFF8E7;--yellow:#FFE9A0;--brown:#5A3E30;--turq:#8ED9CF;--lav:#DCCDF2;}
.stApp,.stApp *{font-family:'Quicksand',sans-serif;}
.stApp{background:var(--cream);}
h1,h2,h3{font-family:'Fraunces',serif!important;color:var(--brown)!important;}
.stApp label,.stApp p,.stApp li,[data-testid="stWidgetLabel"] *,[data-testid="stSliderThumbValue"],
[data-testid="stTickBarMin"],[data-testid="stTickBarMax"],[role="radiogroup"] *{color:var(--brown)!important;opacity:1!important;}
[data-baseweb="select"]>div,[data-baseweb="input"]>div{background:#fff!important;border:2px solid var(--lav)!important;border-radius:14px!important;}
[data-baseweb="select"] *,[data-baseweb="input"] input{color:var(--brown)!important;}
.hero{background:linear-gradient(135deg,var(--lav),var(--turq));padding:26px;border-radius:24px;text-align:center;margin-bottom:18px;}
.hero h1{margin:0;font-size:2.1rem;} .hero p{margin:6px 0 0;color:var(--brown)!important;}
.card{background:#fff;border:2px solid var(--yellow);border-radius:20px;padding:16px 18px;margin:12px 0;box-shadow:0 4px 0 var(--yellow);color:var(--brown);}
.card h3{margin:2px 0 4px;} .score{float:right;background:var(--turq);padding:4px 14px;border-radius:999px;font-weight:700;}
.tag{display:inline-block;background:var(--lav);border-radius:999px;padding:2px 10px;margin:6px 6px 0 0;font-size:.82rem;}
.tag.y{background:var(--yellow);} .why{font-size:.88rem;margin-top:8px;}
div.stButton>button{background:#5A3E30;color:#FFF8E7!important;border:none;border-radius:999px;padding:.6rem 1.6rem;font-weight:700;width:100%;}
div.stButton>button *{color:#FFF8E7!important;}
footer,#MainMenu{visibility:hidden;}
</style>""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
USERS = "users.json"
def _load(): return json.load(open(USERS)) if os.path.exists(USERS) else {}
def _hash(pw, salt): return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 120_000).hex()

def signup(u, pw):
    users = _load()
    if u in users: return False
    salt = os.urandom(16)
    users[u] = {"salt": salt.hex(), "hash": _hash(pw, salt)}
    json.dump(users, open(USERS, "w")); return True

def login(u, pw):
    rec = _load().get(u)
    return bool(rec) and hmac.compare_digest(rec["hash"], _hash(pw, bytes.fromhex(rec["salt"])))

if "user" not in st.session_state:
    st.markdown('<div class="hero"><h1>☕ Doon Cafe Finder</h1><p>Sign in to find your perfect cafe in Dehradun 🏔️</p></div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔑 Log in", "✨ Sign up"])
    with tab1:
        u = st.text_input("Username", key="lu"); p = st.text_input("Password", type="password", key="lp")
        if st.button("Log in"):
            if login(u.strip().lower(), p): st.session_state.user = u.strip(); st.rerun()
            else: st.error("Wrong username or password.")
    with tab2:
        u2 = st.text_input("Choose a username", key="su"); p2 = st.text_input("Choose a password (min 6 chars)", type="password", key="sp")
        if st.button("Create account"):
            if len(u2.strip()) < 3 or len(p2) < 6: st.warning("Username 3+ chars, password 6+ chars.")
            elif signup(u2.strip().lower(), p2): st.success("Account created! Now log in 🎉")
            else: st.error("That username is taken.")
    if st.button("Continue as guest 👀"): st.session_state.user = "Guest"; st.rerun()
    st.stop()

# ---------------- DATA ----------------
@st.cache_data
def load():
    df = pd.read_csv("cafes.csv"); df["rating_f"] = df["rating"].fillna(4.0); return df
cafes = load()

COLLEGES = {"🎓 Near GEU (Graphic Era)": "GEU", "🎓 Near UPES": "UPES",
            "🎓 Near DIT University": "DIT", "🎓 Near Uttaranchal University": "UU"}
LOCS = ["Anywhere in Dehradun"] + sorted(cafes.area.unique()) + list(COLLEGES)
PURPOSES = {"☕ Coffee": "coffee", "📚 Studying": "study", "❤️ Date": "date", "👯 Friends": "friends",
            "💻 Working": "work", "🎂 Birthday": "birthday", "🌿 Peaceful ambience": "peaceful"}
NOISE = {1: "Quiet", 2: "Moderate", 3: "Lively"}
FEATS = ["loc_match", "over_budget", "under_budget", "purpose_fit", "wifi_ok", "noise_gap", "rating", "rating_ok"]

def features(u, df):
    X = pd.DataFrame(index=df.index)
    if u["loc"] == "Anywhere in Dehradun": X["loc_match"] = 1
    elif u["loc"] in COLLEGES: X["loc_match"] = (df.college == COLLEGES[u["loc"]]).astype(int)
    else: X["loc_match"] = (df.area == u["loc"]).astype(int)
    X["over_budget"] = ((df.cost - u["budget"]).clip(lower=0) / u["budget"]).clip(upper=2)
    X["under_budget"] = ((u["budget"] - df.cost).clip(lower=0) / u["budget"]).clip(upper=1)
    X["purpose_fit"] = df[u["purpose"]]
    X["wifi_ok"] = ((not u["wifi"]) | (df.wifi == 1)).astype(int)
    X["noise_gap"] = (df.noise - u["noise"]).abs()
    X["rating"] = df.rating_f
    X["rating_ok"] = (df.rating_f >= u["min_rating"]).astype(int)
    return X[FEATS]

@st.cache_resource
def train():
    """Learn preference->match score from simulated user/cafe pairs."""
    r = np.random.default_rng(7); n = 8000
    X = pd.DataFrame({"loc_match": r.integers(0, 2, n), "over_budget": r.choice([0, 0, .1, .3, .6, 1, 2], n),
        "under_budget": r.choice([0, 0, .2, .5, .8], n), "purpose_fit": r.integers(1, 6, n),
        "wifi_ok": r.choice([0, 1, 1], n), "noise_gap": r.integers(0, 3, n),
        "rating": r.uniform(3.5, 5, n), "rating_ok": r.integers(0, 2, n)})
    y = (30 * (X.purpose_fit - 1) / 4 + 22 * X.loc_match + 14 * X.wifi_ok - 28 * X.over_budget.clip(upper=1.2)
         + 6 * X.under_budget + 14 * (1 - X.noise_gap / 2) + 12 * (X.rating - 3.5) / 1.5 + 10 * X.rating_ok
         + r.normal(0, 2, n) + 18)
    return GradientBoostingRegressor(n_estimators=250, max_depth=4, random_state=0).fit(X, y.clip(0, 100))
model = train()

# ---------------- APP ----------------
c0, c1 = st.columns([4, 1])
c0.markdown(f"**Hi, {st.session_state.user}! 👋**")
if c1.button("Logout"): del st.session_state.user; st.rerun()
st.markdown('<div class="hero"><h1>☕ Doon Cafe Finder</h1><p>Tell us your vibe, our ML model finds your cafe 🏔️</p></div>', unsafe_allow_html=True)

loc = st.selectbox("📍 Where? (area or college)", LOCS)
purpose_label = st.selectbox("✨ Best cafe for…", list(PURPOSES))
budget = st.slider("💰 Max budget for two (₹)", 200, 2000, 600, 100)
st.caption(f"Your budget: up to ₹{budget} for two")
noise = st.radio("🤫 Ambience", [1, 2, 3], format_func=lambda x: NOISE[x], horizontal=True)
min_rating = st.slider("⭐ Minimum rating", 3.5, 5.0, 4.0, 0.1)
wifi = st.checkbox("📶 I need Wi-Fi", value=True)

if st.button("Find my cafe 🔍"):
    u = dict(loc=loc, budget=budget, purpose=PURPOSES[purpose_label], wifi=wifi, noise=noise, min_rating=min_rating)
    X = features(u, cafes)
    if loc in COLLEGES and X.loc_match.sum() == 0:
        st.info("No cafes listed near that college yet, so showing the best matches across Dehradun.")
        X["loc_match"] = 1
    res = cafes.assign(match=np.clip(model.predict(X), 0, 100).round(0).astype(int)).sort_values("match", ascending=False).head(3)
    st.subheader("Your top picks")
    for medal, (i, r) in zip(["🥇 Best match", "🥈 Runner-up", "🥉 Also great"], res.iterrows()):
        why = []
        if X.loc[i, "loc_match"] and loc != "Anywhere in Dehradun": why.append("in your area")
        why.append("within budget" if r.cost <= budget else f"₹{int(r.cost - budget)} over budget")
        why.append(f"{purpose_label.split(' ',1)[1].lower()} score {int(r[u['purpose']])}/5")
        if wifi: why.append("Wi-Fi ✓" if r.wifi else "no Wi-Fi")
        rating = f"⭐ {r.rating}" if pd.notna(r.rating) else "⭐ n/a"
        near = f'<span class="tag">🎓 near {r.college}</span>' if isinstance(r.college, str) and r.college else ""
        st.markdown(f'<div class="card"><span class="score">{r.match}%</span><small>{medal}</small><h3>{r["name"]}</h3>'
                    f'<span class="tag y">📍 {r.area}</span><span class="tag y">{"₹" if r.verified else "~₹"}{int(r.cost)} for two{"" if r.verified else " (est.)"}</span>'
                    f'<span class="tag y">{rating}</span><span class="tag">{NOISE[int(r.noise)]}</span>{near}'
                    f'<div class="why">Why: {" · ".join(why)}</div></div>', unsafe_allow_html=True)

st.caption("Cafe names, areas, cost and ratings come from public listings (EazyDiner, Curly Tales, Google). "
           "Rows with (est.) have estimated prices, and all Wi-Fi/ambience/purpose scores are estimates. Please verify before visiting. ML: Gradient Boosting + Streamlit.")
