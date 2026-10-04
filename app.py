import hashlib, hmac, json, os
from urllib.parse import quote_plus
import numpy as np, pandas as pd, streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Doon Cafe Finder", page_icon="☕", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,700..900,100&family=DM+Sans:wght@400;500;700&display=swap');
:root{--cream:#F6EBDD;--olive:#3B3A1A;--terra:#B8502A;--mustard:#E7B948;--lav:#CBBCDD;--purple:#8E7BB3;--peach:#EAB79B;}
.stApp,.stApp *{font-family:'DM Sans',sans-serif;}
.stApp{background:var(--cream);}
.block-container{max-width:760px;padding-top:1.2rem;}
h1,h2,h3,.serif{font-family:'Fraunces',serif!important;font-weight:900!important;color:var(--olive)!important;letter-spacing:-.01em;}
.stApp label,.stApp p,.stApp li,[data-testid="stWidgetLabel"] *,[data-testid="stSliderThumbValue"],
[data-testid="stTickBarMin"],[data-testid="stTickBarMax"],[role="radiogroup"] *{color:var(--olive)!important;opacity:1!important;}
[data-baseweb="select"]>div,[data-baseweb="input"]>div{background:#FBF4EA!important;border:2px solid var(--lav)!important;border-radius:10px!important;}
[data-baseweb="select"] *,[data-baseweb="input"] input{color:var(--olive)!important;}
.eyebrow{color:var(--terra);font-weight:700;letter-spacing:.14em;font-size:.78rem;text-transform:uppercase;}
.wave{height:9px;width:110px;margin:8px 0 12px;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='36' height='9'%3E%3Cpath d='M0 4.5Q9 0 18 4.5T36 4.5' fill='none' stroke='%238E7BB3' stroke-width='2.6'/%3E%3C/svg%3E") repeat-x;}
.brand{text-align:center;margin:6px 0 4px;} .brand .serif{font-size:2.3rem;line-height:.95;display:block;}
.brand small{letter-spacing:.3em;font-size:.68rem;color:var(--olive);font-weight:700;}
.hero{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:10px 0 0;}
.hero-l{flex:1 1 280px;padding:10px 6px;} .hero-l h1{font-size:2.9rem;line-height:.98;margin:6px 0;}
.hero-l p{max-width:320px;font-size:1rem;}
.hero-r{flex:1 1 240px;min-height:240px;background:var(--terra);border-radius:60% 40% 30% 70%/50% 30% 70% 50%;
        display:flex;align-items:center;justify-content:center;position:relative;font-size:6.5rem;}
.badge{position:absolute;right:6px;bottom:6px;width:92px;height:92px;border-radius:50%;background:var(--mustard);
       display:flex;align-items:center;justify-content:center;text-align:center;font-size:.64rem;font-weight:700;
       letter-spacing:.1em;color:var(--olive);transform:rotate(-12deg);border:2px dashed var(--olive);}
.band{background:var(--lav);display:flex;flex-wrap:wrap;justify-content:space-around;gap:10px;padding:16px 10px;margin:18px -4px;border-radius:6px;}
.band div{text-align:center;font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--olive);}
.band b{display:block;font-family:'Fraunces',serif;font-size:1.5rem;color:var(--terra);}
.pcard{background:#FBF4EA;border-radius:16px;overflow:hidden;box-shadow:0 2px 0 rgba(59,58,26,.12);margin-bottom:6px;}
.ptop{height:120px;position:relative;display:flex;align-items:center;justify-content:center;font-size:3.2rem;}
.pm{position:absolute;right:10px;top:10px;background:var(--cream);border-radius:999px;padding:2px 12px;font-weight:700;font-size:.9rem;color:var(--olive);}
.ptop small{position:absolute;left:10px;top:12px;color:#fff;font-weight:700;letter-spacing:.06em;font-size:.72rem;text-transform:uppercase;}
.pbody{padding:12px 14px 14px;} .pname{font-family:'Fraunces',serif;font-weight:800;font-size:1.05rem;color:var(--olive);line-height:1.15;}
.psub{font-size:.8rem;color:var(--olive);opacity:.75;margin:2px 0 4px;}
.tag{display:inline-block;background:var(--lav);border-radius:999px;padding:2px 10px;margin:5px 5px 0 0;font-size:.76rem;color:var(--olive);}
.tag.y{background:var(--mustard);}
.why{font-size:.8rem;margin-top:8px;color:var(--olive);} .pprice{font-weight:700;margin-top:8px;color:var(--terra);}
.foot{background:var(--terra);border-radius:16px;padding:22px;margin:22px 0 8px;} .foot .serif{color:#FFF3E0!important;font-size:1.7rem;}
.foot p{color:#FFE9CF!important;margin:4px 0 0;}
div.stButton>button{background:var(--terra);color:#FFF3E0!important;border:none;border-radius:6px;padding:.65rem 1.4rem;
     font-weight:700;letter-spacing:.1em;text-transform:uppercase;font-size:.8rem;width:100%;}
div.stButton>button *{color:#FFF3E0!important;} div.stButton>button:hover{background:#9c4222;}
[data-testid="stLinkButton"] a{background:var(--lav);border:none;color:var(--olive)!important;border-radius:6px;font-weight:700;}
footer,#MainMenu{visibility:hidden;}
</style>""", unsafe_allow_html=True)

HERO = ('<div class="brand"><span class="serif">Doon<br>Cafe Finder</span><small>— DEHRADUN CAFE GUIDE —</small></div>'
        '<div class="hero"><div class="hero-l"><span class="eyebrow">ML-matched cafes</span>'
        '<h1>Good coffee. Good vibes.</h1><div class="wave"></div>'
        '<p>Tell us your area, budget and mood. We find the cafe that fits your people.</p></div>'
        '<div class="hero-r">☕<div class="badge">MADE IN<br>DOON WITH<br>LOVE</div></div></div>')

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
    st.markdown(HERO, unsafe_allow_html=True)
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
    df = pd.read_csv("cafes.csv")
    for c in ["address", "price_range", "cuisine", "features", "closes", "college"]:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    if "reviews" not in df.columns: df["reviews"] = np.nan
    if "verified" not in df.columns: df["verified"] = 0
    df["rating_f"] = df["rating"].fillna(4.0); return df
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
st.markdown(HERO, unsafe_allow_html=True)
st.markdown(f'<div class="band"><div><b>{len(cafes)}</b>Cafes</div><div><b>{cafes.area.nunique()}</b>Localities</div>'
            '<div><b>4</b>Colleges</div><div><b>ML</b>Matched</div></div>', unsafe_allow_html=True)
st.markdown('<span class="eyebrow">Tell us your vibe</span><h2 style="margin:2px 0">Find your cup.</h2><div class="wave"></div>', unsafe_allow_html=True)

loc = st.selectbox("📍 Where? (area or college)", LOCS)
purpose_label = st.selectbox("✨ Best cafe for…", list(PURPOSES))
people = st.slider("👥 How many people?", 1, 10, 2)
budget = st.slider("💰 Budget per person (₹)", 200, 1000, 300, 50)
st.caption(f"Up to ₹{budget} per person · about ₹{budget * people} for your group of {people}")
noise = st.radio("🤫 Ambience", [1, 2, 3], format_func=lambda x: NOISE[x], horizontal=True)
min_rating = st.slider("⭐ Minimum rating", 3.5, 5.0, 4.0, 0.1)
wifi = st.checkbox("📶 I need Wi-Fi", value=True)

# ---------------- DETAILS (Google-style) ----------------
def gmaps(r, mode="search"):
    q = quote_plus(f"{r['name']} {r.area} Dehradun")
    if mode == "dir": return f"https://www.google.com/maps/dir/?api=1&destination={q}"
    return f"https://www.google.com/maps/search/?api=1&query={q}"

def details_body(r):
    st.markdown(f"## {r['name']}")
    if pd.notna(r.rating):
        n = f" ({int(r.reviews):,} reviews)" if pd.notna(r.reviews) else ""
        rating = f"**{r.rating} ⭐**{n}"
    else:
        rating = "⭐ rating not verified yet"
    price = r.price_range if r.price_range else f"~₹{int(r.cost)} for two (est.)"
    parts = [rating, f"**{price}**"] + ([r.cuisine] if r.cuisine else [])
    st.markdown(" · ".join(parts))
    if r.closes: st.markdown(f"🟢 **Open** · Closes {r.closes}")
    if r.features: st.markdown(" ".join(f'<span class="tag">{f.strip()}</span>' for f in r.features.split("·")), unsafe_allow_html=True)
    st.markdown(f"📍 {r.address if r.address else r.area + ', Dehradun, Uttarakhand'}")
    if isinstance(r.college, str) and r.college: st.markdown(f"🎓 Near {r.college}")
    top = sorted(PURPOSES.items(), key=lambda kv: -r[kv[1]])[:3]
    st.markdown("**Best for:** " + " · ".join(k for k, _ in top))
    st.markdown(f"**Ambience:** {NOISE[int(r.noise)]} · {'Wi-Fi' if r.wifi else 'No Wi-Fi'} *(estimated)*")
    b1, b2, b3 = st.columns(3)
    b1.link_button("🧭 Directions", gmaps(r, "dir"), use_container_width=True)
    b2.link_button("📍 Maps", gmaps(r), use_container_width=True)
    b3.link_button("📸 Photos", gmaps(r), use_container_width=True)
    st.caption("Photos, menu and live hours open on Google Maps. Values without “(estimated)” come from Google listings.")

if hasattr(st, "dialog"):
    @st.dialog("Cafe details")
    def show_details(r): details_body(r)
else:
    def show_details(r):
        with st.expander("Cafe details", expanded=True): details_body(r)

# ---------------- RESULTS ----------------
if st.button("Find my cafe →"):
    u = dict(loc=loc, budget=budget * 2, purpose=PURPOSES[purpose_label], wifi=wifi, noise=noise, min_rating=min_rating)
    X = features(u, cafes)
    st.session_state.note = ""
    if loc in COLLEGES and X.loc_match.sum() == 0:
        st.session_state.note = "No cafes listed near that college yet, so showing the best matches across Dehradun."
        X["loc_match"] = 1
    adj = (np.maximum(cafes.friends, cafes.birthday) - 3) * 2.5 if people >= 5 else (cafes.peaceful - 3) * 1.0 if people <= 2 else 0
    cafes_s = cafes.assign(match=np.clip(model.predict(X) + adj, 0, 100).round(0).astype(int))
    res = cafes_s.sort_values("match", ascending=False).head(3)
    out = []
    for i, r in res.iterrows():
        pp = r.cost / 2
        why = []
        if X.loc[i, "loc_match"] and loc != "Anywhere in Dehradun": why.append("in your area")
        why.append("within budget" if pp <= budget else f"₹{int(pp - budget)} over per person")
        why.append(f"{purpose_label.split(' ',1)[1].lower()} score {int(r[u['purpose']])}/5")
        if people >= 5: why.append(f"group fit {int(max(r.friends, r.birthday))}/5")
        if wifi: why.append("Wi-Fi ✓" if r.wifi else "no Wi-Fi")
        out.append((i, int(r.match), " · ".join(why), int(pp * people)))
    st.session_state.results = out
    st.session_state.people = people

if st.session_state.get("results"):
    if st.session_state.get("note"): st.info(st.session_state.note)
    st.markdown('<span class="eyebrow">Our picks for you</span><h2 style="margin:2px 0">Better cafe, better day.</h2><div class="wave"></div>', unsafe_allow_html=True)
    tops = ["#5B5A2B", "#E7B948", "#C96B4B"]; icons = ["☕", "🍰", "🌿"]
    medals = ["🥇 Best match", "🥈 Runner-up", "🥉 Also great"]
    cols = st.columns(3)
    for col, medal, top, ic, (i, match, why, grp) in zip(cols, medals, tops, icons, st.session_state.results):
        r = cafes.loc[i]
        rating = f"⭐ {r.rating}" if pd.notna(r.rating) else "⭐ n/a"
        price = r.price_range if r.price_range else f'{"₹" if r.verified else "~₹"}{int(r.cost)} for two{"" if r.verified else " (est.)"}'
        near = f'<span class="tag">🎓 near {r.college}</span>' if r.college else ""
        with col:
            st.markdown(f'<div class="pcard"><div class="ptop" style="background:{top}">{ic}<small>{medal}</small><span class="pm">{match}%</span></div>'
                        f'<div class="pbody"><div class="pname">{r["name"]}</div><div class="psub">📍 {r.area}</div>'
                        f'<span class="tag y">{price}</span><span class="tag y">{rating}</span><span class="tag">{NOISE[int(r.noise)]}</span>{near}'
                        f'<div class="why">{why}</div>'
                        f'<div class="pprice">~₹{grp} for {st.session_state.get("people", 2)} people</div></div></div>', unsafe_allow_html=True)
            if st.button("View details", key=f"d{i}"): show_details(r)

st.markdown('<div class="foot"><span class="serif">Everyday is a good cafe day.</span>'
            '<p>Made in Dehradun for students, friends and coffee people.</p></div>', unsafe_allow_html=True)
st.caption("Cafe names, areas, cost and ratings come from public listings (Google, EazyDiner, Curly Tales). "
           "Rows with (est.) have estimated prices, and Wi-Fi/ambience/purpose scores are estimates. Please verify before visiting. ML: Gradient Boosting + Streamlit.")
