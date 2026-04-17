"""
CineAI — Application de recommandation de films
Inscription / Connexion / SVD MovieLens / OMDB pour affiches
"""

import streamlit as st
import requests
import json
import hashlib
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
OMDB_API_KEY = "1163cc6b"
OMDB_BASE    = "http://www.omdbapi.com/"
USERS_FILE   = "../data/users.json"
RATINGS_FILE = "../data/ratings_app.json"
MODEL_PATH   = "../models/svd_model.pkl"

st.set_page_config(page_title="CineAI", page_icon="🎬", layout="wide",
                   initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root { --red:#E50914; --red2:#B20710; --dark:#0a0a0a; --dark2:#141414;
        --dark3:#1e1e1e; --dark4:#252525; --muted:#6b7280; --border:rgba(255,255,255,0.07); }
* { box-sizing: border-box; }
.stApp { background:var(--dark) !important; font-family:'Inter',sans-serif !important; }
#MainMenu,footer,header { visibility:hidden !important; }
[data-testid="collapsedControl"],.stDeployButton { display:none !important; }
section[data-testid="stSidebar"] { display:none !important; }

.stTextInput > div > div > input {
    background:var(--dark4) !important; border:1px solid var(--border) !important;
    border-radius:8px !important; color:white !important;
    font-family:'Inter',sans-serif !important; font-size:14px !important; padding:10px 14px !important; }
.stTextInput > div > div > input:focus {
    border-color:rgba(229,9,20,0.5) !important; box-shadow:0 0 0 2px rgba(229,9,20,0.1) !important; }
.stTextInput label,.stSelectbox label,.stSlider label,.stRadio > label,.stNumberInput label {
    color:#4b5563 !important; font-size:10px !important; letter-spacing:2px !important;
    text-transform:uppercase !important; font-family:'JetBrains Mono',monospace !important; }
.stSelectbox > div > div { background:var(--dark4) !important; border:1px solid var(--border) !important;
    border-radius:8px !important; color:white !important; }
.stButton > button { background:var(--red) !important; color:white !important; border:none !important;
    border-radius:6px !important; font-family:'Inter',sans-serif !important; font-size:12px !important;
    font-weight:700 !important; letter-spacing:1.5px !important; text-transform:uppercase !important;
    padding:11px 20px !important; width:100% !important; transition:all 0.18s !important; }
.stButton > button:hover { background:var(--red2) !important; transform:translateY(-1px) !important; }
.stTabs [data-baseweb="tab-list"] { background:transparent !important;
    border-bottom:1px solid var(--border) !important; gap:0 !important;
    padding:0 !important; margin-bottom:24px !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:var(--muted) !important;
    font-family:'Inter',sans-serif !important; font-size:12px !important; font-weight:600 !important;
    letter-spacing:1px !important; text-transform:uppercase !important; padding:14px 22px !important;
    border:none !important; border-bottom:2px solid transparent !important;
    border-radius:0 !important; margin-bottom:-1px !important; }
.stTabs [aria-selected="true"] { color:white !important; border-bottom-color:var(--red) !important; }
.stRadio > div { flex-direction:row !important; gap:6px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

model_data   = load_model()
movies_df    = model_data['movies']       # movieId, title, genres
ratings_df   = model_data['ratings']     # userId, movieId, rating
pred_df      = model_data['pred_df']     # matrice prédictions (userId × movieId)

# ─────────────────────────────────────────────────────────────
# UTILITAIRES — Stockage
# ─────────────────────────────────────────────────────────────
def _load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def hash_pwd(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def register_user(email, username, password):
    users = _load_json(USERS_FILE)
    if email in users:
        return False, "Cet email est déjà utilisé."
    users[email] = {"username": username, "password": hash_pwd(password),
                    "created": datetime.now().isoformat()}
    _save_json(USERS_FILE, users)
    return True, "Compte créé avec succès !"

def login_user(email, password):
    users = _load_json(USERS_FILE)
    if email not in users:
        return False, "", "Email introuvable."
    if users[email]["password"] != hash_pwd(password):
        return False, "", "Mot de passe incorrect."
    return True, users[email]["username"], ""

def get_app_ratings(email):
    return _load_json(RATINGS_FILE).get(email, {})

def save_app_rating(email, movie_id, title, genres, rating):
    data = _load_json(RATINGS_FILE)
    if email not in data:
        data[email] = {}
    data[email][str(movie_id)] = {
        "title": title, "genres": genres,
        "rating": rating, "date": datetime.now().isoformat()
    }
    _save_json(RATINGS_FILE, data)

# ─────────────────────────────────────────────────────────────
# OMDB — Affiches et infos
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_poster(title):
    clean = title.split('(')[0].strip()
    try:
        r = requests.get(OMDB_BASE,
            params={"t": clean, "apikey": OMDB_API_KEY, "type": "movie"}, timeout=6)
        d = r.json()
        if d.get("Poster") and d["Poster"] != "N/A":
            return d["Poster"]
    except:
        pass
    return None

@st.cache_data(ttl=86400)
def search_omdb(query):
    try:
        r = requests.get(OMDB_BASE,
            params={"s": query, "apikey": OMDB_API_KEY, "type": "movie"}, timeout=6)
        d = r.json()
        if d.get("Response") == "True":
            return d.get("Search", [])
    except:
        pass
    return []

# ─────────────────────────────────────────────────────────────
# LOGIQUE SVD — Recommandation
# ─────────────────────────────────────────────────────────────
def get_svd_recommendations(app_ratings: dict, n: int = 12):
    """
    Trouve l'utilisateur MovieLens le plus similaire aux notes de l'utilisateur connecté,
    puis retourne les films recommandés par le SVD pour cet utilisateur.
    """
    if not app_ratings:
        return []

    # Construire un vecteur de notes pour l'utilisateur connecté
    # sur les films qu'il a notés (en utilisant le movieId MovieLens)
    rated_movie_ids = []
    rated_scores    = []
    for mid_str, info in app_ratings.items():
        try:
            mid = int(mid_str)
            if mid in pred_df.columns:
                rated_movie_ids.append(mid)
                rated_scores.append(info["rating"])
        except:
            pass

    if not rated_movie_ids:
        # Fallback : recommandations globales (films les mieux notés)
        popular = (
            ratings_df.groupby('movieId')['rating']
            .agg(count='count', mean='mean')
            .query('count >= 50')
            .sort_values('mean', ascending=False)
            .head(n)
        )
        result = movies_df[movies_df['movieId'].isin(popular.index)].copy()
        result['score'] = result['movieId'].map(popular['mean'])
        return result.sort_values('score', ascending=False).to_dict('records')

    # Trouver l'utilisateur MovieLens le plus similaire
    # On compare les notes sur les films en commun
    rated_set = set(rated_movie_ids)
    best_user = None
    best_sim  = -1

    for uid in pred_df.index:
        user_ratings_ml = ratings_df[
            (ratings_df['userId'] == uid) &
            (ratings_df['movieId'].isin(rated_set))
        ]
        if len(user_ratings_ml) < 2:
            continue
        # Corrélation de Pearson simplifiée
        common_movies = list(user_ratings_ml['movieId'])
        ml_scores  = user_ratings_ml.set_index('movieId')['rating'].reindex(common_movies).values
        app_scores = np.array([
            next(info["rating"] for m, info in app_ratings.items()
                 if int(m) == cm and cm in rated_set)
            for cm in common_movies
        ], dtype=float)
        if ml_scores.std() == 0 or app_scores.std() == 0:
            continue
        corr = np.corrcoef(ml_scores, app_scores)[0, 1]
        if corr > best_sim:
            best_sim  = corr
            best_user = uid

    # Films déjà notés par l'utilisateur connecté (à exclure)
    seen_ids = set(int(m) for m in app_ratings.keys())

    if best_user is not None and best_sim > 0.1:
        # Recommandations SVD pour l'utilisateur similaire
        user_preds = pred_df.loc[best_user]
        user_seen  = set(ratings_df[ratings_df['userId'] == best_user]['movieId'])
        candidates = user_preds.drop(index=list(user_seen | seen_ids), errors='ignore')
        top_ids    = candidates.nlargest(n).index.tolist()
    else:
        # Fallback : meilleurs films non notés globalement
        popular = (
            ratings_df[~ratings_df['movieId'].isin(seen_ids)]
            .groupby('movieId')['rating']
            .agg(count='count', mean='mean')
            .query('count >= 30')
            .sort_values('mean', ascending=False)
            .head(n)
        )
        top_ids = popular.index.tolist()

    result = movies_df[movies_df['movieId'].isin(top_ids)].copy()
    result['score'] = result['movieId'].map(
        pred_df.loc[best_user] if best_user else
        ratings_df.groupby('movieId')['rating'].mean()
    )
    return result.sort_values('score', ascending=False).head(n).to_dict('records')

# ─────────────────────────────────────────────────────────────
# COMPOSANTS UI
# ─────────────────────────────────────────────────────────────
_card_keys = set()

def movie_card(movie_id, title, genres, email, score=None, rank=None):
    global _card_keys
    base_key = f"{movie_id}_{rank}_{title[:6]}"
    count = 0
    while base_key + str(count) in _card_keys:
        count += 1
    ukey = base_key + str(count)
    _card_keys.add(ukey)

    app_ratings = get_app_ratings(email)
    my_note = app_ratings.get(str(movie_id), {}).get("rating") if str(movie_id) in app_ratings else None
    poster  = get_poster(title)
    year    = title[-5:-1] if title.endswith(')') else ""
    genres_short = " · ".join(genres.split("|")[:2]) if genres and genres != "(no genres listed)" else ""

    rk = f'<div style="position:absolute;top:8px;left:8px;background:#E50914;color:white;font-size:10px;font-weight:900;width:22px;height:22px;border-radius:4px;display:flex;align-items:center;justify-content:center;z-index:3;">#{rank}</div>' if rank else ""
    my = f'<div style="position:absolute;top:8px;right:8px;background:rgba(245,197,24,0.15);border:1px solid #f5c518;color:#f5c518;font-size:9px;font-weight:700;padding:2px 6px;border-radius:4px;z-index:3;">⭐{my_note}/5</div>' if my_note else ""

    st.markdown(f'<div style="background:#141414;border-radius:8px;overflow:hidden;margin-bottom:6px;position:relative;">{rk}{my}', unsafe_allow_html=True)

    if poster:
        st.image(poster, use_container_width=True)
    else:
        st.markdown(f'<div style="aspect-ratio:2/3;background:#1e1e1e;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#374151;"><div style="font-size:28px;">🎬</div><div style="font-size:9px;margin-top:6px;text-align:center;padding:0 8px;">{title[:30]}</div></div>', unsafe_allow_html=True)

    score_txt = f"★ {score:.2f}" if score else ""
    st.markdown(f"""
    <div style="padding:9px 9px 4px;">
        <div style="font-size:11px;font-weight:700;color:white;margin-bottom:4px;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{title}">{title}</div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:3px;">
            <span style="color:#f5c518;font-size:11px;font-weight:700;">{score_txt}</span>
            <span style="color:#4b5563;font-size:10px;font-family:'JetBrains Mono',monospace;">{year}</span>
        </div>
        <div style="font-size:9px;color:#4b5563;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                    font-family:'JetBrains Mono',monospace;">{genres_short}</div>
    </div></div>""", unsafe_allow_html=True)

    with st.expander("⭐ Noter ce film"):
        note = st.radio("Note", [1,2,3,4,5], format_func=lambda x:"⭐"*x,
                        index=(my_note-1) if my_note else 2, key=f"r_{ukey}")
        if st.button("Enregistrer", key=f"b_{ukey}"):
            save_app_rating(email, movie_id, title, genres, note)
            st.success(f"Noté {note}/5 !")
            st.rerun()

def divider():
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,rgba(229,9,20,0.4),transparent);margin:20px 0;"></div>', unsafe_allow_html=True)

def section_title(icon, text, badge=""):
    b = f'<span style="font-size:11px;color:#4ade80;font-family:\'JetBrains Mono\',monospace;">{badge}</span>' if badge else ""
    st.markdown(f'<div style="font-size:19px;font-weight:700;color:white;margin:22px 0 14px;display:flex;align-items:center;gap:10px;">{icon} {text} {b}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
for k, v in [("logged_in", False), ("email", ""), ("username", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# PAGE AUTH
# ─────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown("""
    <div style="text-align:center;padding:52px 0 8px;">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:68px;color:#E50914;letter-spacing:4px;line-height:1;">
            CINE<span style="color:white;">AI</span></div>
        <div style="font-size:10px;color:#2d2d2d;letter-spacing:6px;text-transform:uppercase;
                    font-family:'JetBrains Mono',monospace;margin-top:4px;">
            Votre cinéma personnel</div>
    </div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        t1, t2 = st.tabs(["  Connexion  ", "  Inscription  "])

        with t1:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("login"):
                em = st.text_input("Email", placeholder="exemple@email.com")
                pw = st.text_input("Mot de passe", type="password", placeholder="••••••••")
                sb = st.form_submit_button("Se connecter")
            if sb:
                if not em or not pw:
                    st.error("Veuillez remplir tous les champs.")
                else:
                    ok, uname, err = login_user(em.strip().lower(), pw)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.email     = em.strip().lower()
                        st.session_state.username  = uname
                        st.rerun()
                    else:
                        st.error(f"❌ {err}")

        with t2:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("signup"):
                sn = st.text_input("Nom d'utilisateur", placeholder="JohnDoe")
                se = st.text_input("Email", placeholder="exemple@email.com")
                sp = st.text_input("Mot de passe", type="password", placeholder="8 caractères minimum")
                sc = st.text_input("Confirmer", type="password", placeholder="••••••••")
                sb2 = st.form_submit_button("Créer mon compte")
            if sb2:
                if not all([sn,se,sp,sc]):
                    st.error("Remplissez tous les champs.")
                elif len(sp) < 8:
                    st.error("Mot de passe trop court.")
                elif sp != sc:
                    st.error("Les mots de passe ne correspondent pas.")
                elif "@" not in se:
                    st.error("Email invalide.")
                else:
                    ok, msg = register_user(se.strip().lower(), sn.strip(), sp)
                    if ok:
                        st.session_state["signup_success"] = True
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

        if st.session_state.get("signup_success"):
            st.success("✅ Compte créé ! Connectez-vous dans l'onglet Connexion.")
            st.session_state["signup_success"] = False
    st.stop()

# ─────────────────────────────────────────────────────────────
# APP PRINCIPALE
# ─────────────────────────────────────────────────────────────
email        = st.session_state.email
username     = st.session_state.username
app_ratings  = get_app_ratings(email)
nb_notes     = len(app_ratings)
avg_note     = round(sum(v["rating"] for v in app_ratings.values()) / nb_notes, 1) if nb_notes else 0
_card_keys.clear()

# ── NAVBAR ──
st.markdown(f"""
<div style="background:rgba(0,0,0,0.97);padding:13px 32px;display:flex;align-items:center;
            justify-content:space-between;margin:-1rem -1rem 0;border-bottom:1px solid rgba(229,9,20,0.12);">
    <div style="font-family:'Bebas Neue',sans-serif;font-size:28px;color:#E50914;letter-spacing:3px;">
        CINE<span style="color:white;">AI</span></div>
    <div style="display:flex;align-items:center;gap:12px;">
        <span style="color:#4b5563;font-size:11px;font-family:'JetBrains Mono',monospace;">
            👤 {username} &nbsp;·&nbsp; {nb_notes} film{"s" if nb_notes!=1 else ""} noté{"s" if nb_notes!=1 else ""}
        </span>
    </div>
</div>""", unsafe_allow_html=True)

# ── HERO ──
st.markdown(f"""
<div style="background:linear-gradient(180deg,rgba(10,10,10,0)0%,#0a0a0a 100%),
             linear-gradient(90deg,rgba(0,0,0,0.82)30%,transparent),
             linear-gradient(135deg,#1a0000,#050005);
            padding:50px 32px 36px;margin:0 -1rem;min-height:240px;
            display:flex;flex-direction:column;justify-content:flex-end;overflow:hidden;position:relative;">
    <div style="position:absolute;top:-60px;right:-60px;width:380px;height:380px;
                background:radial-gradient(circle,rgba(229,9,20,0.05)0%,transparent 70%);"></div>
    <div style="font-family:'Bebas Neue',sans-serif;font-size:48px;color:white;line-height:1;
                margin-bottom:8px;letter-spacing:2px;">
        Bonjour, <span style="color:#E50914;">{username}</span>
    </div>
    <div style="font-size:13px;color:#6b7280;max-width:460px;line-height:1.8;">
        Découvrez des films qui vous correspondent. Notez ce que vous avez vu
        et laissez nos recommandations faire le reste.
    </div>
</div>""", unsafe_allow_html=True)

# ── ONGLETS ──
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Rechercher un Film",
    "🎬  Catalogue",
    "⭐  Mes Recommandations",
    "📋  Mes Films Notés"
])

# ─── TAB 1 : RECHERCHE ─────────────────────────────────────
with tab1:
    c1, c2 = st.columns([5, 1])
    with c1:
        query = st.text_input("", placeholder="Rechercher dans le catalogue... Ex: Toy Story, Matrix, Inception",
                              label_visibility="collapsed")
    with c2:
        go = st.button("🔍 Rechercher")

    if query:
        # Recherche dans le dataset MovieLens
        mask = movies_df['title'].str.contains(query, case=False, na=False)
        results = movies_df[mask].head(12)

        if len(results) > 0:
            section_title("🔍", "Résultats", f"{len(results)} films trouvés")
            cols = st.columns(4)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 4]:
                    avg = ratings_df[ratings_df['movieId']==row['movieId']]['rating'].mean()
                    movie_card(row['movieId'], row['title'], row['genres'], email,
                               score=avg if not np.isnan(avg) else None)
        else:
            # Fallback OMDB si pas trouvé dans MovieLens
            st.info("Film non trouvé dans le catalogue MovieLens. Recherche en ligne...")
            omdb_results = search_omdb(query)
            if omdb_results:
                section_title("🔍", "Résultats OMDB", f"{len(omdb_results)} films")
                cols = st.columns(4)
                for i, m in enumerate(omdb_results[:8]):
                    with cols[i % 4]:
                        poster = m.get("Poster")
                        year   = m.get("Year","")
                        title  = m.get("Title","—")
                        st.markdown(f'<div style="background:#141414;border-radius:8px;overflow:hidden;margin-bottom:6px;">', unsafe_allow_html=True)
                        if poster and poster != "N/A":
                            st.image(poster, use_container_width=True)
                        else:
                            st.markdown('<div style="aspect-ratio:2/3;background:#1e1e1e;display:flex;align-items:center;justify-content:center;font-size:28px;">🎬</div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="padding:9px;"><div style="font-size:11px;font-weight:700;color:white;">{title}</div><div style="font-size:10px;color:#4b5563;font-family:\'JetBrains Mono\',monospace;">{year}</div></div></div>', unsafe_allow_html=True)
            else:
                st.warning("Aucun résultat trouvé.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;color:#1f2937;">
            <div style="font-size:50px;margin-bottom:12px;">🎬</div>
            <div style="font-size:14px;font-weight:600;color:#374151;">
                Tapez le nom d'un film pour le retrouver dans le catalogue</div>
            <div style="font-size:12px;color:#1f2937;margin-top:6px;">
                9 742 films disponibles · 1902 → 2018</div>
        </div>""", unsafe_allow_html=True)

# ─── TAB 2 : CATALOGUE ─────────────────────────────────────
with tab2:
    col_g, col_s, col_n, col_y = st.columns([2, 2, 1, 1])
    with col_g:
        all_genres = sorted(set(
            g for gs in movies_df['genres'].str.split('|') for g in gs
            if g != "(no genres listed)"
        ))
        genre_sel = st.selectbox("Genre", ["Tous"] + all_genres, label_visibility="collapsed")
    with col_s:
        sort_by = st.selectbox("Trier par", ["Les mieux notés", "Les plus notés", "Récents"], label_visibility="collapsed")
    with col_n:
        n_display = st.selectbox("Afficher", [8, 12, 20], label_visibility="collapsed")
    with col_y:
        all_years = sorted(
            movies_df['title'].str.extract(r'\((\d{4})\)')[0].dropna().unique().astype(int),
            reverse=True
        )
        year_sel = st.selectbox("Année", ["Toutes"] + [str(y) for y in all_years], label_visibility="collapsed")

    divider()

    # Filtrer par genre
    df_cat = movies_df.copy()
    if genre_sel != "Tous":
        df_cat = df_cat[df_cat['genres'].str.contains(genre_sel, na=False)]

    # Filtrer par année
    if year_sel != "Toutes":
        df_cat = df_cat[df_cat['title'].str.contains(f"({year_sel})", regex=False, na=False)]

    # Joindre les stats
    stats = ratings_df.groupby('movieId')['rating'].agg(mean='mean', count='count').reset_index()
    df_cat = df_cat.merge(stats, on='movieId', how='left')
    df_cat['mean']  = df_cat['mean'].fillna(0)
    df_cat['count'] = df_cat['count'].fillna(0)

    if sort_by == "Les mieux notés":
        df_cat = df_cat[df_cat['count'] >= 20].sort_values('mean', ascending=False)
    elif sort_by == "Les plus notés":
        df_cat = df_cat.sort_values('count', ascending=False)
    else:
        df_cat['year'] = df_cat['title'].str.extract(r'\((\d{4})\)').astype(float)
        df_cat = df_cat.sort_values('year', ascending=False)

    df_cat = df_cat.head(n_display)
    section_title("🎬", f"Catalogue — {genre_sel}", f"{len(df_cat)} films")

    cols = st.columns(4)
    for i, (_, row) in enumerate(df_cat.iterrows()):
        with cols[i % 4]:
            movie_card(row['movieId'], row['title'], row['genres'], email,
                       score=row['mean'] if row['mean'] > 0 else None, rank=i+1)

# ─── TAB 3 : RECOMMANDATIONS SVD ──────────────────────────
with tab3:
    if nb_notes < 3:
        st.markdown(f"""
        <div style="text-align:center;padding:60px 20px;background:#141414;border-radius:12px;
                    border:1px solid rgba(255,255,255,0.06);margin-top:10px;">
            <div style="font-size:46px;margin-bottom:14px;">🤖</div>
            <div style="font-size:17px;font-weight:700;color:white;margin-bottom:8px;">
                Notez au moins 3 films pour activer le moteur SVD</div>
            <div style="font-size:13px;color:#4b5563;line-height:1.7;">
                Vous avez noté <b style="color:#E50914;">{nb_notes}</b> film{"s" if nb_notes!=1 else ""}.<br>
                Allez dans <b style="color:white;">Rechercher</b> ou <b style="color:white;">Catalogue</b> pour noter des films.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("🤖 Le modèle SVD calcule vos recommandations..."):
            recos = get_svd_recommendations(app_ratings, n=12)

        # Stats utilisateur
        best = max(app_ratings.values(), key=lambda x: x["rating"])
        p1, p2, p3 = st.columns(3)
        for col, val, lbl in [
            (p1, str(nb_notes), "Films notés"),
            (p2, f"{avg_note} ⭐", "Note moyenne"),
            (p3, best["title"][:18]+("…" if len(best["title"])>18 else ""), f"Coup de ❤️ ({best['rating']}/5)")
        ]:
            with col:
                st.markdown(f"""
                <div style="background:#141414;border:1px solid rgba(255,255,255,0.07);border-radius:8px;
                            padding:14px 18px;border-bottom:2px solid #E50914;">
                    <div style="font-size:20px;font-weight:900;color:white;">{val}</div>
                    <div style="font-size:9px;color:#4b5563;letter-spacing:2px;text-transform:uppercase;
                                font-family:'JetBrains Mono',monospace;margin-top:3px;">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        divider()
        section_title("🎯", "Recommandé pour vous", f"{len(recos)} films")

        if recos:
            cols = st.columns(4)
            for i, m in enumerate(recos):
                with cols[i % 4]:
                    movie_card(m['movieId'], m['title'], m.get('genres',''),
                               email, score=m.get('score'), rank=i+1)
        else:
            st.info("Notez des films pour obtenir des recommandations.")

# ─── TAB 4 : MES FILMS NOTÉS ──────────────────────────────
with tab4:
    if not app_ratings:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;color:#1f2937;">
            <div style="font-size:46px;margin-bottom:12px;">📋</div>
            <div style="font-size:14px;color:#374151;">Vous n'avez encore noté aucun film.</div>
        </div>""", unsafe_allow_html=True)
    else:
        section_title("📋", "Mes films notés", f"{nb_notes} films")
        rows = sorted(app_ratings.items(), key=lambda x: x[1]["rating"], reverse=True)
        for mid_str, info in rows:
            stars = "⭐" * info["rating"]
            date  = info["date"][:10]
            poster = get_poster(info["title"])

            c1, c2, c3, c4, c5 = st.columns([1, 3, 2, 1, 1])

            with c1:
                if poster:
                    st.image(poster, width=52)
                else:
                    st.markdown('<div style="width:52px;height:72px;background:#1e1e1e;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:18px;">🎬</div>', unsafe_allow_html=True)

            with c2:
                g = info.get("genres","").replace("|"," · ")[:40]
                st.markdown(f'<div style="padding-top:6px;"><div style="font-size:13px;font-weight:600;color:white;">{info["title"]}</div><div style="font-size:10px;color:#4b5563;font-family:\'JetBrains Mono\',monospace;">{g}</div></div>', unsafe_allow_html=True)

            with c3:
                st.markdown(f'<div style="font-size:14px;color:#f5c518;padding-top:10px;">{stars}</div>', unsafe_allow_html=True)

            with c4:
                # Bouton modifier
                if st.button("✏️ Modifier", key=f"edit_{mid_str}"):
                    st.session_state[f"editing_{mid_str}"] = True

            with c5:
                # Bouton supprimer
                if st.button("🗑️ Supprimer", key=f"del_{mid_str}"):
                    data = _load_json(RATINGS_FILE)
                    if email in data and mid_str in data[email]:
                        del data[email][mid_str]
                        _save_json(RATINGS_FILE, data)
                        st.rerun()

            # Formulaire de modification (apparaît sous le film)
            if st.session_state.get(f"editing_{mid_str}"):
                with st.container():
                    st.markdown(f'<div style="background:#1e1e1e;border:1px solid rgba(229,9,20,0.3);border-radius:8px;padding:14px 18px;margin:6px 0 10px;">',unsafe_allow_html=True)
                    nouvelle_note = st.radio(
                        f"Nouvelle note pour {info['title'][:30]}",
                        [1, 2, 3, 4, 5],
                        format_func=lambda x: "⭐" * x,
                        index=info["rating"] - 1,
                        key=f"new_note_{mid_str}",
                        horizontal=True
                    )
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("✅ Enregistrer", key=f"save_edit_{mid_str}"):
                            save_app_rating(email, int(mid_str), info["title"], info.get("genres",""), nouvelle_note)
                            st.session_state[f"editing_{mid_str}"] = False
                            st.rerun()
                    with col_cancel:
                        if st.button("❌ Annuler", key=f"cancel_{mid_str}"):
                            st.session_state[f"editing_{mid_str}"] = False
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div style="height:1px;background:rgba(255,255,255,0.04);margin:4px 0;"></div>', unsafe_allow_html=True)

# ── DÉCONNEXION ──
divider()
_, lc = st.columns([5, 1])
with lc:
    if st.button("⏏ Déconnexion"):
        st.session_state.logged_in = False
        st.session_state.email     = ""
        st.session_state.username  = ""
        st.rerun()