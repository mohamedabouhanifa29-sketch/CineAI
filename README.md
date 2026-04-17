# 🎬 CineAI — Système de Recommandation de Films

Application web de recommandation de films personnalisée, développée dans le cadre d'un projet de fin d'année.

---

## 📌 Fonctionnalités

- **Inscription & Connexion** sécurisée (mot de passe hashé SHA-256)
- **Recherche** de films dans un catalogue de 9 742 titres
- **Parcourir** le catalogue par genre et par tri
- **Noter** des films de 1 à 5 étoiles
- **Recommandations personnalisées** basées sur vos notes

---

## 🗂️ Structure du projet

```
movie-recommender/
├── app/
│   └── app.py                  ← Application Streamlit
├── data/
│   ├── movies.csv              ← 9 742 films (MovieLens Latest Small)
│   └── ratings.csv             ← 100 836 notes utilisateurs
├── models/
│   └── svd_model.pkl           ← Modèle SVD entraîné
├── notebooks/
│   ├── 01_EDA.ipynb            ← Analyse exploratoire des données
│   └── 02_modele.ipynb         ← Entraînement du modèle SVD
├── rapport/                    ← Rapport du projet
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation et lancement

### 1. Cloner le dépôt

```bash
git clone https://github.com/TON_USERNAME/movie-recommender.git
cd movie-recommender
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l'application

```bash
cd app
python -m streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`

---

## 🧠 Modèle utilisé

**Algorithme :** Décomposition en Valeurs Singulières (SVD) via `scikit-learn`

| Métrique | Valeur |
|----------|--------|
| RMSE | 0.6217 |
| MAE | 0.3998 |

**Pipeline :**
1. Construction de la matrice Utilisateur × Film (610 × 9724)
2. Centrage des notes par utilisateur (suppression du biais)
3. Décomposition SVD en 50 facteurs latents
4. Reconstruction de la matrice pour prédire toutes les notes manquantes
5. Recommandation des films les mieux prédits non encore vus

---

## 📦 Dataset

**MovieLens Latest Small** — GroupLens Research, University of Minnesota

| Statistique | Valeur |
|------------|--------|
| Notes | 100 836 |
| Films | 9 742 |
| Utilisateurs | 610 |
| Période | 1996 – 2018 |

---

## 🛠️ Technologies utilisées

- **Python** — Langage principal
- **Streamlit** — Interface web
- **Pandas / NumPy** — Traitement des données
- **Scikit-learn** — Modèle SVD
- **OMDB API** — Affiches des films
> ⚠️ Le fichier `models/svd_model.pkl` n'est pas inclus (trop lourd).
> Pour le générer, lance le notebook `notebooks/02_modele.ipynb`.