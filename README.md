# 🚗 Carsify — Estimation de prix de véhicules

Projet de **régression supervisée** qui estime le prix de vente d'un véhicule à partir de ses
caractéristiques (année, kilométrage, carburant, boîte de vitesse, type de vendeur, nombre de
propriétaires), avec une **interface Streamlit** pour la prédiction.

- 🧹 **Nettoyage des données** : imputation, détection d'outliers par IQR, encodage des variables
  catégorielles, standardisation.
- 🤖 **Benchmark de modèles** : 6 modèles comparés sur MAE / RMSE / R², puis optimisation des
  hyperparamètres des deux meilleurs par `GridSearchCV`.
- 🚀 **Mise en production** : le pipeline complet (prétraitement + modèle) est sérialisé en un
  seul artefact `.pkl` consommé directement par l'application.
- 📊 **Rapports** : toutes les figures d'analyse sont exportées dans `reports/`.

---

## 📊 Résultats

Benchmark réalisé sur le jeu de test (20 % des données, `random_state=42`).
Source : [`reports/post-processing/model_benchmakr_results.csv`](reports/post-processing/model_benchmakr_results.csv)

| Modèle | MAE (DH) | RMSE (DH) | R² |
|---|---:|---:|---:|
| **XGBoost (optimisé)** | 132 531.84 | 185 066.15 | **0.626** |
| **Random Forest (optimisé)** ← *modèle livré* | 131 103.66 | 186 471.30 | **0.620** |
| XGBoost | 132 099.03 | 192 601.50 | 0.595 |
| Random Forest | 131 770.67 | 194 068.63 | 0.588 |
| Régression linéaire | 150 076.55 | 197 630.24 | 0.573 |
| SVR (RBF) | 237 695.09 | 313 768.10 | -0.076 |

**Modèle en production** : `RandomForestRegressor(n_estimators=200, max_depth=10,
min_samples_split=5)` enveloppé dans un `Pipeline` `ColumnTransformer`
(`StandardScaler` sur `year`/`km_driven` + `OneHotEncoder(handle_unknown='ignore')` sur les
variables catégorielles).

> **Note** : XGBoost optimisé obtient un meilleur R² (+0.006) mais l'artefact exporté par le
> notebook 02 est le Random Forest (la variable `best_model` pointe vers `grid_rf.best_estimator_`).
> Les deux performances sont néanmoins quasi équivalentes.

---

## 🗂️ Structure du projet

```
carsify/
├── app.py                        # point d'entrée (vide, l'app réelle est dans src/)
├── data/
│   ├── raw/car-price.csv         # 4 340 véhicules - dataset source
│   └── processed/                # datasets nettoyés / encodés (générés)
├── models/
│   └── best_car_price_model.pkl  # pipeline sérialisé (généré)
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb # EDA, nettoyage, encodage, séparation train/test
│   └── 02_model_training.ipynb   # benchmark, GridSearchCV, export du modèle
├── reports/
│   ├── pre-cleaning/             # figures EDA sur les données brutes
│   └── post-processing/          # figures après nettoyage + résultats du benchmark
├── src/
│   ├── app.py                    # application Streamlit
│   ├── data_preprocessing.py     # (à compléter)
│   ├── train.py                  # (à compléter)
│   └── test_inference.py         # test de prédiction en conditions réelles
└── requirements.txt
```

> `data/` et `models/` sont ignorés par Git (voir `.gitignore`) : le dataset, les datasets
> intermédiaires et le modèle doivent être **régénérés en exécutant les notebooks**.

---

## 🚀 Installation

Prérequis : **Python 3.12**

```bash
git clone <url-du-depot>
cd carsify

python3 -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

Dépendances principales : `pandas`, `scikit-learn`, `xgboost`, `joblib`, `matplotlib`,
`seaborn`, `streamlit` (+ `jupyter` pour les notebooks, à installer séparément si besoin).

---

## 📖 Utilisation

### 1. Reconstruire la chaîne de traitement

Les notebooks utilisent des chemins relatifs : **lancer Jupyter depuis le dossier `notebooks/`**
(cases 1 → 2 dans l'ordre).

```bash
cd notebooks
jupyter lab
```

- **`01_eda_and_cleaning.ipynb`** produit :
  - `../data/processed/cleaned_car_price_raw.csv` (colonnes catégorielles conservées)
  - `../data/processed/cleaned_car_price.csv` (données encodées)
  - les figures de `../reports/pre-cleaning/` et `../reports/post-processing/`
- **`02_model_training.ipynb`** produit :
  - `../reports/post-processing/model_benchmakr_results.csv`
  - les figures d'évaluation et de distribution des résidus
  - `../models/best_car_price_model.pkl`

### 2. Lancer l'application

`src/app.py` charge le modèle via un chemin relatif : **lancer Streamlit depuis la racine du dépôt.**

```bash
streamlit run src/app.py
```

L'interface est alors disponible sur <http://localhost:8501>. Renseigner le formulaire
(année, kilométrage, carburant, vendeur, boîte, propriétaires) puis cliquer sur
**« 💡 Estimer le prix »**.

### 3. Tester la prédiction en ligne de commande

```bash
python src/test_inference.py
```

Charge `models/best_car_price_model.pkl` et affiche le prix estimé pour un véhicule d'exemple.

---

## 🔍 Détail du traitement

### Données

| Colonne | Type | Rôle |
|---|---|---|
| `name` | texte | **supprimée** (texte libre, aucune information métier) |
| `year` | numérique | prédicteur — ancienneté du véhicule |
| `km_driven` | numérique | prédicteur — usure |
| `fuel` | catégoriel | prédicteur — Diesel / Petrol / CNG / LPG / Electric |
| `seller_type` | catégoriel | prédicteur — Individual / Dealer / Trustmark Dealer |
| `transmission` | catégoriel | prédicteur — Manual / Automatic |
| `owner` | catégoriel | prédicteur — historique de propriété |
| `selling_price` | numérique | **variable cible** |

### Étapes du nettoyage (`01_eda_and_cleaning.ipynb`)

1. **Doublons** : suppression des lignes dupliquées.
2. **Valeurs manquantes** :
   - médiane pour `year` et `km_driven` ;
   - mode pour `fuel`, `seller_type` et `owner`.
3. **Outliers** (méthode IQR, bornes à 1.5 × IQR) :
   - `selling_price` → **plafonnement** (capping) à la borne haute : conserve les véhicules
     haut de gamme sans les supprimer ;
   - `km_driven` → **plafonnement** à la borne haute : supprime le bruit de saisie ;
   - `year` → **suppression** des véhicules trop anciens : recentre le modèle sur le marché principal.
4. **Encodage** : `get_dummies(drop_first=True)` sur les 4 variables catégorielles.
5. **Séparation** : 80 % entraînement / 20 % test, `random_state=42` (reproductible).
6. **Standardisation** : `StandardScaler` ajusté **uniquement sur le jeu d'entraînement**.

Résultat : **4 235 lignes** exploitables à partir de 4 340 lignes brutes.

### Pipeline servi par l'application

L'application ne reproduit pas ces étapes à la main : elle utilise le `Pipeline` sérialisé, qui
applique lui-même le `StandardScaler`, le `OneHotEncoder` puis le modèle. Les valeurs categorielles
inconnues sont ignorées plutôt que de provoquer une erreur (`handle_unknown='ignore'`).

---

## ⚠️ Limites connues & pistes d'amélioration

- **R² ≈ 0.62** : le modèle explique environ 62 % de la variance du prix. L'erreur absolue
  moyenne reste de l'ordre de **131 k DH**, ce qui est important par rapport au prix prédit.
  Les véhicules de luxe font partie des cas les moins bien captés.
- **Optimisation partielle** : seuls Random Forest et XGBoost ont été optimisés. Ridge, Lasso,
  Gradient Boosting et Decision Tree sont commentés dans le notebook 02 et méritent un essai
  (`feature_importance`, `permutation_importance`, `stacking`).
- **Feature engineering** : la variable `name` est supprimée alors qu'elle contient
  information exploitable (marque, modèle, cylindrée). Un `TargetEncoder` ou une extraction de
  marque donnerait probablement un gain significatif.
- **Validation** : une validation croisée *out-of-fold* sur l'ensemble final compléterait
  l'évaluation sur une seule split de test.
- **Données** : le dataset source est un jeu de données public utilisé pour un marché différent ;
  l'affichage en **DH** suppose une conversion qui n'est pas appliquée dans le pipeline.
  À valider avant tout usage commercial.
- **Passage en production** : l'application n'a ni authentification ni historique des prédictions ;
  `src/train.py` et `src/data_preprocessing.py` sont encore vides et devraient héberger la logique
  des notebooks pour rendre l'entraînement reproductible en ligne de commande.
- **Tests** : il n'existe pas encore de suite de tests automatisés (pytest n'est pas installé).

---

## 🛠️ Technologies

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-4E2A8A?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
