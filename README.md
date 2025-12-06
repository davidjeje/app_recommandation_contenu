# 📚 My Content - MVP

Système de recommandation d'articles pour encourager la lecture.

## 🎯 Objectif

My Content est une start-up qui recommande des contenus pertinents (articles et livres) à ses utilisateurs. Ce MVP démontre la capacité à générer des recommandations personnalisées basées sur l'historique de lecture.

## 🏗️ Architecture

**Architecture serverless (Option 2):**
- **Application Streamlit**: Interface utilisateur locale
- **Azure Function**: Système de recommandation en serverless
- **Azure Blob Storage**: Stockage des données (embeddings, métadonnées, clicks)

## 🧠 Système de Recommandation

**Type**: Content-Based Filtering avec embeddings

**Fonctionnement**:
1. Récupération de l'historique utilisateur (articles lus)
2. Calcul de similarité cosinus entre embeddings d'articles
3. Agrégation des scores de recommandation
4. Retour du top 5 des articles recommandés

**Avantages**:
- ✅ Fonctionne pour les nouveaux utilisateurs (pas de cold start)
- ✅ Utilise des embeddings pré-entraînés (250 dimensions)
- ✅ Rapide et léger pour un déploiement serverless
- ✅ Facilement extensible

## 📦 Installation

### Prérequis
- Python 3.11+ (testé avec 3.13.1)
- Poetry
- Compte Azure (pour le déploiement)

### Configuration locale

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/my-content-mvp.git
cd my-content-mvp
```

2. **Installer les dépendances avec Poetry**
```bash
poetry install
```

3. **Préparer les données**
   
Téléchargez les données depuis Kaggle et placez-les dans le dossier `data/`:
- `data/clicks/` - Fichiers CSV des interactions
- `data/articles_metadata.csv` - Métadonnées des articles
- `data/articles_embeddings.pickle` - Embeddings des articles

Structure attendue:
```
my-content-mvp/
├── data/
│   ├── clicks/
│   │   ├── clicks_hour_0.csv
│   │   ├── clicks_hour_1.csv
│   │   └── ...
│   ├── articles_metadata.csv
│   └── articles_embeddings.pickle
```

## 🚀 Utilisation

### Tester en local (Streamlit)

```bash
poetry run streamlit run streamlit_app/app.py
```

L'application s'ouvrira dans votre navigateur à `http://localhost:8501`

### Tester le moteur de recommandation

```bash
poetry run python my_content/recommendation_engine.py
```

## 📊 Données utilisées

**Source**: [Kaggle - News Portal User Interactions by Globo.com](https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom)

**Période**: 1-16 octobre 2017  
**Volume**: ~3M de clics, 314K utilisateurs, 46K articles

**Fichiers**:
- `clicks/`: Interactions utilisateurs (pages vues)
- `articles_metadata.csv`: Métadonnées de 364K articles
- `articles_embeddings.pickle`: Embeddings de dimension 250

## 🔧 Déploiement Azure Functions

### 1. Configuration Azure

```bash
# Connexion à Azure
az login

# Créer un Resource Group
az group create --name my-content-rg --location westeurope

# Créer un Storage Account
az storage account create --name mycontentstorage --resource-group my-content-rg --location westeurope

# Créer une Function App
az functionapp create --name my-content-func --resource-group my-content-rg --storage-account mycontentstorage --consumption-plan-location westeurope --runtime python --runtime-version 3.11
```

### 2. Upload des données vers Blob Storage

```bash
# Upload des fichiers
az storage blob upload-batch --account-name mycontentstorage --destination data --source ./data
```

### 3. Déploiement de la fonction

```bash
cd azure_function
func azure functionapp publish my-content-func
```

## 🧪 Tests

```bash
poetry run pytest
```

## 📁 Structure du projet

```
my-content-mvp/
├── my_content/              # Package principal
│   ├── __init__.py
│   └── recommendation_engine.py
├── streamlit_app/           # Application Streamlit
│   └── app.py
├── azure_function/          # Azure Function
│   ├── __init__.py
│   ├── function_app.py
│   └── requirements.txt
├── data/                    # Données (non versionnées)
├── tests/                   # Tests unitaires
├── presentation/            # Slides PowerPoint
├── pyproject.toml           # Configuration Poetry
├── README.md
└── .gitignore

```

## 🎤 Présentation

Voir `presentation/MyContent_Presentation.pdf` pour les slides de présentation à Samia.

## 🔮 Architecture cible

Pour gérer l'ajout de nouveaux utilisateurs et articles:

1. **Pipeline de données en temps réel**
   - Ingestion des nouveaux clicks via Azure Event Hubs
   - Mise à jour incrémentale des embeddings

2. **Système hybride**
   - Content-based pour les nouveaux users/articles
   - Collaborative filtering pour les utilisateurs établis

3. **Cache et optimisation**
   - Redis pour les recommandations fréquentes
   - Pré-calcul des similarités pour les articles populaires

## 👥 Équipe

- **CTO**: Vous
- **CEO**: Samia

## 📄 Licence

Usage académique uniquement (conformément aux conditions des données Globo.com)

## 🙏 Remerciements

Données fournies par Globo.com pour la recherche académique.