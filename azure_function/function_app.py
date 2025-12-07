"""
Azure Function pour le système de recommandation My Content
Expose une API HTTP pour générer des recommandations d'articles
"""

import azure.functions as func
import logging
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from azure.storage.blob import BlobServiceClient
import os
from io import BytesIO

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Variables globales pour mettre en cache les données
_embeddings = None
_article_ids = None
_articles_metadata = None
_user_clicks = None

def load_data_from_blob():
    """
    Charge les données depuis Azure Blob Storage.
    Utilise un cache global pour éviter de recharger à chaque requête.
    """
    global _embeddings, _article_ids, _articles_metadata, _user_clicks
    
    # Si déjà chargé, retourner les données en cache
    if _embeddings is not None:
        logging.info("Utilisation des données en cache")
        return
    
    logging.info("Chargement des données depuis Blob Storage...")
    
    try:
        # Récupérer la connection string depuis les variables d'environnement
        connect_str = os.environ.get('AzureWebJobsStorage')
        
        if not connect_str:
            raise ValueError("AzureWebJobsStorage connection string non trouvée")
        
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_client = blob_service_client.get_container_client("data")
        
        # 1. Charger les embeddings
        logging.info("Chargement des embeddings...")
        blob_client = container_client.get_blob_client("articles_embeddings.pickle")
        embeddings_data = pickle.loads(blob_client.download_blob().readall())
        
        # Les embeddings sont un array numpy
        _embeddings = embeddings_data
        
        # 2. Charger les métadonnées pour obtenir les article_ids
        logging.info("Chargement des métadonnées...")
        blob_client = container_client.get_blob_client("articles_metadata.csv")
        metadata_bytes = blob_client.download_blob().readall()
        _articles_metadata = pd.read_csv(BytesIO(metadata_bytes))
        
        # Mapper les embeddings aux article_ids
        _article_ids = _articles_metadata['article_id'].iloc[:len(_embeddings)].tolist()
        
        logging.info(f"✓ {len(_article_ids)} embeddings chargés")
        
        # 3. Charger quelques fichiers de clicks
        logging.info("Chargement des clicks...")
        all_clicks = []
        
        # Liste des blobs dans le dossier clicks/
        blob_list = container_client.list_blobs(name_starts_with="clicks/")
        click_files = [blob.name for blob in blob_list if blob.name.endswith('.csv')][:10]
        
        if not click_files:
            # Si pas de dossier clicks/, chercher les fichiers à la racine
            blob_list = container_client.list_blobs()
            click_files = [blob.name for blob in blob_list if 'clicks_hour' in blob.name][:10]
        
        for blob_name in click_files:
            try:
                blob_client = container_client.get_blob_client(blob_name)
                click_bytes = blob_client.download_blob().readall()
                df = pd.read_csv(BytesIO(click_bytes))
                all_clicks.append(df)
            except Exception as e:
                logging.warning(f"Erreur lors du chargement de {blob_name}: {e}")
        
        if all_clicks:
            _user_clicks = pd.concat(all_clicks, ignore_index=True)
            logging.info(f"✓ {len(_user_clicks)} clicks chargés")
        else:
            _user_clicks = pd.DataFrame()
            logging.warning("Aucun fichier de clicks chargé")
        
        logging.info("✓ Données chargées avec succès!")
        
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données: {e}")
        raise


def get_user_history(user_id: int):
    """Récupère l'historique d'un utilisateur."""
    if _user_clicks is None or len(_user_clicks) == 0:
        return []
    
    user_data = _user_clicks[_user_clicks['user_id'] == user_id]
    if len(user_data) == 0:
        return []
    
    return user_data['click_article_id'].unique().tolist()


def get_similar_articles(article_id: int, top_k: int = 10):
    """Trouve les articles similaires à un article donné."""
    try:
        article_idx = _article_ids.index(article_id)
        article_embedding = _embeddings[article_idx].reshape(1, -1)
        similarities = cosine_similarity(article_embedding, _embeddings)[0]
        
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        results = [(int(_article_ids[idx]), float(similarities[idx])) for idx in similar_indices]
        return results
    except ValueError:
        logging.warning(f"Article {article_id} non trouvé")
        return []


def get_article_info(article_id: int):
    """Récupère les infos d'un article."""
    article_row = _articles_metadata[_articles_metadata['article_id'] == article_id]
    
    if len(article_row) == 0:
        return {
            'article_id': int(article_id),
            'title': f'Article {article_id}',
            'category': 'N/A',
            'words_count': 0
        }
    
    article = article_row.iloc[0]
    return {
        'article_id': int(article_id),
        'title': f'Article {article_id}',
        'category': int(article.get('category_id', 0)),
        'words_count': int(article.get('words_count', 0))
    }


def recommend_for_user(user_id: int, top_n: int = 5):
    """Génère des recommandations pour un utilisateur."""
    user_history = get_user_history(user_id)
    
    if not user_history:
        # Recommandations par défaut (articles populaires ou premiers articles)
        logging.info(f"Pas d'historique pour user {user_id}")
        results = []
        for i, article_id in enumerate(_article_ids[:top_n]):
            article_info = get_article_info(article_id)
            article_info['recommendation_score'] = float(top_n - i)
            results.append(article_info)
        return results
    
    # Calculer les recommandations basées sur la similarité
    recommendations_scores = {}
    
    for article_id in user_history[:5]:
        similar_articles = get_similar_articles(article_id, top_k=20)
        
        for recommended_id, score in similar_articles:
            if recommended_id not in user_history:
                if recommended_id not in recommendations_scores:
                    recommendations_scores[recommended_id] = 0
                recommendations_scores[recommended_id] += score
    
    sorted_recommendations = sorted(
        recommendations_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    results = []
    for article_id, score in sorted_recommendations:
        article_info = get_article_info(article_id)
        article_info['recommendation_score'] = float(score)
        results.append(article_info)
    
    return results


@app.route(route="recommend", methods=["GET"])
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    """
    Endpoint HTTP pour générer des recommandations.
    
    Paramètres:
        user_id (int): ID de l'utilisateur
        top_n (int, optional): Nombre de recommandations (défaut: 5)
    
    Exemple:
        GET /api/recommend?user_id=123&top_n=5
    """
    logging.info('Requête de recommandation reçue')
    
    try:
        # Charger les données (utilise le cache si déjà chargé)
        load_data_from_blob()
        
        # Récupérer les paramètres
        user_id = req.params.get('user_id')
        top_n = req.params.get('top_n', '5')
        
        if not user_id:
            return func.HttpResponse(
                json.dumps({"error": "Paramètre 'user_id' manquant"}),
                mimetype="application/json",
                status_code=400
            )
        
        try:
            user_id = int(user_id)
            top_n = int(top_n)
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "user_id et top_n doivent être des entiers"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Générer les recommandations
        logging.info(f"Génération de {top_n} recommandations pour user {user_id}")
        recommendations = recommend_for_user(user_id, top_n)
        
        response = {
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
        return func.HttpResponse(
            json.dumps(response, ensure_ascii=False),
            mimetype="application/json",
            status_code=200
        )
    
    except Exception as e:
        logging.error(f"Erreur: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Endpoint de santé pour vérifier que la fonction est opérationnelle."""
    return func.HttpResponse(
        json.dumps({"status": "healthy", "service": "My Content Recommendation API"}),
        mimetype="application/json",
        status_code=200
    )