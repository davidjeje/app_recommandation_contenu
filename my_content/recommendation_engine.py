"""
Système de recommandation content-based pour My Content MVP
Utilise les embeddings d'articles pour recommander du contenu similaire
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentRecommender:
    """
    Système de recommandation basé sur le contenu des articles.
    Utilise la similarité cosinus entre les embeddings d'articles.
    """
    
    def __init__(self, data_path: str = "data"):
        """
        Initialise le système de recommandation.
        
        Args:
            data_path: Chemin vers le dossier contenant les données
        """
        self.data_path = Path(data_path)
        self.embeddings = None
        self.articles_metadata = None
        self.user_clicks = None
        self.article_ids = None
        
        logger.info("Initialisation du système de recommandation...")
        self._load_data()
    
    def _load_data(self):
        """Charge toutes les données nécessaires."""
        try:
            # 1. Charger les embeddings des articles
            logger.info("Chargement des embeddings...")
            embeddings_path = self.data_path / "articles_embeddings.pickle"
            with open(embeddings_path, 'rb') as f:
                embeddings_data = pickle.load(f)
            
            # Gérer différents formats possibles du fichier pickle
            if isinstance(embeddings_data, dict):
                # Format: {article_id: embedding_vector}
                self.article_ids = list(embeddings_data.keys())
                self.embeddings = np.array([embeddings_data[aid] for aid in self.article_ids])
            elif isinstance(embeddings_data, tuple) and len(embeddings_data) == 2:
                # Format: (article_ids_list, embeddings_matrix)
                self.article_ids = embeddings_data[0]
                self.embeddings = embeddings_data[1]
            elif isinstance(embeddings_data, np.ndarray):
                # Format: seulement la matrice d'embeddings (sans IDs)
                # Dans ce cas, on doit récupérer les IDs depuis les métadonnées
                logger.warning("Embeddings sans IDs détectés, chargement des métadonnées d'abord...")
                metadata_path = self.data_path / "articles_metadata.csv"
                temp_metadata = pd.read_csv(metadata_path)
                
                # Prendre les N premiers articles (N = nombre de lignes dans embeddings)
                self.article_ids = temp_metadata['article_id'].iloc[:len(embeddings_data)].tolist()
                self.embeddings = embeddings_data
            else:
                raise ValueError(f"Format d'embeddings non reconnu: {type(embeddings_data)}")
            
            logger.info(f"✓ {len(self.article_ids)} embeddings chargés (dimension: {self.embeddings.shape[1]})")
            
            # 2. Charger les métadonnées des articles
            logger.info("Chargement des métadonnées...")
            metadata_path = self.data_path / "articles_metadata.csv"
            self.articles_metadata = pd.read_csv(metadata_path)
            logger.info(f"✓ {len(self.articles_metadata)} articles chargés")
            
            # 3. Charger les clicks (on prend seulement quelques fichiers pour le MVP)
            logger.info("Chargement des clicks utilisateurs...")
            self._load_clicks()
            
            logger.info("✓ Système de recommandation prêt!")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise
    
    def _load_clicks(self):
        """
        Charge les fichiers de clicks et construit l'historique utilisateur.
        Pour le MVP, on charge seulement les premiers fichiers.
        """
        clicks_folder = self.data_path / "clicks"
        
        # Vérifier si le dossier existe
        if not clicks_folder.exists():
            logger.warning(f"Dossier clicks non trouvé: {clicks_folder}")
            self.user_clicks = pd.DataFrame()
            return
        
        click_files = sorted(list(clicks_folder.glob("*.csv")))[:10]  # Limiter à 10 fichiers pour le MVP
        
        if len(click_files) == 0:
            logger.warning(f"Aucun fichier CSV trouvé dans {clicks_folder}")
            self.user_clicks = pd.DataFrame()
            return
        
        all_clicks = []
        for file in click_files:
            try:
                df = pd.read_csv(file)
                all_clicks.append(df)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de {file}: {e}")
        
        if len(all_clicks) == 0:
            logger.warning("Aucun fichier de clicks chargé avec succès")
            self.user_clicks = pd.DataFrame()
        else:
            self.user_clicks = pd.concat(all_clicks, ignore_index=True)
            logger.info(f"✓ {len(self.user_clicks)} clicks chargés de {len(click_files)} fichiers")
    
    def get_user_history(self, user_id: int) -> List[int]:
        """
        Récupère l'historique de lecture d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Liste des article_ids lus par l'utilisateur
        """
        if self.user_clicks is None or len(self.user_clicks) == 0:
            return []
        
        user_data = self.user_clicks[self.user_clicks['user_id'] == user_id]
        
        if len(user_data) == 0:
            logger.warning(f"Aucun historique trouvé pour l'utilisateur {user_id}")
            return []
        
        return user_data['click_article_id'].unique().tolist()
    
    def get_similar_articles(self, article_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Trouve les articles les plus similaires à un article donné.
        
        Args:
            article_id: ID de l'article de référence
            top_k: Nombre d'articles similaires à retourner
            
        Returns:
            Liste de tuples (article_id, score_similarité)
        """
        try:
            # Trouver l'index de l'article dans notre liste
            article_idx = self.article_ids.index(article_id)
            
            # Calculer la similarité cosinus avec tous les autres articles
            article_embedding = self.embeddings[article_idx].reshape(1, -1)
            similarities = cosine_similarity(article_embedding, self.embeddings)[0]
            
            # Trier par similarité décroissante (en excluant l'article lui-même)
            similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            # Retourner les IDs et scores
            results = [(self.article_ids[idx], similarities[idx]) for idx in similar_indices]
            return results
            
        except ValueError:
            logger.warning(f"Article {article_id} non trouvé dans les embeddings")
            return []
    
    def recommend_for_user(self, user_id: int, top_n: int = 5) -> List[Dict]:
        """
        Génère des recommandations pour un utilisateur.
        
        Stratégie:
        1. Récupérer l'historique de l'utilisateur
        2. Pour chaque article lu, trouver des articles similaires
        3. Agréger et scorer les recommandations
        4. Retourner le top N
        
        Args:
            user_id: ID de l'utilisateur
            top_n: Nombre de recommandations à retourner
            
        Returns:
            Liste de dictionnaires contenant les recommandations
        """
        # Récupérer l'historique utilisateur
        user_history = self.get_user_history(user_id)
        
        if not user_history:
            # Si pas d'historique, recommander les articles les plus populaires
            logger.info(f"Pas d'historique pour user {user_id}, recommandations par défaut")
            return self._get_popular_articles(top_n)
        
        # Collecter les articles similaires pour chaque article lu
        recommendations_scores = {}
        
        for article_id in user_history[:5]:  # Limiter à 5 articles les plus récents
            similar_articles = self.get_similar_articles(article_id, top_k=20)
            
            for recommended_id, score in similar_articles:
                # Ne pas recommander des articles déjà lus
                if recommended_id not in user_history:
                    if recommended_id not in recommendations_scores:
                        recommendations_scores[recommended_id] = 0
                    recommendations_scores[recommended_id] += score
        
        # Trier par score décroissant
        sorted_recommendations = sorted(
            recommendations_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        # Enrichir avec les métadonnées
        results = []
        for article_id, score in sorted_recommendations:
            article_info = self._get_article_info(article_id)
            article_info['recommendation_score'] = float(score)
            results.append(article_info)
        
        return results
    
    def _get_article_info(self, article_id: int) -> Dict:
        """Récupère les informations d'un article depuis les métadonnées."""
        article_row = self.articles_metadata[
            self.articles_metadata['article_id'] == article_id
        ]
        
        if len(article_row) == 0:
            return {
                'article_id': article_id,
                'title': f'Article {article_id}',
                'category': 'N/A',
                'words_count': 0
            }
        
        article = article_row.iloc[0]
        # Si pas de titre, utiliser l'ID de l'article
        title = article.get('title', f'Article {article_id}')
        if pd.isna(title) or title == '':
            title = f'Article {article_id}'
            
        return {
            'article_id': int(article_id),
            'title': title,
            'category': article.get('category_id', 'N/A'),
            'words_count': int(article.get('words_count', 0))
        }
    
    def _get_popular_articles(self, top_n: int = 5) -> List[Dict]:
        """
        Retourne les articles les plus populaires (fallback).
        Utilisé quand un utilisateur n'a pas d'historique.
        """
        # Si pas de données de clicks, retourner les premiers articles des métadonnées
        if self.user_clicks is None or len(self.user_clicks) == 0:
            logger.info("Pas de données de clicks, utilisation des premiers articles")
            results = []
            for i, row in self.articles_metadata.head(top_n).iterrows():
                results.append({
                    'article_id': int(row['article_id']),
                    'title': row.get('title', f"Article {row['article_id']}"),
                    'category': row.get('category_id', 'N/A'),
                    'words_count': int(row.get('words_count', 0)),
                    'recommendation_score': float(top_n - len(results))
                })
            return results
        
        # Compter les clicks par article
        article_popularity = self.user_clicks['click_article_id'].value_counts().head(top_n)
        
        results = []
        for article_id, count in article_popularity.items():
            article_info = self._get_article_info(article_id)
            article_info['recommendation_score'] = float(count)
            results.append(article_info)
        
        return results
    
    def get_available_users(self, limit: int = 100) -> List[int]:
        """
        Retourne une liste d'utilisateurs disponibles pour test.
        
        Args:
            limit: Nombre maximum d'utilisateurs à retourner
            
        Returns:
            Liste d'IDs utilisateurs
        """
        if self.user_clicks is None or len(self.user_clicks) == 0:
            # Si pas de clicks, générer des IDs utilisateur fictifs pour la démo
            logger.warning("Pas de données utilisateurs, génération d'IDs fictifs")
            return list(range(1, min(limit + 1, 101)))
        
        return self.user_clicks['user_id'].unique()[:limit].tolist()


# Fonction utilitaire pour tester le système
if __name__ == "__main__":
    # Déterminer le chemin correct selon d'où le script est lancé
    import os
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Vérifier où se trouve le dossier data
    if (current_dir / "data").exists():
        data_path = "data"
    elif (project_root / "data").exists():
        data_path = str(project_root / "data")
    else:
        print("❌ Erreur: Dossier 'data' introuvable")
        print(f"Cherché dans: {current_dir / 'data'} et {project_root / 'data'}")
        exit(1)
    
    print(f"📂 Utilisation du dossier data: {data_path}\n")
    
    # Test du système
    recommender = ContentRecommender(data_path=data_path)
    
    # Récupérer des utilisateurs de test
    users = recommender.get_available_users(limit=5)
    print(f"\n📊 Utilisateurs de test: {users}")
    
    # Tester les recommandations
    for user_id in users[:2]:
        print(f"\n👤 Recommandations pour l'utilisateur {user_id}:")
        recommendations = recommender.recommend_for_user(user_id, top_n=5)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. Article {rec['article_id']}: {rec['title'][:50]}... (score: {rec['recommendation_score']:.3f})")