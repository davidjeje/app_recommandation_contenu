"""
Application Streamlit pour My Content MVP
Interface utilisateur pour tester le système de recommandation
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour importer le module
sys.path.append(str(Path(__file__).parent.parent))

from my_content.recommendation_engine import ContentRecommender


# Configuration de la page
st.set_page_config(
    page_title="My Content - MVP",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def load_recommender():
    """Charge le système de recommandation (mis en cache pour éviter de recharger)"""
    return ContentRecommender(data_path="data")


def main():
    # En-tête de l'application
    st.title("📚 My Content - Système de Recommandation MVP")
    st.markdown("---")
    
    # Chargement du système de recommandation
    with st.spinner("🔄 Chargement du système de recommandation..."):
        try:
            recommender = load_recommender()
            st.success("✅ Système de recommandation chargé avec succès!")
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement: {e}")
            st.info("Assurez-vous que le dossier 'data/' contient les fichiers nécessaires.")
            return
    
    # Sidebar avec informations
    with st.sidebar:
        st.header("ℹ️ Informations")
        st.markdown("""
        **My Content** est une start-up qui encourage la lecture 
        en recommandant des contenus pertinents.
        
        ### Comment ça marche ?
        1. Sélectionnez un utilisateur
        2. Cliquez sur "Recommander"
        3. Découvrez les 5 articles recommandés
        
        ### Algorithme
        - **Type**: Content-Based Filtering
        - **Méthode**: Similarité cosinus sur embeddings
        - **Dimension**: 250 features par article
        """)
        
        st.markdown("---")
        st.markdown("**Développé par:** Vous (CTO) & Samia (CEO)")
    
    # Section principale - Sélection de l'utilisateur
    st.header("👤 Sélection de l'utilisateur")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Récupérer la liste des utilisateurs disponibles
        available_users = recommender.get_available_users(limit=100)
        
        if not available_users:
            st.warning("Aucun utilisateur disponible dans les données.")
            return
        
        # Sélection de l'utilisateur
        selected_user = st.selectbox(
            "Choisissez un ID utilisateur:",
            options=available_users,
            index=0
        )
    
    with col2:
        st.metric("Utilisateurs disponibles", len(available_users))
    
    # Bouton de recommandation
    st.markdown("---")
    
    if st.button("🎯 Générer les recommandations", type="primary", use_container_width=True):
        with st.spinner(f"🔍 Analyse de l'historique de l'utilisateur {selected_user}..."):
            # Récupérer l'historique utilisateur
            user_history = recommender.get_user_history(selected_user)
            
            # Afficher l'historique
            st.subheader(f"📖 Historique de lecture (Utilisateur {selected_user})")
            
            if user_history:
                st.info(f"L'utilisateur a lu **{len(user_history)} articles** différents.")
                
                # Afficher quelques articles lus
                with st.expander("Voir les 5 derniers articles lus"):
                    for i, article_id in enumerate(user_history[:5], 1):
                        article_info = recommender._get_article_info(article_id)
                        st.markdown(f"**{i}.** Article {article_id}: *{article_info['title']}*")
            else:
                st.warning("Aucun historique trouvé. Recommandations basées sur la popularité.")
            
            st.markdown("---")
            
            # Générer les recommandations
            recommendations = recommender.recommend_for_user(selected_user, top_n=5)
            
            # Afficher les recommandations
            st.subheader("⭐ Top 5 des articles recommandés")
            
            if recommendations:
                # Afficher sous forme de cartes
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        col_rank, col_content, col_score = st.columns([0.5, 3, 1])
                        
                        with col_rank:
                            st.markdown(f"### #{i}")
                        
                        with col_content:
                            st.markdown(f"**Article ID:** {rec['article_id']}")
                            st.markdown(f"**Titre:** {rec['title']}")
                            st.markdown(f"**Catégorie:** {rec['category']} | **Mots:** {rec['words_count']}")
                        
                        with col_score:
                            st.metric("Score", f"{rec['recommendation_score']:.2f}")
                        
                        st.markdown("---")
                
                # Bouton de téléchargement des résultats
                import json
                import numpy as np
                
                # Convertir les types numpy en types Python natifs pour JSON
                def convert_to_json_serializable(obj):
                    if isinstance(obj, (np.integer, np.int64)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                # Nettoyer les recommandations pour JSON
                json_recommendations = []
                for rec in recommendations:
                    json_rec = {k: convert_to_json_serializable(v) for k, v in rec.items()}
                    json_recommendations.append(json_rec)
                
                results_json = json.dumps(json_recommendations, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Télécharger les résultats (JSON)",
                    data=results_json,
                    file_name=f"recommendations_user_{selected_user}.json",
                    mime="application/json"
                )
            else:
                st.error("Aucune recommandation générée. Vérifiez les données.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>My Content MVP - Version 1.0 | Architecture Serverless Azure Functions</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()