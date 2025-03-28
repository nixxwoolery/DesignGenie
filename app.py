from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    return sqlite3.connect('DesignGenie.db')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        user_input = request.json
        logger.info(f"Received user input: {user_input}")
        
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get all unique guidelines
        cursor.execute("""
            SELECT DISTINCT category, section, subsection, guideline, details, keywords
            FROM Guidelines
            WHERE guideline IS NOT NULL
            ORDER BY category, section, subsection
        """)
        all_guidelines = cursor.fetchall()
        
        # Create search text using all user input fields
        search_terms = []
        for key, value in user_input.items():
            if value:
                if isinstance(value, list):
                    search_terms.extend(value)
                else:
                    search_terms.append(str(value))
        search_text = " ".join(search_terms)

        # Calculate similarities
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=2000,
            ngram_range=(1, 2)  # Consider both unigrams and bigrams
        )
        
        # Create guideline texts with all fields
        guidelines_text = [
            f"{g[0]} {g[1]} {g[2]} {g[3]} {g[4]}".lower() 
            for g in all_guidelines
        ]
        
        # Calculate similarity scores
        try:
            tfidf_matrix = vectorizer.fit_transform(guidelines_text + [search_text.lower()])
            similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
            
            # Create unique recommendations
            seen_guidelines = set()
            recommendations = []
            
            for idx, score in enumerate(similarities):
                guideline = all_guidelines[idx]
                guideline_text = guideline[2]  # Use guideline text as unique identifier
                
                if guideline_text not in seen_guidelines and score > 0.01:
                    # Calculate relevance boost
                    relevance_boost = 0
                    guideline_str = " ".join(str(g) for g in guideline).lower()
                    
                    for key, value in user_input.items():
                        if str(value).lower() in guideline_str:
                            relevance_boost += 0.1
                    
                    final_score = score + relevance_boost
                    
                    recommendations.append({
                        'category': guideline[0],
                        'section': guideline[1],
                        'subsection': guideline[2],  # Make sure subsection is included
                        'guideline': guideline[3],
                        'details': guideline[4],
                        'similarity_score': float(score)
                    })

                    seen_guidelines.add(guideline_text)

            # Sort and get top recommendations
            recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            top_recommendations = recommendations[:20]  # Get top 20 unique recommendations

            logger.info(f"Generated {len(top_recommendations)} unique recommendations")

            return jsonify({
                "status": "success",
                "recommendations": top_recommendations,
                "debug_info": {
                    "total_guidelines": len(all_guidelines),
                    "unique_recommendations": len(top_recommendations),
                    "search_terms_used": search_terms
                }
            })

        except Exception as e:
            logger.error(f"Error in similarity calculation: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error in submit: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5501, debug=True)