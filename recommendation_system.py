# import sqlite3
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def get_db_connection(db_path="DesignGenie.db"):
#     return sqlite3.connect(db_path)

# def validate_userInput(conn, user_input):
#     try:
#         required_fields = ["device_type", "operating_system", "content_type"]
#         for field in required_fields:
#             if field not in user_input:
#                 return False
#         return True
#     except Exception as e:
#         logger.error(f"Validation error: {e}")
#         return False

# def get_recommendations(conn, user_input):
#     try:
#         # Get guidelines from database
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT category, section, guideline, details, keywords 
#             FROM Guidelines
#         """)
#         guidelines = cursor.fetchall()

#         # Create feature vectors
#         vectorizer = TfidfVectorizer(stop_words='english')
        
#         # Prepare guidelines text
#         guidelines_text = [
#             f"{g[0]} {g[1]} {g[2]} {g[3]} {g[4]}" for g in guidelines
#         ]
        
#         # Prepare user input text
#         user_text = " ".join(str(v) for v in user_input.values())
        
#         # Calculate similarity
#         text_features = vectorizer.fit_transform(guidelines_text + [user_text])
#         similarities = cosine_similarity(
#             text_features[-1:], text_features[:-1]
#         )[0]

#         # Create recommendations list
#         recommendations = []
#         for idx, score in enumerate(similarities):
#             if score > 0.1:  # Minimum relevance threshold
#                 guideline = guidelines[idx]
#                 recommendations.append({
#                     'category': guideline[0],
#                     'section': guideline[1],
#                     'guideline': guideline[2],
#                     'details': guideline[3],
#                     'similarity_score': float(score),
#                     'context_matches': get_context_matches(guideline, user_input)
#                 })

#         # Sort by similarity score
#         recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
#         return recommendations[:10]  # Return top 10 recommendations

#     except Exception as e:
#         logger.error(f"Error generating recommendations: {e}")
#         return []

# def get_context_matches(guideline, user_input):
#     """Determine context matches for recommendations."""
#     guideline_text = " ".join(str(g) for g in guideline).lower()
#     return {
#         'device': user_input.get('device_type', '').lower() in guideline_text,
#         'interaction': user_input.get('interaction_requirements', '').lower() in guideline_text,
#         'content': user_input.get('content_type', '').lower() in guideline_text,
#         'accessibility': 'accessibility' in guideline_text or user_input.get('accessibility_needs', '').lower() in guideline_text
#     }

# def store_userInput(conn, user_input):
#     try:
#         cursor = conn.cursor()
#         columns = ", ".join(user_input.keys())
#         placeholders = ", ".join("?" * len(user_input))
#         query = f"INSERT INTO UserInputs ({columns}) VALUES ({placeholders})"
#         cursor.execute(query, list(user_input.values()))
#         conn.commit()
#         return cursor.lastrowid
#     except Exception as e:
#         logger.error(f"Error storing user input: {e}")
#         raise

# def store_recommendations(conn, user_input_id, recommendations):
#     try:
#         cursor = conn.cursor()
#         for rec in recommendations:
#             cursor.execute("""
#                 INSERT INTO RecommendationsMapping 
#                 (userInput_id, guideline, category, similarity_score) 
#                 VALUES (?, ?, ?, ?)
#             """, (
#                 user_input_id,
#                 rec['guideline'],
#                 rec['category'],
#                 rec['similarity_score']
#             ))
#         conn.commit()
#     except Exception as e:
#         logger.error(f"Error storing recommendations: {e}")
#         raise

# import sqlite3
# import numpy as np
# import logging
# import xgboost as xgb

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Utility function to connect to the database
# def get_db_connection(db_path="DesignGenie.db"):
#     """
#     Establish a connection to the SQLite database.

#     Parameters:
#         db_path (str): Path to the database file.

#     Returns:
#         sqlite3.Connection: A database connection object.
#     """
#     try:
#         conn = sqlite3.connect(db_path)
#         conn.row_factory = sqlite3.Row  # Enable column access by name
#         return conn
#     except Exception as e:
#         logger.error(f"Error connecting to database: {e}")
#         raise

# # Function to validate user input
# def validate_userInput(conn, user_input):
#     """
#     Validate user input against the schema constraints in the database.

#     Parameters:
#         conn (sqlite3.Connection): Active database connection.
#         user_input (dict): User input dictionary.

#     Returns:
#         bool: True if valid, False otherwise.
#     """
#     try:
#         cursor = conn.cursor()
#         cursor.execute("PRAGMA table_info(UserInputs)")
#         schema = cursor.fetchall()

#         # Extract valid fields and constraints
#         valid_fields = {col[1]: col[5] for col in schema}  # {column_name: constraint}
#         for field, value in user_input.items():
#             if field not in valid_fields:
#                 logger.warning(f"Unexpected field: {field}")
#                 return False

#             if valid_fields[field]:  # Check for constraints
#                 allowed_values = set(
#                     [val.strip("'") for val in valid_fields[field].split(" ")]
#                 )
#                 if value not in allowed_values:
#                     logger.warning(f"Invalid value for {field}: {value}")
#                     return False

#         return True
#     except Exception as e:
#         logger.error(f"Error validating user input: {e}")
#         return False

# # Function to generate hybrid recommendations with XGBoost
# def get_hybrid_recommendations(user_input):
#     """
#     Generate hybrid recommendations using XGBoost.

#     Parameters:
#         user_input (dict): User input dictionary.

#     Returns:
#         list: A list of recommendation dictionaries.
#     """
#     try:
#         # Simulate feature extraction for recommendations
#         features = extract_features(user_input)

#         # Load a pre-trained XGBoost model
#         model = xgb.Booster()
#         model.load_model("xgboost_model.json")  # Ensure this file exists

#         # Convert features to DMatrix (required for XGBoost)
#         dmatrix = xgb.DMatrix(features)

#         # Predict scores using the XGBoost model
#         predictions = model.predict(dmatrix)

#         # Combine predictions with static recommendations for now
#         recommendations = [
#             {
#                 "guideline": "Use high-contrast colors for accessibility",
#                 "category": "Accessibility",
#                 "similarity_score": predictions[0],  # First prediction score
#                 "final_score": predictions[0] * 1.1,  # Example weighted final score
#             },
#             {
#                 "guideline": "Optimize images for faster loading",
#                 "category": "Performance",
#                 "similarity_score": predictions[1],  # Second prediction score
#                 "final_score": predictions[1] * 1.2,  # Example weighted final score
#             },
#         ]
#         return recommendations
#     except Exception as e:
#         logger.error(f"Error generating recommendations with XGBoost: {e}")
#         return []

# # Function to extract features from user input for XGBoost
# def extract_features(user_input):
#     """
#     Extract features from user input to be used in XGBoost.

#     Parameters:
#         user_input (dict): User input dictionary.

#     Returns:
#         numpy.ndarray: A feature array for prediction.
#     """
#     try:
#         # Example: Feature extraction logic (customize as per your use case)
#         feature_mapping = {
#             "device_type": {"mobile": 0, "desktop": 1, "tablet": 2},
#             "operating_system": {"windows": 0, "macos": 1, "linux": 2},
#             "content_type": {"text": 0, "image": 1, "video": 2},
#         }
#         features = [
#             feature_mapping["device_type"].get(user_input.get("device_type"), -1),
#             feature_mapping["operating_system"].get(user_input.get("operating_system"), -1),
#             feature_mapping["content_type"].get(user_input.get("content_type"), -1),
#         ]
#         return np.array([features])
#     except Exception as e:
#         logger.error(f"Error extracting features: {e}")
#         return np.array([])

# # Function to store user input in the database
# def store_userInput(conn, user_input):
#     """
#     Store user input into the UserInputs table.

#     Parameters:
#         conn (sqlite3.Connection): Active database connection.
#         user_input (dict): User input dictionary.

#     Returns:
#         int: The ID of the inserted user input.
#     """
#     try:
#         columns = ", ".join(user_input.keys())
#         placeholders = ", ".join(["?"] * len(user_input))
#         query = f"INSERT INTO UserInputs ({columns}) VALUES ({placeholders})"
#         cursor = conn.cursor()
#         cursor.execute(query, tuple(user_input.values()))
#         conn.commit()
#         return cursor.lastrowid
#     except Exception as e:
#         logger.error(f"Error storing user input: {e}")
#         raise

# # Function to store recommendations in the database
# def store_recommendations(conn, user_input_id, recommendations):
#     """
#     Store generated recommendations in the RecommendationsMapping table.

#     Parameters:
#         conn (sqlite3.Connection): Active database connection.
#         user_input_id (int): The ID of the user input.
#         recommendations (list): A list of recommendation dictionaries.

#     Returns:
#         None
#     """
#     try:
#         cursor = conn.cursor()
#         for rec in recommendations:
#             query = """
#             INSERT INTO RecommendationsMapping (userInput_id, guideline, category, similarity_score, final_score)
#             VALUES (?, ?, ?, ?, ?)
#             """
#             cursor.execute(
#                 query,
#                 (
#                     user_input_id,
#                     rec["guideline"],
#                     rec["category"],
#                     rec["similarity_score"],
#                     rec["final_score"],
#                 ),
#             )
#         conn.commit()
#     except Exception as e:
#         logger.error(f"Error storing recommendations: {e}")
#         raise


import sqlite3
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from flask import Flask

# Flask app setupF
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection context manager
def connect_db(db_path="DesignGenie.db"):
    return sqlite3.connect(db_path)

# Validate user input against schema
def validate_user_input(conn, user_input):
    """Validate user input against the database schema."""
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(UserInputs)")
        columns = cursor.fetchall()
        logger.info("Validating user input against database schema.")

        for column in columns:
            column_name = column[1]
            if column_name in user_input:
                check_constraint = column[5] if len(column) > 5 else None
                if check_constraint and "CHECK" in check_constraint:
                    allowed_values = set(re.findall(r"'([^']*)'", check_constraint))
                    if user_input[column_name] not in allowed_values:
                        return {
                            "status": "error",
                            "field": column_name,
                            "message": f"Invalid value for {column_name}. Allowed values: {allowed_values}",
                        }
        return {"status": "success", "message": "Validation successful"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Store validated user input
def store_user_input(conn, user_input):
    try:
        validate_result = validate_user_input(conn, user_input)
        if validate_result["status"] == "error":
            raise ValueError(validate_result["message"])

        cursor = conn.cursor()
        columns = ", ".join(user_input.keys())
        placeholders = ", ".join(["?"] * len(user_input))
        query = f"INSERT INTO UserInputs ({columns}) VALUES ({placeholders})"
        cursor.execute(query, tuple(user_input.values()))
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        raise ValueError(f"Error storing user input: {e}")
    
def get_content_based_recommendations(user_input, conn, min_similarity=0.01, limit=30):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT category, section, guideline, details, keywords
            FROM Guidelines
            WHERE category IS NOT NULL
        """)
        guidelines = cursor.fetchall()

        # Create search text from user input
        search_text = " ".join(str(v).lower() for v in user_input.values())
        
        # Create TF-IDF vectorizer with broader features
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=2000,
            ngram_range=(1, 2)
        )

        # Prepare guideline texts
        guideline_texts = [
            f"{g[0]} {g[1]} {g[2]} {g[3]} {g[4]}".lower() 
            for g in guidelines
        ]

        # Calculate similarities
        tfidf_matrix = vectorizer.fit_transform(guideline_texts + [search_text])
        similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]

        # Create recommendations
        recommendations = []
        for idx, score in enumerate(similarities):
            if score > min_similarity:
                guideline = guidelines[idx]
                recommendations.append({
                    'category': guideline[0],
                    'section': guideline[1],
                    'guideline': guideline[2],
                    'details': guideline[3],
                    'similarity_score': float(score)
                })

        return sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True)[:limit]

    except Exception as e:
        logger.error(f"Error in content-based recommendations: {e}")
        return []

def field_matches_guideline(field, value, guideline):
    """
    Check if field value matches the guideline's characteristics.
    """
    value = str(value).lower()
    guideline_text = (
        f"{guideline['category']} {guideline['section']} "
        f"{guideline['guideline']} {guideline['details']} "
        f"{guideline.get('keywords', '')}".lower()
    )

    if value in guideline_text:
        return True

    return False

def get_similar_users(user_input, conn):
    """
    Dynamically find similar users based on fields present in user_input.
    """
    try:
        # Define fields for matching
        matching_fields = ["age_group", "project_goals", "content_type", "interaction_requirements"]

        # Dynamically construct WHERE clause
        conditions = []
        params = []
        for field in matching_fields:
            if field in user_input and user_input[field]:
                conditions.append(f"{field} = ?")
                params.append(user_input[field])

        # If no valid conditions are found, skip the query
        if not conditions:
            print("No valid matching fields found in user_input.")
            return pd.DataFrame()

        # Build and execute query
        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT * 
        FROM UserInputs
        WHERE {where_clause}
        LIMIT 10
        """
        return pd.read_sql_query(query, conn, params)

    except Exception as e:
        print(f"Error finding similar users: {e}")
        return pd.DataFrame()
    
def get_collaborative_recommendations(user_input, conn):
    """
    Generate recommendations based on similar users' data.
    """
    try:
        # Find similar users
        similar_users = get_similar_users(user_input, conn)
        if similar_users.empty:
            print("No similar users found.")
            return []

        similar_user_ids = similar_users["userInput_id"].tolist()

        # Query recommendations for similar users
        placeholders = ",".join("?" for _ in similar_user_ids)
        query = f"""
        SELECT g.*, COUNT(*) AS frequency
        FROM RecommendationsMapping rm
        JOIN Guidelines g ON rm.guideline_id = g.id
        WHERE rm.userInput_id IN ({placeholders})
        GROUP BY g.id
        ORDER BY frequency DESC
        LIMIT 10
        """
        recommendations = pd.read_sql_query(query, conn, similar_user_ids)

        # Format recommendations
        return recommendations.to_dict("records")
    except Exception as e:
        print(f"Error generating collaborative recommendations: {e}")
        return []
    
def combine_recommendations(content_based_recs, collaborative_recs, weights=None):
    """
    Combine content-based and collaborative recommendations.
    """
    weights = weights or {"content": 0.8, "collaborative": 0.2}
    combined_recs = {}

    # Add content-based recommendations
    for rec in content_based_recs:
        rec_id = rec["guideline"]
        combined_recs[rec_id] = rec
        combined_recs[rec_id]["final_score"] = rec["similarity_score"] * weights["content"]

    # Add collaborative recommendations
    for rec in collaborative_recs:
        rec_id = rec["guideline"]
        if rec_id in combined_recs:
            combined_recs[rec_id]["final_score"] += rec["frequency"] * weights["collaborative"]
        else:
            rec["final_score"] = rec["frequency"] * weights["collaborative"]
            combined_recs[rec_id] = rec

    # Sort recommendations by final score
    return sorted(combined_recs.values(), key=lambda x: x["final_score"], reverse=True)
    
def apply_personalization(recommendations, user_input):
    """
    Adjust recommendation scores based on personalization criteria.
    """
    try:
        personalized_recs = []

        for rec in recommendations:
            # Initialize personalization multiplier
            personalization_multiplier = 1.0

            # Apply personalization criteria
            if "device_type" in user_input and user_input["device_type"].lower() in rec.get("keywords", "").lower():
                personalization_multiplier *= 1.2
            if "accessibility_needs" in user_input and user_input["accessibility_needs"].lower() in rec.get("keywords", "").lower():
                personalization_multiplier *= 1.15
            if "age_group" in user_input and user_input["age_group"].lower() in rec.get("keywords", "").lower():
                personalization_multiplier *= 1.1
            if "content_type" in user_input and user_input["content_type"].lower() in rec.get("keywords", "").lower():
                personalization_multiplier *= 1.15

            # Adjust final score
            rec["final_score"] *= personalization_multiplier
            personalized_recs.append(rec)

        # Normalize scores
        max_score = max((rec["final_score"] for rec in personalized_recs), default=1)
        for rec in personalized_recs:
            rec["final_score"] /= max_score  # Scale to [0, 1]

        # Sort recommendations by adjusted scores
        return sorted(personalized_recs, key=lambda x: x["final_score"], reverse=True)

    except Exception as e:
        print(f"Error applying personalization: {e}")
        return recommendations
    
def get_hybrid_recommendations(user_input, conn, weights=None, limit=20):
    """
    Generate hybrid recommendations with improved logic.
    """
    try:
        # Default weights
        weights = weights or {
            "similarity": 0.6,
            "collaborative": 0.3,
            "personalization": 0.1,
        }

        # Get content-based recommendations with lower threshold
        content_based_recs = get_content_based_recommendations(
            user_input, 
            conn,
            min_similarity=0.01,  # Lower threshold to get more initial matches
            limit=30  # Get more recommendations initially
        )

        # Get collaborative recommendations
        collaborative_recs = get_collaborative_recommendations(user_input, conn)

        # Combine all recommendations
        combined_recs = {}
        
        # Add content-based recommendations
        for rec in content_based_recs:
            rec_id = rec["guideline"]
            combined_recs[rec_id] = rec
            combined_recs[rec_id]["final_score"] = rec["similarity_score"] * weights["similarity"]
            # Add category context score
            if rec["category"].lower() in str(user_input).lower():
                combined_recs[rec_id]["final_score"] *= 1.2

        # Add collaborative recommendations
        for rec in collaborative_recs:
            rec_id = rec["guideline"]
            if rec_id in combined_recs:
                combined_recs[rec_id]["final_score"] += rec.get("frequency", 0) * weights["collaborative"]
            else:
                rec["final_score"] = rec.get("frequency", 0) * weights["collaborative"]
                combined_recs[rec_id] = rec

        # Convert to list and sort
        recommendations = list(combined_recs.values())
        
        # Apply personalization based on user input
        personalized_recs = apply_personalization(recommendations, user_input)

        # Sort by final score and return top results
        final_recommendations = sorted(
            personalized_recs,
            key=lambda x: x["final_score"],
            reverse=True
        )[:limit]

        # Log recommendation distribution
        categories = set(rec["category"] for rec in final_recommendations)
        logger.info(f"Recommendation categories: {categories}")
        
        return final_recommendations

    except Exception as e:
        logger.error(f"Error in hybrid recommendations: {e}")
        return []
    
def analyze_recommendations(recommendations):
    """
    Analyze the quality and trends of recommendations.
    
    Args:
        recommendations (list): List of recommendation dictionaries.

    Returns:
        dict: Metrics analysis of recommendations.
    """
    if not recommendations:
        print("No recommendations generated for analysis.")
        return {
            "total_recommendations": 0,
            "average_similarity": 0.0,
            "categories": set(),
            "sections": set(),
        }

    try:
        # Calculate metrics
        total_recommendations = len(recommendations)
        average_similarity = np.mean([rec.get("similarity_score", 0.0) for rec in recommendations])
        categories = set(rec.get("category", "Unknown") for rec in recommendations)
        sections = set(rec.get("section", "Unknown") for rec in recommendations)

        logger.info(f"Total Recommendations: {total_recommendations}")
        logger.info(f"Average Similarity Score: {average_similarity:.4f}")
        logger.info(f"Categories Covered: {categories}")
        logger.info(f"Sections Covered: {sections}")

        # Identify top guidelines
        top_guidelines = sorted(recommendations, key=lambda x: x.get("similarity_score", 0.0), reverse=True)[:10]

        # Prepare analysis
        analysis = {
            "total_recommendations": total_recommendations,
            "average_similarity": average_similarity,
            "categories": categories,
            "sections": sections,
            "top_guidelines": top_guidelines,
        }

        # Print analysis
        print("\nRecommendation Analysis:")
        print(f"Total Recommendations: {total_recommendations}")
        print(f"Average Similarity Score: {average_similarity:.2f}")
        print(f"Categories Covered: {', '.join(categories) if categories else 'None'}")
        print(f"Sections Covered: {', '.join(sections) if sections else 'None'}")
        print("Top Guidelines:")
        for i, guideline in enumerate(top_guidelines, 1):
            print(f"{i}. {guideline['guideline']} (Score: {guideline['similarity_score']:.2f})")

        return analysis

    except Exception as e:
        print(f"Error analyzing recommendations: {e}")
        return {
            "total_recommendations": 0,
            "average_similarity": 0.0,
            "categories": set(),
            "sections": set(),
        }
    
def calculate_recommendation_relevance(guideline, user_input, weights):
    """
    Calculate a weighted relevance score for a guideline based on user input.
    
    Args:
        guideline (dict): A recommendation guideline with text fields.
        user_input (dict): User input fields and their values.
        weights (dict): Weights assigned to input fields.
    
    Returns:
        float: Relevance score between 0 and 1.
    """
    relevance_score = 0
    max_possible_score = sum(weights.values())

    for field, weight in weights.items():
        if field in user_input and user_input[field]:
            if field_matches_guideline(field, user_input[field], guideline):
                relevance_score += weight
            else:
                relevance_score -= weight * 0.1  # Penalize slightly for non-matching fields.

    # Normalize score between 0 and 1
    return max(0, min(relevance_score / max_possible_score, 1))

def visualize_metrics(recommendations):
    """
    Create visualizations for recommendation metrics.
    """
    try:
        if not recommendations:
            print("No recommendations available for visualization.")
            return

        categories = [rec.get("category", "Unknown") for rec in recommendations]
        sections = [rec.get("section", "Unknown") for rec in recommendations]
        similarity_scores = [rec.get("similarity_score", 0) for rec in recommendations]

        # Categories distribution
        plt.figure(figsize=(10, 5))
        pd.Series(categories).value_counts().plot(kind="bar")
        plt.title("Category Distribution")
        plt.xlabel("Categories")
        plt.ylabel("Count")
        plt.show()

        # Sections distribution
        plt.figure(figsize=(10, 5))
        pd.Series(sections).value_counts().plot(kind="bar", color="orange")
        plt.title("Section Distribution")
        plt.xlabel("Sections")
        plt.ylabel("Count")
        plt.show()

        # Similarity score distribution
        plt.figure(figsize=(8, 5))
        plt.hist(similarity_scores, bins=10, alpha=0.7, color="green")
        plt.title("Similarity Score Distribution")
        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        plt.show()

    except Exception as e:
        print(f"Error visualizing metrics: {e}")



def prepare_training_data(user_inputs, recommendations):
    """
    Prepare training data for XGBoost based on user inputs and recommendations.
    """
    try:
        # Encode categorical fields
        encoders = {}
        features = []
        target = []

        for user_input, rec_list in zip(user_inputs, recommendations):
            for rec in rec_list:
                # Collect features
                feature = {
                    "similarity_score": rec["similarity_score"],
                    "category": rec["category"],
                    "section": rec["section"],
                }
                feature.update(user_input)
                features.append(feature)

                # Collect target
                target.append(rec["final_score"])

        # Encode categorical fields
        encoded_features = pd.DataFrame(features)
        for col in encoded_features.select_dtypes(include="object").columns:
            encoders[col] = LabelEncoder()
            encoded_features[col] = encoders[col].fit_transform(encoded_features[col])

        return encoded_features, target, encoders

    except Exception as e:
        print(f"Error preparing training data: {e}")
        return None, None, None


def train_xgboost_model(features, target):
    """
    Train an XGBoost model to predict final scores with compatible parameters.
    """
    try:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Define model parameters
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42
        )
        
        # Fit model with evaluation set and early stopping
        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,  # Stop if no improvement after 10 rounds
            verbose=True
        )

        # Evaluate performance
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"Model RMSE: {rmse:.4f}")

        return model

    except Exception as e:
        print(f"Error training XGBoost model: {e}")
        return None
    
def enhance_with_xgboost(user_input, base_recommendations, model, encoders):
    """
    Enhance recommendations using an XGBoost model.
    """
    try:
        # Encode user input
        encoded_input = {}
        for field, encoder in encoders.items():
            if field in user_input:
                encoded_input[field] = encoder.transform([user_input[field]])[0]

        # Prepare features for prediction
        features = []
        for rec in base_recommendations:
            feature = encoded_input.copy()
            feature["similarity_score"] = rec["similarity_score"]
            feature["category"] = encoders["category"].transform([rec["category"]])[0]
            feature["section"] = encoders["section"].transform([rec["section"]])[0]
            features.append(feature)

        # Predict scores
        predicted_scores = model.predict(pd.DataFrame(features))
        for i, rec in enumerate(base_recommendations):
            rec["xgboost_score"] = predicted_scores[i]
            rec["final_score"] = 0.7 * rec["similarity_score"] + 0.3 * rec["xgboost_score"]

        return sorted(base_recommendations, key=lambda x: x["final_score"], reverse=True)

    except Exception as e:
        print(f"Error enhancing recommendations with XGBoost: {e}")
        return base_recommendations
    

def visualize_metrics(recommendations):
    """
    Visualize metrics for the recommendations.
    """
    if not recommendations:
        print("No recommendations available for visualization.")
        return

    # Convert recommendations to DataFrame for analysis
    rec_df = pd.DataFrame(recommendations)

    # Plot 1: Distribution of Recommendations by Category
    plt.figure(figsize=(10, 6))
    sns.countplot(y="category", data=rec_df, order=rec_df["category"].value_counts().index)
    plt.title("Distribution of Recommendations by Category")
    plt.xlabel("Count")
    plt.ylabel("Category")
    plt.show()

    # Plot 2: Distribution of Relevance Scores
    plt.figure(figsize=(10, 6))
    sns.histplot(rec_df["similarity_score"], bins=10, kde=True, color='blue', label="Similarity Scores")
    sns.histplot(rec_df["final_score"], bins=10, kde=True, color='green', label="Final Scores")
    plt.title("Relevance Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Plot 3: Coverage of Sections
    plt.figure(figsize=(10, 6))
    section_counts = rec_df["section"].value_counts()
    plt.pie(section_counts, labels=section_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    plt.title("Coverage of Recommendations by Section")
    plt.show()

# Example Recommendations Data (replace with actual system output)
recommendations = [
    {"category": "Interaction Patterns", "section": "Content Discovery", "guideline": "Provide tooltips.", "similarity_score": 0.85, "final_score": 0.92},
    {"category": "Interaction Patterns", "section": "Form Interactions", "guideline": "Use autocomplete.", "similarity_score": 0.78, "final_score": 0.85},
    {"category": "Design Strategy", "section": "UX Optimization", "guideline": "The Pareto Principle.", "similarity_score": 0.65, "final_score": 0.75},
    {"category": "Accessibility", "section": "Best Practices", "guideline": "Provide high contrast.", "similarity_score": 0.7, "final_score": 0.82},
    {"category": "Content Design", "section": "Data Presentation", "guideline": "Use charts.", "similarity_score": 0.6, "final_score": 0.68}
]

# Visualize metrics
visualize_metrics(recommendations)