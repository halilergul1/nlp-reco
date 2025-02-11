import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import src.config as config

def clean_and_convert_embedding(embedding_str):
    """
    This is needed because I realized after saving the data into csv that embeddings are saved as strings and distorted.

    Converts string representation of topic embedding into a NumPy array.

    Args:
        embedding_str (str): String representation of an embedding.

    Returns:
        np.array or NaN: Converted NumPy array or NaN if invalid.
    """
    if isinstance(embedding_str, str):
        try:
            cleaned_str = re.sub(r"[\[\]]", "", embedding_str).strip()
            embedding_values = cleaned_str.split()
            return np.array([float(x) for x in embedding_values])
        except ValueError:
            return np.nan  #  NaN if conversion fails
    return np.nan  #  NaN if not a string

def compute_cosine_similarity(final_df):
    """
    Computes cosine similarity between topic embeddings of hotels.

    Args:
        final_df (DataFrame): Processed dataset containing topic embeddings.

    Returns:
        cosine_sim_matrix (np.array): Cosine similarity matrix.
        valid_indices (Series): Boolean mask for rows with valid embeddings.
    """
    #  valid embeddings which are not NaN
    valid_indices = final_df['topic_embedding'].apply(lambda x: isinstance(x, np.ndarray))
    valid_embeddings = np.stack(final_df.loc[valid_indices, 'topic_embedding'].values)

    # cosine similarity only for valid embeddings
    cosine_sim_matrix = cosine_similarity(valid_embeddings)

    return cosine_sim_matrix, valid_indices

def generate_recommendations(final_df, cosine_sim_matrix, valid_indices):
    """
    Generates hotel recommendations based on topic similarity.

    Args:
        final_df (DataFrame): Processed dataset containing hotel metadata.
        cosine_sim_matrix (np.array): Cosine similarity matrix.
        valid_indices (Series): Boolean mask for rows with valid embeddings.

    Returns:
        recommendations (dict): Dictionary mapping hotel_id to recommended hotels.
    """
    recommendations = {}

    valid_hotel_ids = final_df.loc[valid_indices, 'hotel_id'].values

    for idx, hotel_id in tqdm(enumerate(valid_hotel_ids), total=len(valid_hotel_ids)):
        scores = cosine_sim_matrix[idx]
        
        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order

        # Removing the current hotel itself (where hotel_id matches)
        sorted_indices = [i for i in sorted_indices if valid_hotel_ids[i] != hotel_id]

        #  top 10 recommendations
        top_indices = sorted_indices[:10]

        # Store recommendations
        recommendations[hotel_id] = final_df.loc[valid_indices].iloc[top_indices]['hotel_id'].values


    return recommendations

def handle_missing_embeddings(final_df, recommendations, valid_indices):
    """
    Fills in recommendations for hotels without valid embeddings using popularity score.

    Args:
        final_df (DataFrame): Processed dataset containing hotel metadata.
        recommendations (dict): Current recommendation dictionary.
        valid_indices (Series): Boolean mask for rows with valid embeddings.

    Returns:
        recommendations (dict): Updated recommendation dictionary.
    """
    invalid_indices = ~valid_indices  # indices where embeddings are NaN
    popular_hotels = final_df.sort_values(by='popularity_score', ascending=False)['hotel_id'].values

    for hotel_id in final_df.loc[invalid_indices, 'hotel_id'].values:
        filtered_hotels = [h for h in popular_hotels if h != hotel_id]  # Remove self-recommendation
        recommendations[hotel_id] = filtered_hotels[:10]  # Select top 10

    return recommendations

def rank_recommendations(final_df, recommendations):
    """
    Ranks recommendations for each hotel based on combined popularity score.

    Args:
        final_df (DataFrame): Processed dataset containing hotel metadata.
        recommendations (dict): Dictionary mapping hotel_id to recommended hotels.

    Returns:
        recommendation_df (DataFrame): Final ranked recommendation dataframe.
    """
    final_recommendations = []

    for hotel_id, reco_hotels in recommendations.items():
        reco_final = final_df[(final_df['hotel_id'].isin(reco_hotels)) & (final_df['hotel_id'] != hotel_id)][['hotel_id', 'popularity_rank']]
        reco_final = reco_final.sort_values(by='popularity_rank', ascending=False).head(10)

        for rank, reco_hotel_id in enumerate(reco_final['hotel_id'], 1):
            final_recommendations.append([hotel_id, reco_hotel_id, rank])


    recommendation_df = pd.DataFrame(final_recommendations, columns=['hotel_id', 'reco_hotel_id', 'rank'])

    return recommendation_df

def save_recommendations(recommendation_df):
    """
    Saves the final hotel recommendations to a CSV file.

    Args:
        recommendation_df (DataFrame): Final ranked recommendation dataframe.
    """
    recommendation_df.to_csv(f"{config.DATA_PATH}/hotel_recommendations.csv", index=False)
    print("Recommendation file 'hotel_recommendations.csv' saved successfully!")