import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import src.config as config

def load_data():
    """
    Loads hotel datasets from the data directory.
    Returns:
        hotel_info (DataFrame): Hotel metadata
        hotel_desc (DataFrame): Hotel descriptions
        session_data (DataFrame): User session interactions
    """
    hotel_info = pd.read_csv(f"{config.DATA_PATH}/hotel_info.csv")
    hotel_desc = pd.read_csv(f"{config.DATA_PATH}/hotel_desc.csv")
    session_data = pd.read_csv(f"{config.DATA_PATH}/session_data.csv")
    
    return hotel_info, hotel_desc, session_data

def clean_hotel_info(hotel_info):
    """
    Cleans and processes the hotel_info dataset by handling missing values.
    
    Args:
        hotel_info (DataFrame): Raw hotel info data
    
    Returns:
        hotel_info (DataFrame): Cleaned hotel info data
    """

    # this is for missing categorical values
    hotel_info["hotel_subtown"] = hotel_info["hotel_subtown"].fillna("Unknown")
    hotel_info["facility_type"] = hotel_info["facility_type"].fillna(hotel_info["facility_type"].mode()[0])

    # filling numerical values with median (I checked how many in eda part in a notebook)
    rating_columns = list(config.STAR_RATING_WEIGHTS.keys())
    for col in rating_columns:
        hotel_info[col] = hotel_info[col].fillna(hotel_info[col].median())

    return hotel_info

def clean_session_data(session_data):
    """
    Cleans the session_data dataset by dropping critical missing values.

    Args:
        session_data (DataFrame): Raw session data

    Returns:
        session_data (DataFrame): Cleaned session data
    """

    session_data.dropna(subset=['funnel_id', 'adult_count', 'child_count', 'check_in_date', 'check_out_date'], inplace=True)
    session_data["check_in_date"] = pd.to_datetime(session_data["check_in_date"])
    session_data["check_out_date"] = pd.to_datetime(session_data["check_out_date"])
    
    return session_data

def compute_weighted_star_rating(hotel_info):
    """
    Computes a normalized star rating for each hotel using weighted ratings.
    
    Args:
        hotel_info (DataFrame): Cleaned hotel metadata
    
    Returns:
        hotel_info (DataFrame): Updated hotel metadata with normalized_star_rating
    """

    rating_columns = list(config.STAR_RATING_WEIGHTS.keys())
    weights = np.array(list(config.STAR_RATING_WEIGHTS.values()))
    
    #  weighted sum of ratings
    hotel_info["weighted_star_rating"] = np.dot(hotel_info[rating_columns], weights)

    # outliers at 99th percentile, cap
    cap_value = np.percentile(hotel_info["weighted_star_rating"], 99)
    hotel_info["capped_star_rating"] = np.minimum(hotel_info["weighted_star_rating"], cap_value)

    # 1-5 scale
    scaler = MinMaxScaler(feature_range=(1, 5))
    hotel_info["normalized_star_rating"] = scaler.fit_transform(hotel_info[["capped_star_rating"]])

    return hotel_info

def compute_popularity_score(hotel_info):
    """
    Computes popularity score based on comment count and image count.

    Args:
        hotel_info (DataFrame): Hotel metadata with rating information.

    Returns:
        hotel_info (DataFrame): Updated hotel data with a popularity_score column.
    """
    
    comment_weight = config.COMMENT_WEIGHT
    image_weight = config.IMAGE_WEIGHT

    #  weighted popularity score
    hotel_info["raw_popularity"] = (
        comment_weight * hotel_info["hotel_comment_count"] + 
        image_weight * hotel_info["hotel_image_count"]
    )

    # Handling outliers by capping at 99th percentile
    cap_value = np.percentile(hotel_info["raw_popularity"], 99)
    hotel_info["capped_popularity"] = np.minimum(hotel_info["raw_popularity"], cap_value)

    # between 0 and 1
    scaler = MinMaxScaler()
    hotel_info["popularity_score"] = scaler.fit_transform(hotel_info[["capped_popularity"]])

    return hotel_info

def merge_processed_data(hotel_desc, hotel_info):
    """
    Merges hotel descriptions with computed star rating and popularity.

    Args:
        hotel_desc (DataFrame): Processed hotel descriptions with topic modeling
        hotel_info (DataFrame): Processed hotel metadata with computed ratings

    Returns:
        merged_df (DataFrame): Final processed dataset for recommendations
    """

    merged_df = hotel_desc.merge(
        hotel_info[["hotel_id", "normalized_star_rating", "popularity_score"]],
        on="hotel_id", how="left"
    )

    # necessary columns
    merged_df = merged_df[["hotel_id", "topic", "representation", "topic_embedding", "normalized_star_rating", "popularity_score"]]

    return merged_df

def preprocess_data():
    """
    Orchestrates the full data preprocessing pipeline.

    Returns:
        hotel_info (DataFrame): Cleaned hotel metadata with star ratings and popularity scores.
        hotel_desc (DataFrame): Raw hotel descriptions.
        session_data (DataFrame): Cleaned session data.
    """
    hotel_info, hotel_desc, session_data = load_data()
    hotel_info = clean_hotel_info(hotel_info)
    session_data = clean_session_data(session_data)
    hotel_info = compute_weighted_star_rating(hotel_info)
    hotel_info = compute_popularity_score(hotel_info)
    return hotel_info, hotel_desc, session_data


def save_final_dataset(final_df):
    """
    Saves the final processed dataset to CSV.

    Args:
        final_df (DataFrame): The fully processed dataset
    """
    final_df.to_csv(f"{config.DATA_PATH}/final_dataset.csv", index=False)
    print("Final dataset saved as 'final_dataset.csv'.")

