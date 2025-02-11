import pandas as pd
import src.config as config

def merge_hotel_metadata(recommendation_df, hotel_info):
    """
    Merges recommendations with hotel metadata to get hotel names and city locations.

    Args:
        recommendation_df (DataFrame): Ranked hotel recommendations.
        hotel_info (DataFrame): Hotel metadata containing names and city locations.

    Returns:
        merged_df (DataFrame): Recommendations enriched with hotel details.
    """
    #  hotel_name and hotel_city for original hotel_id
    merged_df = recommendation_df.merge(
        hotel_info[['hotel_id', 'hotel_name', 'hotel_city']], 
        left_on='hotel_id', 
        right_on='hotel_id', 
        how='left'
    )

    #  columns after first merge
    merged_df = merged_df.rename(columns={
        'hotel_name': 'hotel_id_hotel_name',
        'hotel_city': 'hotel_city_hotel_name'
    })

    # hotel_name and hotel_city for recommended hotel_id
    merged_df = merged_df.merge(
        hotel_info[['hotel_id', 'hotel_name', 'hotel_city']], 
        left_on='reco_hotel_id', 
        right_on='hotel_id', 
        how='left'
    )

    # here I rename columns after second merge
    merged_df = merged_df.rename(columns={
        'hotel_name': 'reco_hotel_id_hotel_name',
        'hotel_city': 'reco_hotel_city_hotel_name'
    })

    # drop unnecessary 
    merged_df = merged_df.drop(columns=['hotel_id_y'])

    # renaming
    merged_df = merged_df.rename(columns={'hotel_id_x': 'hotel_id'})

    return merged_df

def compute_city_divergence(merged_df):
    """
    Computes the proportion of recommendations that suggest hotels in different cities.

    Args:
        merged_df (DataFrame): Recommendations enriched with hotel details.

    Returns:
        city_divergence (DataFrame): Hotel-wise city divergence stats.
        overall_divergence_ratio (float): Overall proportion of out-of-city recommendations.
    """
    # a column to indicate if the recommendation is from another city
    merged_df['is_another_city'] = merged_df['hotel_city_hotel_name'] != merged_df['reco_hotel_city_hotel_name']

    # divergence ratio
    city_divergence = merged_df.groupby('hotel_id').agg(
        total_recommendations=('is_another_city', 'size'),
        another_city_recommendations=('is_another_city', 'sum')
    )

    #  divergence ratio per hotel
    city_divergence['divergence_ratio'] = city_divergence['another_city_recommendations'] / city_divergence['total_recommendations']

    # overall divergence ratio across all hotel recommendations
    overall_divergence_ratio = city_divergence['another_city_recommendations'].sum() / city_divergence['total_recommendations'].sum()

    return city_divergence, overall_divergence_ratio

def validate_recommendations():
    """
    Loads recommendations and hotel metadata, computes validation metrics, and prints results.
    """
    #  hotel metadata and recommendations
    hotel_info = pd.read_csv(f"{config.DATA_PATH}/hotel_info.csv")
    recommendation_df = pd.read_csv(f"{config.DATA_PATH}/hotel_recommendations.csv")

    #  recommendations with hotel metadata
    merged_df = merge_hotel_metadata(recommendation_df, hotel_info)

    #  city divergence statistics
    city_divergence, overall_divergence_ratio = compute_city_divergence(merged_df)

    #  overall divergence ratio
    print(f" Overall divergence ratio (recommendations from another city): {overall_divergence_ratio:.2%}")

    return merged_df, city_divergence, overall_divergence_ratio
