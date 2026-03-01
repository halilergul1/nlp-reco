import argparse
import pandas as pd
import src.config as config
from src.data_preprocessing import preprocess_data
from utils.text_processing import process_hotel_descriptions, save_processed_descriptions, get_text_processor
from src.topic_modeling import merge_hotel_descriptions, train_topic_model, extract_topic_embeddings, save_processed_topic_data
from src.recommendation import clean_and_convert_embedding, compute_cosine_similarity, generate_recommendations, handle_missing_embeddings, rank_recommendations, save_recommendations
from src.validation import validate_recommendations


def parse_args():
    parser = argparse.ArgumentParser(description="NLP Hotel Recommendation Pipeline")
    parser.add_argument(
        "--processor",
        choices=["turkish", "fallback"],
        default="turkish",
        help="Text processor to use. 'turkish' requires Zemberek + Java. 'fallback' works anywhere.",
    )
    return parser.parse_args()


def main():
    """Main function to run the hotel recommendation pipeline."""
    args = parse_args()

    print("\n Step 1: Data Preprocessing...")
    hotel_info, hotel_desc, session_data = preprocess_data()

    print("\n Step 2: Text Processing...")
    processor = get_text_processor(args.processor)
    hotel_desc_processed = process_hotel_descriptions(hotel_desc, processor=processor)
    save_processed_descriptions(hotel_desc_processed)

    print("\n Step 3: Topic Modeling...")
    hotel_desc_merged = merge_hotel_descriptions(hotel_desc_processed)
    hotel_desc_merged, topic_model = train_topic_model(hotel_desc_merged)
    hotel_desc_merged = extract_topic_embeddings(hotel_desc_merged, topic_model)
    save_processed_topic_data(hotel_desc_merged)

    print("\n Step 4: Generating Recommendations...")
    final_df = hotel_desc_merged.merge(hotel_info[["hotel_id", "normalized_star_rating", "popularity_score"]], on="hotel_id", how="left")

    # topic embeddings from string to NumPy arrays
    final_df['topic_embedding'] = final_df['topic_embedding'].apply(clean_and_convert_embedding)

    # cosine similarity
    cosine_sim_matrix, valid_indices = compute_cosine_similarity(final_df)

    # recommendations
    recommendations = generate_recommendations(final_df, cosine_sim_matrix, valid_indices)

    # handling missing embeddings using popularity ranking
    recommendations = handle_missing_embeddings(final_df, recommendations, valid_indices)

    # Rank and save recommendations
    recommendation_df = rank_recommendations(final_df, recommendations)
    save_recommendations(recommendation_df)

    print("\n Step 5: Validating Recommendations...")
    merged_df, city_divergence, overall_divergence_ratio = validate_recommendations()

    print("\n Pipeline execution complete!")

    # also validate whether we did not recommend a hotel to itself
    assert merged_df.query("hotel_id == reco_hotel_id").empty, "A hotel is recommended to itself!"


if __name__ == "__main__":
    main()
