import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
import src.config as config

def merge_hotel_descriptions(hotel_desc):
    """
    Merges all descriptions per hotel_id and performs minor text cleaning.
    
    Args:
        hotel_desc (DataFrame): Processed hotel descriptions.
    
    Returns:
        hotel_desc_merged (DataFrame): Merged descriptions per hotel.
    """
    # grouping descriptions per hotel_id
    hotel_desc_merged = hotel_desc.groupby("hotel_id")["lemmatized_text"].apply(lambda x: " ".join(x)).reset_index()

    # Removing unwanted words
    hotel_desc_merged["lemmatized_text"] = hotel_desc_merged["lemmatized_text"].str.replace("misafir", "", regex=True)
    hotel_desc_merged["lemmatized_text"] = hotel_desc_merged["lemmatized_text"].str.replace("km", "", regex=True)

    return hotel_desc_merged

def train_topic_model(hotel_desc_merged):
    """
    Trains a BERTopic model on the hotel descriptions.
    
    Args:
        hotel_desc_merged (DataFrame): Merged descriptions per hotel.
    
    Returns:
        hotel_desc_merged (DataFrame): Dataframe with assigned topics and representations.
        topic_model (BERTopic): Trained BERTopic model.
    """
    # Initialize BERTopic model (multilingual)
    topic_model = BERTopic(language="multilingual")

    # Fitting BERTopic on lemmatized hotel descriptions
    topics, _ = topic_model.fit_transform(hotel_desc_merged["lemmatized_text"])
    hotel_desc_merged["topic"] = topics

    # Extracting topic representations
    topic_representations = topic_model.get_topics()
    topic_words_dict = {topic: ", ".join([word[0] for word in words[:10]]) for topic, words in topic_representations.items()}

    # mapping: topic words to dataset
    hotel_desc_merged["representation"] = hotel_desc_merged["topic"].map(topic_words_dict)

    # saving to folder of model
    topic_model.save(f"{config.MODEL_PATH}/topic_model")

    return hotel_desc_merged, topic_model

def extract_topic_embeddings(hotel_desc_merged, topic_model):
    """
    Extracts topic embeddings from BERTopic.
    
    Args:
        hotel_desc_merged (DataFrame): Dataframe with topic assignments.
        topic_model (BERTopic): Trained BERTopic model.
    
    Returns:
        hotel_desc_merged (DataFrame): Updated dataframe with topic embeddings.
    """
    # Extract topic embeddings
    topic_embeddings = np.array(topic_model.topic_embeddings_)

    # Create a mapping of topic_id -> embedding vector
    topic_embedding_dict = {i: topic_embeddings[i] for i in range(len(topic_embeddings))}

    # Map topic embeddings to hotels
    hotel_desc_merged["topic_embedding"] = hotel_desc_merged["topic"].map(topic_embedding_dict)

    return hotel_desc_merged

def save_processed_topic_data(hotel_desc_merged):
    """
    Saves the processed hotel topic dataset.
    
    Args:
        hotel_desc_merged (DataFrame): Final processed topic data.
    """
    hotel_desc_merged.to_csv(f"{config.DATA_PATH}/hotel_desc_with_topics.csv", index=False)
    print(" Processed topic dataset saved as 'hotel_desc_with_topics.csv'.")

