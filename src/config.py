# config.py

DATA_PATH = "data/"
ZEMBEREK_PATH = "jars/zemberek-full.jar"
STOPWORDS_FILE = "data/stopwords.txt"

# Weight configurations
STAR_RATING_WEIGHTS = {
    "hotel_avg_rating_general": 0.3,
    "hotel_avg_rating_food": 0.25,
    "hotel_avg_rating_cleaning": 0.1,
    "hotel_avg_rating_location": 0.1,
    "hotel_avg_rating_service": 0.1,
    "hotel_avg_rating_wifi": 0.05,
    "hotel_avg_rating_condition": 0.05,
    "hotel_avg_rating_price": 0.05,
}

COMMENT_WEIGHT = 0.7
IMAGE_WEIGHT = 0.3

# Number of top recommendations
TOP_N = 10

MODEL_PATH = "models/"
PROCESSOR_TYPE = "turkish"  # "turkish" | "fallback"
