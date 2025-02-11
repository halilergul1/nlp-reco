# NLP-Powered Hotel Recommendation System

## **1. Introduction**

This document details the implementation of an NLP-powered hotel recommendation system I did. The solution involves extracting key topics from hotel descriptions, defining star attributes, and recommending similar hotels based on extracted information. The recommendation model is validated using city divergence ratio analysis. My approach is as follows:
  - **Retrieval stage**: Retreiving top 10 similar hotels based on cosine similarity of their merged text descriptions topic embeddings. 
  - **Ranking stage**: Ranking these hotels based on their popularity scores.

---

## **2. Approach Overview**

### **2.1 Data Preprocessing & Exploratory Data Analysis (EDA)**

- Three datasets were used (not available publicly):
  1. `information_of_hotels.csv`: Contains structured hotel information (ratings, categories, etc.).
  2. `hotel_detailed_descriptions.csv`: Contains textual descriptions of hotels.
  3. `session_data.csv`: Contains user interactions with hotels. I did not use for this project.

- Basic data quality checks were performed and I also explored the data about descriptive statistics and distributions:
  - Checked for missing values and imputed them appropriately.
  - Checked for duplicate records if any.
  - Explored numerical feature distributions and categorical feature counts.
  - Created correlation matrices for better feature understanding. I saw that some of avg rating columns may not be necessary that much because they are highly correlated with general avg rating.

### **2.2 NLP Processing**

- Preprocessing of hotel descriptions included:
  - Removal of HTML tags. After topic modeling I saw some characters like "br" and "li" which are HTML tags. So I removed them.
  - Text normalization and tokenization using `zemberek` (for Turkish text). I used it with JClass and JString classes becuase this is said to be the most efficient way to use zemberek in python environment. SI I got zemberek ful jar file and used it in my code.
  - Stopword removal and lemmatization. I also tried without lemmatization but I saw that lemmatization was way better in terms of extracted topics.
  - Cleaning specific unwanted terms (e.g., "misafir", "km"). because this is a hotel data "misafir" does not give any information about the hotel, in terms of semantic meaning. So I removed it.

- Topic modeling was performed using `BERTopic`:
  - Extracted key topics for each hotel using multilingual support.
  - Mapped hotels to topics and extracted top representative words per topic.
  - Generated topic embeddings for similarity calculations.
  - I also tried LDA but looking at top representative words and topic visualization, I saw that BERTopic is more satisfactory in this case.
  - Bertmodeling also could not determine some of the hotel descriptions' topics. So they lacked embeddings. I handled this by using popularity score but only for these hotels.

### **2.3 Feature Engineering**

- Defined star attributes:
  - Weighted sum of various hotel rating features (general, food, cleaning, location, etc.).
  - Outlier handling by capping at the 99th percentile. I saw that there are some outliers in the data. So I capped them at 99th percentile to prevent them from affecting the final model.
  - Normalization to a 5-star scale.

- Defined popularity score:
  - Combination of the number of comments and images. I gave 0.5 weight to comments and 0.5 weight to images. I thought that both of them are important in terms of popularity.
  - Capped outliers and normalized between 0 and 1.

- Merged topic information, ratings, and popularity scores into a final dataset.

### **2.4 Recommendation Model**

- Used `cosine similarity` on topic embeddings to find the top 10 similar hotels. Topic embeddings were generated during BERTopic modeling step already and a single embedding reprsent the centroid of the topic.
- If a hotel lacked valid embeddings, recommendations were based on popularity ranking.
- Ranked recommendations using a hybrid score combining similarity and popularity.
- Saved the final recommendations to `hotel_recommendations.csv`.
To sum, recommendation model pipeline based on two things:
    - Retrieval stage: Retreiving top 10 similar hotels based on cosine similarity of their merged text descriptions topic embeddings.
    - Ranking stage: Ranking these hotels based on their popularity score.

# **Results of the model are in hotel_recommendations.csv file.**

## **3. Validation and Observations**

### **3.1 Validation Strategy**

To ensure that the recommendations are reasonable, I examined the ratio of recommended hotels that were located in different cities (divergence ratio):

- **Steps taken:**
  - Merged recommendation results with `hotel_info` to retrieve hotel names and cities.
  - Checked if the recommended hotels belonged to different cities than the source hotel.
  - Computed the percentage of recommendations that were from a different city.

### **3.2 Observations**

- **Overall divergence ratio:** `{overall_divergence_ratio:.2%}`
  - This metric shows the extent to which the recommendations suggest hotels from different cities.
  - A high divergence ratio may indicate issues in the similarity calculation (e.g., hotels in different cities may have similar topics but lack geographical relevance).
  - With some experimental iterations with topic embeddings, I got the best divergence ratio as 0.24. This is a good value in terms of geographical relevance. The low divergence ratio indicates that the recommendations are contextually relevant. This means %75 of the recommendations are from the same city.
  - Certain hotels received recommendations exclusively from their own city, while others had mixed recommendations.
  - This might be an issue if users, or product team, expect geographically relevant recommendations.

- **Potential Bias in Recommendations:**
  - If some cities have a large number of hotels with similar topics (like Antalya), recommendations may become biased toward those locations.