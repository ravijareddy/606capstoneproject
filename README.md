# Sentiment Analysis and Content-Based Recommendation System

## Overview

This project combines Sentiment Analysis and a Content-Based Recommendation System to analyze text data and recommend items based on user preferences. Using state-of-the-art Natural Language Processing (NLP) models, the system processes textual content to extract sentiments and generate meaningful recommendations.

Features
## DataSet Link:
Dataset after cleaning the data for doing sentiment analysis.

https://drive.google.com/file/d/1omBK4sbebDNdsyuq3QOSyUmTBqf6pCNr/view?usp=drive_link

Dataset after implimenting  News recommendation system using Sentiment Analysis.

https://drive.google.com/file/d/1hczTvBZQCHfAfoN6Z-XrI7O9_YNEe8nE/view?usp=drive_link

## Sentiment Analysis:

Implements models like VADER, BERT, RoBERTa, DistilBERT, and ALBERT.

Identifies positive, negative, and neutral sentiments in text data.

Evaluates and compares model performance to select the most accurate one.

## Content-Based Recommendation System:

Uses item attributes (e.g., keywords, descriptions) to compute similarity between items.

Recommends top-N items based on user interactions.

Employs techniques like TF-IDF and cosine similarity for feature extraction and similarity calculation.

Workflow

## Data Preprocessing:

Clean text data by removing stopwords, punctuation, and special characters.

Convert text into numerical formats using TF-IDF or embeddings.

## Sentiment Analysis:

Apply multiple models (e.g., BERT, RoBERTa) to determine sentiment scores.

Compare model outputs using evaluation metrics (e.g., accuracy, precision, recall).

Select the best-performing model for sentiment predictions.

## Content Recommendation:

Build a similarity matrix using item attributes.

Use cosine similarity to find the most relevant items.

Generate recommendations based on user preferences or query input.

## Evaluation:

Measure sentiment analysis performance using metrics like F1-score.

Validate recommendations with metrics like precision, recall, and coverage.

Models Used

## Sentiment Analysis:

VADER

BERT

RoBERTa

ALBERT

DistilBERT

Feature Vectorization:

TF-IDF

Word Embeddings (e.g., BERT-based embeddings)

Similarity Computation:

Cosine Similarity

## Key Results:

BERT outperformed other sentiment analysis models by accurately predicting sentiment for negative articles.

The recommendation system provided personalized suggestions with high precision, leveraging textual features and cosine similarity.

## Future Improvements:

Integrate user feedback for collaborative filtering.

Explore hybrid recommendation systems combining content-based and collaborative methods.

Optimize computational efficiency for real-time applications.

## References:
1.	NYT articles: 2.1M+ (2000-Present) daily updated. (2024, November 30). Kaggle. https://www.kaggle.com/datasets/aryansingh0909/nyt-articles-21m-2000-present 
2.	Jemai, Fatma & Hayouni, Mohamed & Baccar, Sahbi. (2021). Sentiment Analysis Using Machine Learning Algorithms. 775-779. 10.1109/IWCMC51323.2021.9498965.  (PDF) Sentiment Analysis Using Machine Learning Algorithms 
3.	Gupta, S., Ranjan, R., & Singh, S. N. (2024). Comprehensive Study on Sentiment Analysis: From Rule-based to modern LLM  based system. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2409.09989 
4.	Y. Chandra and A. Jana, "Sentiment Analysis using Machine Learning and Deep Learning," 2020 7th International Conference on Computing for Sustainable Global Development (INDIACom), New Delhi, India, 2020, pp. 1-4, doi: 10.23919/INDIACom49435.2020.9083703.  Sentiment Analysis using Machine Learning and Deep Learning | IEEE Conference Publication | IEEE Xplore

## Team Members
Mohammad Abusufiyan

Poornima Kolasani

Ravija Vangala

