# Content Based Movie Recommendation System

This repository contains a movie recommendation system built using the TF-IDF and SVD techniques. The system is designed to recommend movies to users based on the similarity between the movies' descriptions, genres, keywords, cast and crew.

## Introduction

A movie recommendation system is a tool that suggests movies to users based on their preferences and interests. This project is a movie recommendation system that recommends movies to users based on the movies which they have already watched.

## Requirements

To run this recommendation system, you will need the following software:

Python 3.9.12
<br />
Pandas
<br />
Numpy
<br />
Scikit-learn
<br />
NLTK

## Datasets
Source: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
<br />
The dataset used in this project is composed of three files:

movies_metadata.csv: Contains metadata of movies, including the title, id, overview, and genres.
<br />
keywords.csv: Contains keywords for each movie.
<br />
credits.csv: Contains information about the cast and crew of each movie.

## Implementation

The recommendation system is implemented in the recommendations.py file. The code performs the following steps:

1. Load and clean the datasets.

2. Preprocess the dataset, including removing stopwords, stemming and lemmatizing the words.

3. Compute the TF-IDF scores for each movie description.

4. Use Truncated SVD to reduce the dimensionality of the TF-IDF scores.

5. Calculate the cosine similarity between each pair of movies.

6. Make recommendations by finding the most similar movies to the movie of interest.

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/jhasiddhant/movie-recommendation-system.git
```
Install the required packages:
```bash
pip install -r requirements.txt
```
Run the recommendation.py file:
```bash
python recommendation.py
```
When prompted, enter the name of a movie to receive recommendations for similar movies.

## Example:
Enter movie name: The lord of the rings

Recommended Movies:
<br />
THE LORD OF THE RINGS: THE TWO TOWERS
<br />
THE LORD OF THE RINGS: THE FELLOWSHIP OF THE RING
<br />
THE RING THING
<br />
THE HUNT FOR GOLLUM
<br />
THE LORD OF THE RINGS: THE RETURN OF THE KING
<br />
THE RETURN OF THE KING
<br />
WOLF
<br />
THE HOBBIT: AN UNEXPECTED JOURNEY
<br />
THE HOBBIT: THE DESOLATION OF SMAUG
<br />
GOR