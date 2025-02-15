# Galytix-Casestudy
This project is about integrating pretrained Word2Vec embeddings into a data processing pipeline to compute semantic distances between
phrases. The pipeline downloads and converts one million vectors from a binary format using gensim, assigns each word its corresponding
embedding, and approximates phrase-level vectors through normalized aggregation of individual word vectors. It then calculates similarity
measures, both in batch and on the fly, using Euclidean or cosine distance.

------------------------------------------------------------------------------------------------------------------------------------------

## Notes
* I converted the phrases.csv file to excel as there was no reason having it as a csv file

------------------------------------------------------------------------------------------------------------------------------------------

## How to structure and optimise the code in the future (if I had more time):
* The code could be even more OOP-friendly and could be structured into classes for instance
* There could also be a nice user-friendly interface created
* The code cold probably be optimised even further with regards to speed and memory efficience
* I would improve the structure of the project, maybe something like:
* Galytix/
├── galytix/
│   ├── __init__.py
│   ├── data_io.py
│   ├── embedding.py
│   ├── similarity.py
│   ├── visualization.py
│   ├── interactive.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_data_io.py
│   ├── test_embedding.py
│   ├── test_similarity.py
│   └── test_visualization.py
├── main.py
├── setup.py
├── requirements.txt
└── README.md

------------------------------------------------------------------------------------------------------------------------------------------
## Project info
* Author: Martin Kasperlik (martin.kasperlik77@seznam.cz)
* Purpose: To try to convince Galytix that I have solid foundations and that I am teachable in areas that need attention and improvement
* Python version used: 3.9

------------------------------------------------------------------------------------------------------------------------------------------

## How to run the app
* 

------------------------------------------------------------------------------------------------------------------------------------------

## How to run the Unit tests
* Locate to the directory of the project and in the terminal write: "python -m pytest"

------------------------------------------------------------------------------------------------------------------------------------------

## Files info
* requirements.md - Includes necessary packages (and versions I used) to be installed for this app to work
* function_definitions.py - Includes all functions in this app
* main.py - Includes the main runnable python file
* test_functions.py - Includes the unit tests for different parts of the code (not all)

------------------------------------------------------------------------------------------------------------------------------------------

