# Galytix-Casestudy
This project is about integrating pretrained Word2Vec embeddings into a data processing pipeline to compute semantic distances between
phrases. The pipeline downloads and converts one million vectors from a binary format using gensim, assigns each word its corresponding
embedding, and approximates phrase-level vectors through normalized aggregation of individual word vectors. It then calculates similarity
measures, both in batch and on the fly, using Euclidean or cosine distance.

------------------------------------------------------------------------------------------------------------------------------------------

## Notes
* I converted the phrases.csv file to excel as there was no reason having it as a csv file

------------------------------------------------------------------------------------------------------------------------------------------
## Project info
* Author: Martin Kasperlik (martin.kasperlik77@seznam.cz)
* Purpose: To try to convince Galytix that I have solid foundations and that I am teachable in areas that need attention and improvement
* Python version used: 3.9

------------------------------------------------------------------------------------------------------------------------------------------

## How to run the app
* XXX
* XXX

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

