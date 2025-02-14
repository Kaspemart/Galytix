# IMPORTS:
import pandas as pd
import numpy as np
import os
import re
from typing import Tuple, List, Dict
from gensim.models import word2vec
# -----------------------------------------------------------------------------------------------------------------


def set_pandas_output() -> None:
    """This function improves the output of Pandas dataframes in the terminal."""
    pd.set_option('display.max_columns', None)  # Shows all columns
    pd.set_option('display.width', 300)         # Adjusts width for better formatting
    pd.set_option('display.max_rows', 50)       # Shows up to 100 rows
    pd.set_option('display.float_format', '{:.2f}'.format)
    return None


def get_local_username() -> str:
    """This function returns the local username on your machine."""
    try:
        username = os.getlogin()
        return username
    except OSError as e:
        raise RuntimeError(f"Error getting login name: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def read_excel(file_path: str, sheet_name=0, skiprows=None, nrows=None) -> pd.DataFrame:
    """
    This function reads an Excel file and returns a DataFrame with selected columns and rows.
    :param file_path: Path to the Excel file.
    :param sheet_name: Name or index of the sheet to read.
    :param skiprows: Number of rows or list of row indices to skip at the start.
    :param nrows: Number of rows to read.
    :return: DataFrame with the selected data
    """
    try:
        # Checking if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Reading the data
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows, nrows=nrows)
        return df

    except ValueError as ve:
        raise ValueError(f"Invalid sheet name: {ve}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def preprocess_text(phrase: str) -> str:
    """
    This function takes a phrase, converts it to lowercase and removes punctuation, question marks, and other unnecessary characters.
    :param phrase: Individual phrase from the phrases file
    :return: Preprocessed phrase
    """
    phrase = phrase.lower()
    phrase = re.sub(r'[^\w\s]', '', phrase)
    return phrase


def get_word_embeddings(phrase: str, w2v_model: word2vec.KeyedVectors) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    This function computes a normalized phrase embedding by summing word embeddings.
    :param phrase: Individual phrase from the phrases file
    :param w2v_model: Our loaded word2Vec model
    :return: Normalized phrase embedding
    - A normalized numpy array representing the phrase, or a zero vector if no words are found.
    """
    # Firstly pre-processing the phrase by converting to lowercase & removing punctuation and other symbols like "?!."
    preprocessed_phrase = preprocess_text(phrase)

    # Splitting the phrase into a list of strings = individual tokens
    tokens = preprocessed_phrase.split()

    embeddings = {}     # Embeddings for each token
    missing_tokens = [] # List of tokens which are not in our w2v_model

    # Creating the embeddings for each token
    for token in tokens:
        # If the token is in the loaded w2v_model, we add it to our dictionary
        if token in w2v_model:
            embeddings[token] = w2v_model[token]
        else:
            missing_tokens.append(token)
    return embeddings, missing_tokens




