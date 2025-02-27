# IMPORTS:
import pandas as pd
import numpy as np
import os
import re
import logging
import pickle
from scipy.spatial.distance import cdist  # For computing Euclidean distance
from typing import Tuple, List, Dict
from gensim.models import word2vec, KeyedVectors
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
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
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(f"Invalid sheet name: {fnfe}")
    except ValueError as ve:
        raise ValueError(f"Invalid sheet name: {ve}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def save_df_to_excel(output_name: str, df: pd.DataFrame, sheet_name: str = "Sheet1") -> None:
    """
    This function saves a dataframe to an Excel file.
    :param output_name: The name of the Excel file you want to create
    :param df: The dataframe you want to save to Excel
    :param sheet_name: The name of the sheet you want to save to (optional)
    """
    # Validating the input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The 'df' parameter must be a pandas DataFrame.")

    try:
        with pd.ExcelWriter(output_name, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    except Exception as e:
        logging.exception(f"An error occurred while saving the DataFrame to Excel: {e}.")
        raise
    return None


# Download stopwords if not already done
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def clean_phrases(df: pd.DataFrame, phrase_column: str = "Phrases") -> pd.DataFrame:
    """
    This function cleans the phrases DataFrame by:
      - Removing duplicate phrases
      - Removing phrases that are too short (or too long)
      - Removing phrases that consist entirely of stopwords

    :param df: DataFrame containing the phrases
    :param phrase_column: Name of the column with the phrases
    :return: A cleaned DataFrame
    """
    # Removing duplicate phrases.
    df = df.drop_duplicates(subset=phrase_column)

    # Removing outliers by length (= shorter than 5 characters or longer than 300 characters)
    df = df[df[phrase_column].str.len() > 5]
    df = df[df[phrase_column].str.len() < 300]

    # Removing phrases which are composed mostly of stopwords
    def has_few_stopwords(phrase):
        """This function takes a phrase and removes phrases based on the too many stopwords criteria."""
        tokens = phrase.lower().split()
        if not tokens:
            return False

        # If more than 60% words are stopwords, we will filter it out
        stopword_ratio = sum(1 for token in tokens if token in stop_words) / len(tokens)  # Calculating the stopword ratio in the phrase
        return stopword_ratio < 0.6

    df = df[df[phrase_column].apply(has_few_stopwords)]
    return df


def preprocess_text(phrase: str) -> str:
    """
    This function takes a phrase, converts it to lowercase and removes punctuation, question marks, and other unnecessary characters.
    :param phrase: Individual phrase from the phrases file
    :return: Preprocessed phrase
    """
    # Handling the case when the input is not a string
    if not isinstance(phrase, str):
        raise TypeError("Input must be a string")

    phrase = phrase.lower()
    phrase = re.sub(r'[^\w\s]', '', phrase)
    return phrase


def get_word_embeddings(phrase: str, w2v_model: word2vec.KeyedVectors) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    This function creates embeddings for each individual tokens in the phrases and also logs missing tokens not found in our model.
    :param phrase: Individual phrase from the phrases file
    :param w2v_model: Our loaded word2Vec model
    :return: Dictionary of embeddings and a list of missing tokens
    """
    # Ensuring that the phrase can be converted to a string
    try:
        phrase = str(phrase)
    except Exception as e:
        raise ValueError(f"Invalid phrase input; must be convertible to string. {e}")

    # Validate that the model supports the necessary operations
    if not (hasattr(w2v_model, '__contains__') and hasattr(w2v_model, '__getitem__')):
        raise TypeError("w2v_model must support containment (in operator) and item retrieval (bracket notation).")

    # Firstly pre-processing the phrase by converting to lowercase & removing punctuation and other symbols like "?!."
    preprocessed_phrase = preprocess_text(phrase)

    # Splitting the phrase into a list of strings = individual tokens
    tokens = preprocessed_phrase.split()

    embeddings = {}     # Embeddings for each token
    missing_tokens = [] # List of tokens which are not in our w2v_model

    # Creating the embeddings for each token
    for token in tokens:
        # If the token is in the loaded w2v_model, we add it to our dictionary
        try:
            if token in w2v_model:
                embeddings[token] = w2v_model[token]
            else:
                missing_tokens.append(token)
        except Exception as e:
            # Logging the exception and considering the token as missing
            logging.exception(f"Error retrieving embedding for token '{token}': {e}")
            missing_tokens.append(token)

    return embeddings, missing_tokens


def aggregate_phrase_embedding(embeddings: dict, w2v_model: word2vec.KeyedVectors) -> np.ndarray:
    """
    This function takes the single token embeddings and aggregates them into a single normalised vector for the entire phrase.
    This is done by summing the embeddings and normalising this sum
    :param embeddings: Dictionary mapping tokens to their embedding vectors
    :param w2v_model: The loaded Word2Vec model (used here to get the vector size)
    :return: A normalised numpy array representing the aggregated phrase embedding. If no embeddings are present, it returns a zero vector.
    """
    # If we have some embeddings in the dict
    if embeddings and len(embeddings) > 0:
        embedding_vectors = list(embeddings.values())                    # Extracting the vectors from the dict
        aggregated_vector = np.sum(np.array(embedding_vectors), axis=0)  # Computing the sum of the vectors (axis=0 means row by row)

        # Normalising the sum because the Euclidian norm should be = 1 (important for calculating cosine similarity)
        norm = np.linalg.norm(aggregated_vector)
        if norm > 0:
            aggregated_vector = aggregated_vector / norm
        # We do not need to normalise a zero vector if there is one
        return aggregated_vector

    # If there is no embedding, we return a zero vector
    else:
        return np.zeros(w2v_model.vector_size)


def compute_distance_matrix(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    This function computes the pairwise distance matrix for the given embeddings using either Cosine or Euclidean distance.
    To be memory computationally efficient, I will try to use vectorised operations in numpy as these are the most optimised.
    The "cdist" is much faster than Python loops and the np.dot calculation is also much faster because it is vectorised.
    :param embeddings: A NumPy array of shape (n, d) where n is the number of phrases and d is the embedding dimension.
                       It's assumed that the embeddings are normalized if using cosine distance.
    :param metric: A string specifying the metric to use: either "cosine" or "euclidean", default is "cosine".
    :return: A NumPy array of shape (n, n) representing the pairwise distance matrix.
    """
    # Validating the input of embeddings:
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("The 'embeddings' parameter must be a numpy array.")
    if embeddings.ndim != 2:
        raise ValueError("The 'embeddings' array must be two-dimensional (shape: (n, d)).")
    if embeddings.size == 0:
        raise ValueError("The 'embeddings' array is empty.")

    # If the user wants cosine distance
    if metric.lower() == "cosine":
        # If the embeddings are normalized, the cosine similarity is the dot product.
        similarity_matrix = np.dot(embeddings, embeddings.T)
        distance_matrix = 1 - similarity_matrix

    # If the user wants euclidean distance
    elif metric.lower() == "euclidean":
        distance_matrix = cdist(embeddings, embeddings, metric="euclidean")

    # If the user inputs an invalid metric
    else:
        raise ValueError("Unsupported metric. Please choose 'cosine' or 'euclidean' as the metric argument of the 'compute_distance_matrix' function.")

    # Sometimes there are very tiny values close to zero on the diagonals but not exactly zero (due to the calculations like dot product), we can fix this:
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix


def visualise_distance_matrix(distance_matrix: np.ndarray) -> None:
    """
    This function creates a visual representation of the given distance matrix.
    :return: A visual representation of the given distance matrix.
    """
    plt.figure(figsize=(10, 8))
    plt.ion()
    sns.heatmap(distance_matrix, cmap="viridis")
    plt.title("Cosine Distance Matrix")
    plt.xlabel("Phrases")
    plt.ylabel("Phrases")
    plt.show()
    return None


def find_closest_match(input_phrase: str, df: pd.DataFrame, w2v_model: word2vec.KeyedVectors, metric: str = "cosine") -> Tuple[str, float]:
    """
    This function computes the aggregated embedding of an input phrase and then finds the closest matching phrase from the "phrases" dataframe.
    :param input_phrase: The phrase provided by the user
    :param df: A pandas DataFrame containing at least two columns:
               - "Phrases": The original phrases
               - "Aggregated Embedding": The aggregated (normalized) embedding for each phrase
    :param w2v_model: The preloaded Word2Vec model
    :param metric: The distance metric to use: either "cosine" or "euclidean", default is "cosine"
    :return: A tuple containing:
             - The closest matching phrase (str)
             - The distance (float) between the input phrase and the closest phrase
    """
    # Computing embeddings for each token of the given input phrase
    token_embeddings, missing_tokens = get_word_embeddings(input_phrase, w2v_model)

    # Aggregating the token embeddings to obtain a single normalized vector
    input_embedding = aggregate_phrase_embedding(token_embeddings, w2v_model)

    # Stacking all aggregated embeddings from the DataFrame into a NumPy array
    embeddings_matrix = np.vstack(df["Aggregated Embedding"].values)

    # Computing the distances between the input_embedding and each phrase's (from our file) embedding.
    if metric.lower() == "cosine":
        similarities = np.dot(embeddings_matrix, input_embedding)
        distances = 1 - similarities
    elif metric.lower() == "euclidean":
        distances = np.linalg.norm(embeddings_matrix - input_embedding, axis=1)
    else:
        raise ValueError("Unsupported metric. Please choose 'cosine' or 'euclidean' as the metric argument of the 'compute_distance_matrix' function.")

    # Finding the index of the minimum distance (= the closest match phrase)
    best_idx = np.argmin(distances)
    closest_phrase = df.iloc[best_idx]["Phrases"]
    closest_distance = distances[best_idx]

    return closest_phrase, closest_distance


def interactive_phrase_lookup(phrases: pd.DataFrame, w2v_model: word2vec.KeyedVectors) -> None:
    """
    This function interactively prompts the user to input a phrase and a distance metric. It then
    finds the closest matching phrase from the DataFrame using the given Word2Vec model,
    and displays the result. After each query, the user is asked if they wish to continue or not.
    :param phrases: A pandas DataFrame with at least two columns:
                    - "Phrases": The original phrases
                    - "Aggregated Embedding": The aggregated (normalized) embedding for each phrase
    :param w2v_model: The preloaded Word2Vec model (gensim KeyedVectors)
    :return: This function does not return anything, it only runs
    """
    while True:
        # Prompting user for the input phrase
        user_input = input("Enter any phrase: ").strip()
        if not user_input:
            print("No phrase entered. Please try again and enter some phrase.")
            continue

        # Prompting user for the distance metric (default is "cosine")
        user_metric = input("Enter the distance metric (cosine/euclidean) [cosine]: ").strip().lower() or "cosine"

        # Validation
        while user_metric not in ("cosine", "euclidean"):
            print("Invalid input. Please enter 'cosine' or 'euclidean'.")
            user_metric = input("Enter the distance metric (cosine/euclidean) [cosine]: ").strip().lower() or "cosine"

        # Finding the closest matching phrase
        closest_phrase, closest_distance = find_closest_match(user_input, phrases, w2v_model, metric=user_metric)

        # Displaying the results
        print("Closest matching phrase:", closest_phrase)
        print("The {} distance is: {:.4f}".format(user_metric.capitalize(), closest_distance))

        # Asking the user if they want to continue with another phrase or not
        continue_choice = input("Would you like to continue with another phrase? (yes/no): ").strip().lower()
        if continue_choice not in ("yes", "y"):
            print("Exiting the program. Goodbye!")
            break

    return None


def load_word_embeddings(vectors_path: str, embedding_vectors_location: str, vectors_file: str, limit, pickle_file: str = None) -> word2vec.KeyedVectors:
    """
    This function loads the word embeddings for the first million vectors. If a pickle file is provided and exists, it loads the model from it
    to save time. Otherwise, it loads it from the binary file, saves to the text file, and optionally pickles the model.
    :param vectors_path: Path to the saved embeddings file (text format).
    :param embedding_vectors_location: Path to the binary file of pretrained embeddings.
    :param vectors_file: Path to save the embeddings in text format if they are loaded from binary.
    :param limit: The number of vectors you want to load
    :param pickle_file: Optional path to a pickle file for caching.
    :return: A KeyedVectors object with the word embeddings.
    """
    try:
        # If a pickle file is specified and exists, load from it
        if pickle_file and os.path.exists(pickle_file):
            logging.info("Loading word embeddings from pickle file: %s", pickle_file)
            with open(pickle_file, 'rb') as f:
                w2v_model = pickle.load(f)
            return w2v_model

        # If the saved text file exists, load using memory mapping
        if os.path.exists(vectors_path):
            logging.info("Loading word embeddings from saved file: %s", vectors_path)
            w2v_model = KeyedVectors.load_word2vec_format(vectors_path, binary=False, limit=limit)
        else:
            # Otherwise, load from the binary file and save as text
            logging.info("Saved embeddings file not found. Loading from binary file: %s", embedding_vectors_location)
            w2v_model = KeyedVectors.load_word2vec_format(embedding_vectors_location, binary=True, limit=limit)
            w2v_model.save_word2vec_format(vectors_file)
            logging.info("Saved word embeddings to %s", vectors_file)

        # Optionally, cache the model using pickle for faster future loads
        if pickle_file:
            logging.info("Pickling the word embeddings to: %s", pickle_file)
            with open(pickle_file, 'wb') as f:
                pickle.dump(w2v_model, f)

        return w2v_model

    except Exception as e:
        logging.exception("An error occurred while loading word embeddings: %s", e)
        raise RuntimeError(f"Failed to load word embeddings: {e}")


def get_base_dir():
    """
    This function returns the base directory for this specific project. If the "GALYTIX_BASE_DIR"
    environment variable has been set then that value will be used otherwise it defaults to the current working directory.
    """
    return os.environ.get("GALYTIX_BASE_DIR", os.getcwd())




