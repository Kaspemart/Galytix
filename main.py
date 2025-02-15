# IMPORTS:
from function_definitions import *
import logging
import os
# -----------------------------------------------------------------------------------------------------


# Configuring the basic logging:
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


if __name__ == '__main__':
    logging.info("Starting the main process...")

    # 1) Setup:
    set_pandas_output()                 # This just improves the Pandas output in the PyCharm terminal
    base_directory = get_base_dir()     # Returning the current working directory
    input_file_name = "phrases.xlsx"
    embedding_vectors = "GoogleNews-vectors-negative300.bin"
    vectors_file = "vectors.csv"
    input_file_path = os.path.join(base_directory, input_file_name)
    embedding_vectors_location = os.path.join(base_directory, embedding_vectors)
    vectors_path = os.path.join(base_directory, vectors_file)
    pickle_file = os.path.join(base_directory, "w2v_model.pkl")  # Optional

    # 2) Loading the "phrases" file:
    logging.info("Loading the phrases file from %s", input_file_path)
    phrases = read_excel(input_file_path)
    logging.info("Successfully loaded the phrases file with %d rows.", len(phrases))

    # 3) Loading the word embeddings for the first million vectors
    w2v_model = load_word_embeddings(vectors_path=vectors_path,
                                     embedding_vectors_location=embedding_vectors_location,
                                     vectors_file=vectors_file,
                                     limit=1000000,
                                     pickle_file=pickle_file)

    # 4) Cleaning the phrases DataFrame (removing duplicates, outliers, and stopword-heavy phrases)
    phrases = clean_phrases(phrases, phrase_column="Phrases")
    logging.info("Cleaned the phrases file. Remaining rows: %d", len(phrases))

    # 5) Creating 2 new columns for: embeddings and also missing tokens from our model (this can be processed further if necessary)
    logging.info("Generating token embeddings and missing tokens for each phrase...")
    phrases[["Embeddings", "Missing Tokens"]] = phrases["Phrases"].apply(lambda phrase: pd.Series(get_word_embeddings(phrase, w2v_model)))
    logging.info("Token embeddings successfully generated for all phrases.")
    # Note: Each word (token) is represented by a high-dimensional vector

    # 6) Creating a new column which calculates the embedding for the whole phrase and normalises it
    logging.info("Aggregating token embeddings to create a phrase-level embedding...")
    phrases["Aggregated Embedding"] = phrases["Embeddings"].apply(lambda embedding: aggregate_phrase_embedding(embedding, w2v_model))
    save_df_to_excel(output_name="phrases_extended.xlsx", df=phrases)
    logging.info("Aggregating token embeddings completed...Saved the extended phrases file: phrases_extended.xlsx")

    # 7) Calculating the Euclidian or Cosine distance of each phrase to all other phrases
    logging.info("Computing the pairwise distance matrix...")
    phrase_embeddings = np.vstack(phrases['Aggregated Embedding'].values)                 # Converting the column of embeddings into a NumPy array
    # Here you can choose the metric to be cosine or euclidean
    cosine_distance_matrix = compute_distance_matrix(phrase_embeddings, metric='cosine')  # Computing the distance matrix
    logging.info("Distance matrix computed.")

    # 8) Viewing this distance matrix
    labels = phrases["Phrases"]
    distance_matrix_df = pd.DataFrame(cosine_distance_matrix, index=labels, columns=labels)
    save_df_to_excel(output_name="distance_matrix.xlsx", df=distance_matrix_df)
    logging.info("Saved distance matrix to Excel: distance_matrix.xlsx")
    visualise_distance_matrix(cosine_distance_matrix)

    # 9) Finding the most similar phrase from the phrases file to a given phrase
    interactive_phrase_lookup(phrases, w2v_model)