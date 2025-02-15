# IMPORTS:
from gensim.models import KeyedVectors
from function_definitions import *
import logging
# -----------------------------------------------------------------------------------------------------


# Configuring the basic logging:
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


if __name__ == '__main__':
    logging.info("Starting the main process...")

    # 1) Setup:
    set_pandas_output()  # This just improves the Pandas output in the PyCharm terminal
    user_name = get_local_username()
    input_file_name = "phrases"
    input_file_path = f"C:/Users/{user_name}/PycharmProjects/Galytix/{input_file_name}.xlsx"
    embedding_vectors = "GoogleNews-vectors-negative300"
    embedding_vectors_location = f"C:/Users/{user_name}/PycharmProjects/Galytix/{embedding_vectors}.bin"
    vectors_file = "vectors.csv"
    vectors_path = f"C:/Users/{user_name}/PycharmProjects/Galytix/{vectors_file}"
    pickle_file = f"C:/Users/{user_name}/PycharmProjects/Galytix/w2v_model.pkl"  # optional

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

    # 4) Creating 2 new columns for: embeddings and also missing tokens from our model (this can be processed further if necessary)
    logging.info("Generating token embeddings and missing tokens for each phrase...")
    phrases[["Embeddings", "Missing Tokens"]] = phrases["Phrases"].apply(lambda phrase: pd.Series(get_word_embeddings(phrase, w2v_model)))
    logging.info("Token embeddings successfully generated for all phrases.")
    # Note: Each word (token) is represented by a high-dimensional vector

    # 5) Creating a new column which calculates the embedding for the whole phrase and normalises it
    logging.info("Aggregating token embeddings to create a phrase-level embedding...")
    phrases["Aggregated Embedding"] = phrases["Embeddings"].apply(lambda embedding: aggregate_phrase_embedding(embedding, w2v_model))
    save_df_to_excel(output_name="phrases_extended.xlsx", df=phrases)
    logging.info("Aggregating token embeddings completed...Saved the extended phrases file: phrases_extended.xlsx")

    # 6) Calculating the Euclidian or Cosine distance of each phrase to all other phrases
    logging.info("Computing the pairwise distance matrix...")
    phrase_embeddings = np.vstack(phrases['Aggregated Embedding'].values)                 # Converting the column of embeddings into a NumPy array
    # Here you can choose the metric to be cosine or euclidean
    cosine_distance_matrix = compute_distance_matrix(phrase_embeddings, metric='cosine')  # Computing the distance matrix
    logging.info("Distance matrix computed.")

    # 7) Viewing this distance matrix
    labels = phrases["Phrases"]
    distance_matrix_df = pd.DataFrame(cosine_distance_matrix, index=labels, columns=labels)
    save_df_to_excel(output_name="distance_matrix.xlsx", df=distance_matrix_df)
    logging.info("Saved distance matrix to Excel: distance_matrix.xlsx")
    visualise_distance_matrix(cosine_distance_matrix)

    # 8) Finding the most similar phrase from the phrases file to a given phrase
    interactive_phrase_lookup(phrases, w2v_model)