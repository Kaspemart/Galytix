# IMPORTS:
from gensim.models import KeyedVectors
from function_definitions import *
# -----------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # 1) Setup:
    set_pandas_output()  # This just improves the Pandas output in the PyCharm terminal
    user_name = get_local_username()
    input_file_name = "phrases"
    input_file_path = f"C:/Users/{user_name}/PycharmProjects/Galytix/{input_file_name}.xlsx"
    embedding_vectors = "GoogleNews-vectors-negative300"
    embedding_vectors_location = f"C:/Users/{user_name}/PycharmProjects/Galytix/{embedding_vectors}.bin"
    vectors_file = "vectors.csv"
    vectors_path = f"C:/Users/{user_name}/PycharmProjects/Galytix/{embedding_vectors}"

    # 2) Loading the "phrases" file:
    phrases = read_excel(input_file_path)

    # 3) Loading the word embeddings for the first million vectors
    # If the file already exists, we load it from the saved csv file (to avoid having to save the million word embeddings again)
    if os.path.exists(vectors_path):
        w2v_model = KeyedVectors.load_word2vec_format(vectors_path, binary=False)
        # If it does not, we load it for the first time
    else:
        w2v_model = KeyedVectors.load_word2vec_format(embedding_vectors_location, binary=True, limit=1000000)
        w2v_model.save_word2vec_format(vectors_file)

    # 4) Creating 2 new columns for: embeddings and also missing tokens from our model (this can be processed further if necessary)
    phrases[["Embeddings", "Missing Tokens"]] = phrases["Phrases"].apply(lambda phrase: pd.Series(get_word_embeddings(phrase, w2v_model)))
    # Note: Each word (token) is represented by a high-dimensional vector

    # 5) Creating a new column which calculates the embedding for the whole phrase and normalises it
    phrases["Aggregated Embedding"] = phrases["Embeddings"].apply(lambda embedding: aggregate_phrase_embedding(embedding, w2v_model))
    save_df_to_excel(output_name="phrases_extended.xlsx", df=phrases)

    # 6) Calculating the Euclidian or Cosine distance of each phrase to all other phrases
    phrase_embeddings = np.vstack(phrases['Aggregated Embedding'].values)                 # Converting the column of embeddings into a NumPy array
    # Here you can choose the metric to be cosine or euclidean
    cosine_distance_matrix = compute_distance_matrix(phrase_embeddings, metric='cosine')  # Computing the distance matrix

    # 7) Viewing this distance matrix
    labels = phrases["Phrases"]
    distance_matrix_df = pd.DataFrame(cosine_distance_matrix, index=labels, columns=labels)
    save_df_to_excel(output_name="distance_matrix.xlsx", df=distance_matrix_df)
    visualise_distance_matrix(cosine_distance_matrix)

    # 8) Finding the most similar phrase from the phrases file to a given phrase
    interactive_phrase_lookup(phrases, w2v_model)