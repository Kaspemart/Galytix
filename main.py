# IMPORTS:
import gensim
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
    print(phrases)

    # 3) Loading the word embeddings for the first million vectors
    # If the file already exists, we load it from the saved csv file (to avoid having to save the million word embeddings again)


    # 4) Converting the phrases to lowercase & Removing punctuation and other symbols like "?!."
    phrases['Preprocessed Phrases'] = phrases['Phrases'].apply(preprocess_text)
    print(phrases)


    # 5)

