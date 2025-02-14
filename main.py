# IMPORTS:
import gensim
from gensim.models import KeyedVectors
from function_definitions import *
# -----------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # 1) SETUP:
    set_pandas_output()  # This just improves the Pandas output in the PyCharm terminal
    user_name = get_local_username()
    input_file_name = "phrases"
    input_file_path = f"C:/Users/{user_name}/PycharmProjects/Galytix/{input_file_name}.xlsx"
    embedding_vectors = "GoogleNews-vectors-negative300"
    embedding_vectors_location = f"C:/Users/{user_name}/PycharmProjects/Galytix/{embedding_vectors}.bin"

    # 2) Loading the "phrases" file:
    df_raw = read_excel(input_file_path)
    print(df_raw)

    # 3) Loading the word embeddings for first million vectors
    wv = KeyedVectors.load_word2vec_format(embedding_vectors_location, binary=True, limit=1000000)
    wv.save_word2vec_format('vectors.csv')  # Saving the embeddings as a flatfile (csv)



