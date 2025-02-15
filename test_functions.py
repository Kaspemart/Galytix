# IMPORTS:
import pytest
from function_definitions import *
# ---------------------------------------------------------------------------------------------------------------------------------
# IMPORTANT: If using pytest, all tests must start with "test_"
# To run the tests, locate to the directory of the project and in the terminal write: "python -m pytest"
# To create a coverage report, in the terminal write: "python -m pytest --cov"
# ---------------------------------------------------------------------------------------------------------------------------------


# TESTING THE "preprocess_text" FUNCTION:
@pytest.mark.parametrize("input_value, expected_output",
                            [
                                ("Hello, World!", "hello world"),
                                ("TESTING 123!!!", "testing 123"),
                                ("What's up?", "whats up"),
                                ("Hello!! How are you??", "hello how are you"),
                                ("No punctuation", "no punctuation"),
                                ("", ""),
                            ]
                        )
def test_preprocess_text(input_value: str, expected_output: str):
    """This function tests the preprocess_text function."""
    assert preprocess_text(input_value) == expected_output

# ---------------------------------------------------------------------------------------------------------------------------------

# TESTING THE "read_excel" FUNCTION:
@pytest.mark.parametrize("data, sheet_name, skiprows, nrows, expected_df",
                            [
                                # Test 1: Reading the default sheet (sheet 0) from a simple Excel file
                                (
                                        {"A": [1, 2, 3], "B": [4, 5, 6]},               # data to write to Excel
                                        0,                                              # sheet_name
                                        None,                                           # skiprows
                                        None,                                           # nrows
                                        pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})  # expected DataFrame
                                ),
                                # Test 2: Reading only the first 2 rows
                                (
                                        {"A": [10, 20, 30, 40], "B": [50, 60, 70, 80]},
                                        0,
                                        None,
                                        2,
                                        pd.DataFrame({"A": [10, 20], "B": [50, 60]})
                                ),
                            ]
                        )
def test_read_excel_valid(tmp_path, data, sheet_name, skiprows, nrows, expected_df):
    """This function tests the read_excel function."""
    # Creating a temporary Excel file
    file_path = tmp_path / "test.xlsx"
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)

    # Calling the function
    result_df = read_excel(str(file_path), sheet_name=sheet_name, skiprows=skiprows, nrows=nrows)

    # Using pandas testing function to compare DataFrames
    pd.testing.assert_frame_equal(result_df, expected_df)
    return None

def test_read_excel_file_not_found():
    """This function tests the read_excel function that it correctly raises the FileNotFoundError if the file does not exist."""
    non_existent_path = "non_existent_file.xlsx"
    with pytest.raises(FileNotFoundError):
        read_excel(non_existent_path)
    return None

def test_read_excel_invalid_sheet(tmp_path):
    """This function tests the read_excel function that it correctly raises the ValueError when an invalid sheet name is provided."""
    data = {"A": [1, 2, 3], "B": [4, 5, 6]}
    file_path = tmp_path / "test.xlsx"
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)

    with pytest.raises(ValueError):
        # Passing an invalid sheet name. The function should catch the ValueError from pd.read_excel and re-raise it as a ValueError
        read_excel(str(file_path), sheet_name="nonexistent")
    return None

# ---------------------------------------------------------------------------------------------------------------------------------

# TESTING THE "save_df_to_excel" FUNCTION:
@pytest.mark.parametrize("data, sheet_name",
                            [
                                ({"A": [1, 2, 3], "B": [4, 5, 6]}, "Sheet1"),
                                ({"X": [10, 20], "Y": [30, 40]}, "Data"),
                            ]
                        )
def test_save_df_to_excel_success(tmp_path, data, sheet_name):
    """This function tests the save_df_to_excel function."""
    # Creating a temporary file path for the Excel file
    file_path = tmp_path / "test.xlsx"
    df = pd.DataFrame(data)

    # Calling the function to save the DataFrame
    save_df_to_excel(str(file_path), df, sheet_name=sheet_name)

    # Reading back the file from the specified sheet
    df_read = pd.read_excel(str(file_path), sheet_name=sheet_name)

    # Testing
    pd.testing.assert_frame_equal(df, df_read)
    return None

def test_save_df_to_excel_invalid_df(tmp_path):
    """This function tests the save_df_to_excel function. That passing a non-DataFrame to save_df_to_excel raises a TypeError."""
    file_path = tmp_path / "test.xlsx"
    # Passing a list instead of a df
    with pytest.raises(TypeError, match="The 'df' parameter must be a pandas DataFrame."):
        save_df_to_excel(str(file_path), [1, 2, 3])
    return None

# ---------------------------------------------------------------------------------------------------------------------------------

# TESTING THE "aggregate_phrase_embedding" FUNCTION:

# Creating a dummy model to simulate gensim.models.KeyedVectors (only vector_size is needed for this function)
class DummyKeyedVectors:
    def __init__(self, vector_size: int):
        self.vector_size = vector_size

@pytest.mark.parametrize("embeddings, vector_size, expected",
                            [
                                # Test 1: Two tokens with orthogonal vectors
                                # Sum: [1, 0] + [0, 1] = [1, 1]. Norm = sqrt(1^2+1^2) = sqrt(2)
                                # Normalized vector: [1/sqrt(2), 1/sqrt(2)]
                                (
                                    {"word1": np.array([1, 0]), "word2": np.array([0, 1])},
                                    2,
                                    np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                                ),

                                # Test 2: One token only
                                # Vector: [3, 4]. Norm = 5. Normalized vector: [3/5, 4/5]
                                (
                                    {"a": np.array([3, 4])},
                                    2,
                                    np.array([3/5, 4/5]),
                                ),

                                # Test 3: Empty dictionary should return a zero vector
                                (
                                    {},
                                    3,
                                    np.zeros(3),
                                ),
                            ]
                        )

def test_aggregate_phrase_embedding(embeddings, vector_size, expected):
    """This function tests the aggregate_phrase_embedding function that it correctly aggregates token embeddings."""
    # Creating the dummy model
    dummy_model = DummyKeyedVectors(vector_size)
    result = aggregate_phrase_embedding(embeddings, dummy_model)

    # Comparing the result with the expected array, and allowing for minor floating point differences
    np.testing.assert_allclose(result, expected, err_msg="Aggregated embedding did not match expected output.")
    return None

# ---------------------------------------------------------------------------------------------------------------------------------

# TESTING THE "compute_distance_matrix" FUNCTION:
@pytest.mark.parametrize("embeddings, metric, expected",
                            [
                                # Test 1: Cosine metric with two orthonormal (normalized) 2D vectors
                                # For two unit vectors: [1, 0] and [0, 1],
                                # Cosine similarity = dot([[1,0],[0,1]], [[1,0],[0,1]]^T) = [[1,0],[0,1]]
                                # Cosine distance = 1 - similarity = [[0,1],[1,0]]
                                (
                                    np.array([[1, 0], [0, 1]]),
                                    "cosine",
                                    np.array([[0, 1], [1, 0]])
                                ),

                                # Test 2: Euclidean metric with two 2D vectors
                                # For vectors [1,2] and [4,6]:
                                # Euclidean distance = sqrt((4-1)^2 + (6-2)^2) = sqrt(9+16)=sqrt(25)=5
                                # So expected distance matrix is [[0, 5], [5, 0]]
                                (
                                    np.array([[1, 2], [4, 6]]),
                                    "euclidean",
                                    np.array([[0, 5], [5, 0]])
                                ),
                            ]
                        )
def test_compute_distance_matrix_valid(embeddings, metric, expected):
    """
    This function tests the compute_distance_matrix function that it returns the expected pairwise distance matrix
    for both cosine and euclidean metrics.
    """
    result = compute_distance_matrix(embeddings, metric=metric)
    # Using numpy's assert_allclose for floating point comparison
    np.testing.assert_allclose(result, expected, rtol=1e-5, err_msg=f"Distance matrix for metric '{metric}' did not match expected output.")

def test_compute_distance_matrix_invalid_metric():
    """This function tests the compute_distance_matrix function that it correctly raises a ValueError."""
    embeddings = np.array([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="Unsupported metric"):
        compute_distance_matrix(embeddings, metric="invalid")



















