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









# ---------------------------------------------------------------------------------------------------------------------------------
