# IMPORTS:
import pytest
import pandas as pd
from function_definitions import *
# ---------------------------------------------------------------------------------------------------------------------------------
# IMPORTANT: If using pytest, all tests must start with "test_"
# To run the tests, locate to the directory of the project and in the terminal write: "python -m pytest"
# To create a coverage report, in the terminal write: "python -m pytest --cov"
# ---------------------------------------------------------------------------------------------------------------------------------


# TESTING THE "XXX" FUNCTION:
@pytest.mark.parametrize("input_value, expected_output",
                            [
                                (X, Y),
                            ]
                         )
def test_XXX(input_value: XXX, expected_output: XXX):
    """This function tests the "XXX" function."""
    assert XXX(input_value) == expected_output

