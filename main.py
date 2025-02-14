# IMPORTS:
from function_definitions import *
# -----------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # 1) SETUP:
    set_pandas_output()  # This just improves the Pandas output in the PyCharm terminal
    user_name = get_local_username()
    input_file_name = "phrases"
    input_file_path = f"C:/Users/{user_name}/PycharmProjects/Galytix/Files/{input_file_name}.xlsx"

    # 2)
    df_raw = read_excel(input_file_path)
    print(df_raw)

    # 3)



