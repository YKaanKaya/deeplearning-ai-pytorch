import re
import pandas as pd



def remove_comments(code):
    # This regex pattern matches comments in the code
    pattern = r'#.*'
    
    # Use re.sub() to replace comments with an empty string
    code_without_comments = re.sub(pattern, '', code)
    
    # Split the code into lines, strip each line, and filter out empty lines
    lines = code_without_comments.splitlines()
    non_empty_lines = [line.rstrip() for line in lines if line.strip()]
    
    # Join the non-empty lines back into a single string
    return '\n'.join(non_empty_lines)



def load_rows(path_to_csv, row_range=(20, 29)):
    """
    Loads a CSV file and returns a specific range of rows.

    Args:
    path_to_csv (str): The file path to the CSV.
    row_range (tuple): A tuple of two integers (start, end) 
                         specifying the inclusive row range to load.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the specified rows.
    """
    # Unpack the start and end from the range tuple
    start_row, end_row = row_range

    # Load the entire CSV file into a DataFrame
    df = pd.read_csv(path_to_csv)

    # Select rows using .iloc for integer-location based indexing.
    # Add 1 to the end_row to make the slice inclusive.
    selected_df = df.iloc[start_row:end_row + 1]

    return selected_df