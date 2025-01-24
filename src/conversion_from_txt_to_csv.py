# import pandas as pd
# import os

# def convert_txt_to_csv(txt_directory, csv_directory, delimiter='\t'):
#     """
#     Convert .txt files in a directory to .csv files with the same names.
    
#     Parameters:
#     - txt_directory: Directory containing .txt files.
#     - csv_directory: Directory where .csv files will be saved.
#     - delimiter: Delimiter used in .txt files (default is tab).
#     """
#     # Ensure the CSV directory exists
#     os.makedirs(csv_directory, exist_ok=True)

#     # Iterate through each file in the directory
#     for filename in os.listdir(txt_directory):
#         if filename.endswith('.txt'):
#             # Define full file path
#             txt_path = os.path.join(txt_directory, filename)
            
#             # Read the .txt file into a DataFrame
#             df = pd.read_csv(txt_path, delimiter=delimiter, header=None)
            
#             # Define CSV file path with the same name
#             csv_filename = filename.replace('.txt', '.csv')
#             csv_path = os.path.join(csv_directory, csv_filename)
            
#             # Save the DataFrame as a .csv file
#             df.to_csv(csv_path, index=False, header=False)
#             print(f"Converted {txt_path} to {csv_path}")

# # Define directories
# person_txt_directory = '/Users/shivakumarbiru/Desktop/individual_project/rfc/dataset/dataset_csv/person'
# unoccupied_txt_directory = '/Users/shivakumarbiru/Desktop/individual_project/rfc/dataset/dataset_csv/unocupied'

# person_csv_directory = '/Users/shivakumarbiru/Desktop/individual_project/rfc/dataset/dataset_csv/person_csv'
# unoccupied_csv_directory = '/Users/shivakumarbiru/Desktop/individual_project/rfc/dataset/dataset_csv/unocupied_csv'

# # Convert .txt files to .csv files
# convert_txt_to_csv(person_txt_directory, person_csv_directory, delimiter='\t')
# convert_txt_to_csv(unoccupied_txt_directory, unoccupied_csv_directory, delimiter='\t')


# import pandas as pd
# import os

# def convert_txt_to_csv(txt_directory, csv_directory, delimiter='\t'):
#     # Ensure the CSV directory exists
#     os.makedirs(csv_directory, exist_ok=True)

#     # Iterate through each file in the directory
#     for filename in os.listdir(txt_directory):
#         if filename.endswith('.txt'):
#             # Define full file path
#             txt_path = os.path.join(txt_directory, filename)
            
#             # Read the .txt file into a DataFrame
#             df = pd.read_csv(txt_path, delimiter=delimiter, header=None)
            
#             # Define CSV file path with the same name
#             csv_filename = filename.replace('.txt', '.csv')
#             csv_path = os.path.join(csv_directory, csv_filename)
            
#             # Save the DataFrame as a .csv file
#             df.to_csv(csv_path, index=False, header=False)
#             print(f"Converted {txt_path} to {csv_path}")

# # Define directories
# person_txt_directory = '/Users/shivakumarbiru/Desktop/individual_project/rfc/dataset/test_data'


# person_csv_directory = '/Users/shivakumarbiru/Desktop/individual_project/rfc/dataset/test_data'


import os
import glob

def list_files(directory):
    # Use glob to get all files in the directory
    files = glob.glob(os.path.join(directory, '**'), recursive=True)
    
    # Filter out directories, keep only files
    files = [f for f in files if os.path.isfile(f)]
    
    return files

# Example usage
directory_path = '/Users/shivakumarbiru/Desktop/individual_project/rfc/dataset/test_data/person_test'
files = list_files(directory_path)

# Print files in a list format
print(files)
