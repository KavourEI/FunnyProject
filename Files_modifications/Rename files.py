import os

def rename_files(folder_path):
    # Change to the folder
    os.chdir(folder_path)

    # Get a list of all files in the folder
    files = [file for file in os.listdir() if file.endswith('.Santa.jpg')]

    # Sort the files based on their current names
    files.sort(key=lambda x: int(x.split('.')[0]))

    # Rename the files in ascending order
    for index, file in enumerate(files, start=1):
        # Get the file extension
        _, extension = os.path.splitext(file)

        # Create the new filename
        new_name = f"{index}_Santa{extension}"

        # Rename the file
        os.rename(file, new_name)

    print("Files have been renamed and ordered.")

# Set the path to your folder containing the files
folder_path = "/test/santa"

# Call the function to rename files
rename_files(folder_path)

