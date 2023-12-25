import pandas as pd

# Create a DataFrame with two columns for "Santa" files
base_name_santa = "Santa"
start_index_santa = 1
end_index_santa = 308
file_extension_santa = ".jpg"

file_names_santa = [f"{i}_{base_name_santa}{file_extension_santa}" for i in range(start_index_santa, end_index_santa + 1)]
df_santa = pd.DataFrame({"Column1": file_names_santa, "Column2": [1] * len(file_names_santa)})

# Create a DataFrame with two columns for "NotSanta" files
base_name_not_santa = "NotSanta"
start_index_not_santa = 1
end_index_not_santa = 308
file_extension_not_santa = ".jpg"

file_names_not_santa = [f"{i}_{base_name_not_santa}{file_extension_not_santa}" for i in range(start_index_not_santa, end_index_not_santa + 1)]
df_not_santa = pd.DataFrame({"Column1": file_names_not_santa, "Column2": [0] * len(file_names_not_santa)})

# Concatenate the two DataFrames
df_combined = pd.concat([df_santa, df_not_santa], ignore_index=True)

# Export the combined DataFrame to a CSS file
css_file_path = "/test_label.csv"
df_combined.to_csv(css_file_path, sep="\t", index=False, header=False)

print(f"Combined DataFrame exported to {css_file_path}")