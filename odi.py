import os
import pandas as pd


csv_dir = "D:\\GUVI_Second_Project\\ODI_Match\\odis_csv2"


# Parsing match info files (_info.csv)


match_summaries = []

for file in os.listdir(csv_dir):
    if not file.endswith("_info.csv"):
        continue

    match_id = file.replace("_info.csv", "")
    file_path = os.path.join(csv_dir, file)
    
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',', 2)  # splits into max 3 parts
            if len(parts) == 3 and parts[0] == 'info':
                key = parts[1].strip()
                value = parts[2].strip().strip('"')  # removes quotes if present
                data[key] = value

    data['match_id'] = match_id
    match_summaries.append(data)

df_matches = pd.DataFrame(match_summaries)

# Parsing ball-by-ball CSV files (without _info.csv)


ball_data_frames = []

for file in os.listdir(csv_dir):
    if file.endswith("_info.csv") or not file.endswith(".csv"):
        continue

    file_path = os.path.join(csv_dir, file)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue
    
    # Adding match_id column for merging later
    match_id = file.replace(".csv", "")
    df['match_id'] = match_id

    ball_data_frames.append(df)

df_balls = pd.concat(ball_data_frames, ignore_index=True)


# Merging both DataFrames on match_id


# Some columns in match info does not exist for all matches, so fill the missing with "N/A"
df_matches.fillna("N/A", inplace=True)
merged_df = pd.merge(df_balls, df_matches, on='match_id', how='left')

# Saving merged full dataset
merged_df.to_csv("ODI_full_dataset.csv", index=False)

print("ODI data processing complete. Output saved as ODI_full_dataset.csv")
