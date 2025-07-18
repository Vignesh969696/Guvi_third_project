import os
import json
import pandas as pd

json_folder = "D:\\GUVI_Second_project\\T20_Match\\t20s_json"

match_info_list = []
delivery_info_list = []

for json_file in os.listdir(json_folder):
    if not json_file.endswith(".json"):
        continue
    
    file_path = os.path.join(json_folder, json_file)
    match_id = json_file.replace(".json", "")
    
    with open(file_path, "r") as f:
        match_data = json.load(f)

    info = match_data.get("info", {})
    venue = info.get("venue", "Unknown")
    teams = info.get("teams", ["Unknown", "Unknown"])
    match_type = info.get("match_type", "Unknown")
    toss = info.get("toss", {})
    outcome = info.get("outcome", {})

    match_info_list.append({
        "match_id": match_id,
        "venue": venue,
        "team_1": teams[0],
        "team_2": teams[1] if len(teams) > 1 else "Unknown",
        "match_type": match_type,
        "toss_winner": toss.get("winner", "Unknown"),
        "toss_decision": toss.get("decision", "Unknown"),
        "match_winner": outcome.get("winner", "Unknown")
    })

    powerplay_overs = set()
    for inning in match_data.get("innings", []):
        batting_team = inning.get("team", "Unknown")

        for pp in inning.get("powerplays", []):
            start = int(pp.get("from", 0))
            end = int(pp.get("to", 0))
            powerplay_overs.update(range(start, end + 1))

        for over in inning.get("overs", []):
            over_number = int(over.get("over", 0))
            for delivery in over.get("deliveries", []):
                ball_number = int(delivery.get("ball", 0))
                runs = delivery.get("runs", {})

                delivery_info_list.append({
                    "match_id": match_id,
                    "batting_team": batting_team,
                    "over_number": over_number,
                    "ball_number": ball_number,
                    "batter": delivery.get("batter", "Unknown"),
                    "bowler": delivery.get("bowler", "Unknown"),
                    "runs": runs.get("batter", 0),
                    "extras": runs.get("extras", 0),
                    "total_runs": runs.get("total", 0),
                    "wicket": 1 if "wickets" in delivery else 0,
                    "powerplay": 1 if over_number in powerplay_overs else 0
                })

# Creating DataFrames
df_matches = pd.DataFrame(match_info_list)
df_deliveries = pd.DataFrame(delivery_info_list)

# Defining output paths
output_dir = "D:\\GUVI_Second_Project\\T20_Match"
os.makedirs(output_dir, exist_ok=True)

df_matches.to_csv(os.path.join(output_dir, "t20_matches_clean.csv"), index=False)
df_deliveries.to_csv(os.path.join(output_dir, "t20_deliveries_clean.csv"), index=False)

print("T20 JSON files processed successfully!")

# Merging for combined dataset
df_combined = pd.merge(df_matches, df_deliveries, on="match_id")
df_combined.to_csv(os.path.join(output_dir, "T20_combined_data.csv"), index=False)

print("Combined T20 dataset saved.")
