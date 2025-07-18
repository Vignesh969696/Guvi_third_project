import os
import json
import pandas as pd


json_dir_path = "D:\\GUVI_Second_Project\\Test_Match\\tests_json"  

matches = []
deliveries = []

for file_name in os.listdir(json_dir_path):
    if not file_name.endswith(".json"):
        continue

    file_path = os.path.join(json_dir_path, file_name)
    match_id = file_name.replace(".json", "")
    
    with open(file_path, "r") as f:
        data = json.load(f)

    info = data.get("info", {})
    if info.get("match_type") != "Test":
        continue

    venue = info.get("venue", "Unknown")
    teams = info.get("teams", ["Unknown", "Unknown"])
    toss = info.get("toss", {})
    outcome = info.get("outcome", {})

    matches.append({
        "match_id": match_id,
        "venue": venue,
        "team_1": teams[0],
        "team_2": teams[1] if len(teams) > 1 else "Unknown",
        "match_type": info.get("match_type"),
        "toss_winner": toss.get("winner", "Unknown"),
        "toss_decision": toss.get("decision", "Unknown"),
        "match_winner": outcome.get("winner", "Unknown")
    })

    for innings in data.get("innings", []):
        batting_team = innings.get("team", "Unknown")
        for over in innings.get("overs", []):
            over_no = int(over.get("over", 0))
            for ball in over.get("deliveries", []):
                ball_no = int(ball.get("ball", 0))
                run_info = ball.get("runs", {})

                deliveries.append({
                    "match_id": match_id,
                    "batting_team": batting_team,
                    "over_number": over_no,
                    "ball_number": ball_no,
                    "batsman": ball.get("batter", "Unknown"),
                    "bowler": ball.get("bowler", "Unknown"),
                    "runs": run_info.get("batter", 0),
                    "extras": run_info.get("extras", 0),
                    "total_runs": run_info.get("total", 0),
                    "wicket": 1 if "wickets" in ball else 0
                })

# Saving to CSVs
output_folder = "D:\\GUVI_Second_Project\\Test_Match"
os.makedirs(output_folder, exist_ok=True)

matches_df = pd.DataFrame(matches)
deliveries_df = pd.DataFrame(deliveries)

matches_df.to_csv(os.path.join(output_folder, "test_matches_clean.csv"), index=False)
deliveries_df.to_csv(os.path.join(output_folder, "test_deliveries_clean.csv"), index=False)

print("Test match JSON files processed.")

# Merging and saving final combined file
df_merged = pd.merge(matches_df, deliveries_df, on="match_id")
df_merged.to_csv(os.path.join(output_folder, "Test_full_data.csv"), index=False)

print("Test full dataset saved.")
