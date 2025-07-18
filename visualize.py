import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(r"D:\GUVI_Second_Project\ODI_full_dataset.csv", low_memory=False)

# Just using venue_x directly
df = df.dropna(subset=['venue_x'])
venue_counts = df['venue_x'].value_counts().head(10)

# 1. Top venues (using seaborn this once)
sns.barplot(x=venue_counts.values, y=venue_counts.index, palette='muted')
plt.title("Top 10 ODI Venues by Matches")
plt.show()

# 2. Match wins - top 15 teams
winner_counts = df["winner"].value_counts().head(15)
winner_counts.plot(kind="bar", color='skyblue')
plt.title("Top Teams by Wins")
plt.xticks(rotation=45)
plt.show()

# 3. Toss winners
toss_counts = df["toss_winner"].value_counts().head(10)
plt.figure(figsize=(10,5))
toss_counts.plot(kind="bar", color="orange")
plt.title("Top Toss Winners")
plt.xticks(rotation=30)
plt.show()

# 4. Toss decision (just simple pie)
df["toss_decision"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90)
plt.title("Toss Decision Breakdown")
plt.ylabel("")
plt.show()

# 5. Match winners barh
df["winner"].value_counts().head(10).plot(kind="barh", color="purple")
plt.title("Top 10 Match Winners")
plt.gca().invert_yaxis()
plt.show()

# 6. Matches per season (just assuming season_x here)
season_counts = df["season_x"].astype(str).value_counts().sort_index()
season_counts.plot(kind="line", marker="o")
plt.title("Matches per Season")
plt.xticks(rotation=60)
plt.show()

# 7. Runs per ball
df["runs_off_bat"].hist(bins=range(0, 8), color='gray', rwidth=0.9)
plt.title("Runs per Ball")
plt.xlabel("Runs")
plt.show()

# 8. Extras per ball
df["extras"].hist(bins=range(0, 6), color="red")
plt.title("Extras Distribution")
plt.show()

# 9. Wickets per match (quick and dirty)
wicket_counts = df.groupby("match_id")["wicket_type"].apply(lambda x: x.notna().sum())
wicket_counts.hist(bins=range(0,15), color='green')
plt.title("Wickets per Match")
plt.show()

# 10. Toss winner vs match winner
df["toss_winner_won"] = (df["toss_winner"] == df["winner"]).astype(int)
result = df.groupby("toss_decision")["toss_winner_won"].mean()
print(result)  # debug print
result.plot(kind="bar", color="brown")
plt.title("Toss Win → Match Win Probability")
plt.show()

