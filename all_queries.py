import sqlite3

conn = sqlite3.connect("D:/GUVI_Second_Project/cricket_data.db")
cursor = conn.cursor()

# Helper function
def run_query(title, query):
    print(f"{title}")
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        for row in results:
            print(row)
    except Exception as e:
        print("Error:", e)
    print("-" * 50)

print("\nODI QUERIES")
run_query("1. Top 5 venues", "SELECT venue_y, COUNT(*) FROM odi_matches GROUP BY venue_y ORDER BY COUNT(*) DESC LIMIT 5;")
run_query("2. Top 5 teams by wins", "SELECT winner, COUNT(*) FROM odi_matches WHERE winner IS NOT NULL GROUP BY winner ORDER BY COUNT(*) DESC LIMIT 5;")
run_query("3. Toss decisions", "SELECT toss_decision, COUNT(*) FROM odi_matches GROUP BY toss_decision;")
run_query("4. Player of the match", "SELECT player_of_match, COUNT(*) FROM odi_matches GROUP BY player_of_match ORDER BY COUNT(*) DESC LIMIT 5;")
run_query("5. Toss winner being match winner", "SELECT COUNT(*) FROM odi_matches WHERE toss_winner = winner;")
run_query("6. Match outcome distribution", "SELECT outcome, COUNT(*) FROM odi_matches GROUP BY outcome;")
run_query("7. Top 5 cities", "SELECT city, COUNT(*) FROM odi_matches GROUP BY city ORDER BY COUNT(*) DESC LIMIT 5;")


print("\nT20 QUERIES")
run_query("1. Top 5 venues", "SELECT venue, COUNT(*) FROM t20_matches GROUP BY venue ORDER BY COUNT(*) DESC LIMIT 5;")
run_query("2. Top 5 teams by wins", "SELECT match_winner, COUNT(*) FROM t20_matches WHERE match_winner IS NOT NULL GROUP BY match_winner ORDER BY COUNT(*) DESC LIMIT 5;")
run_query("3. Toss decision counts", "SELECT toss_decision, COUNT(*) FROM t20_matches GROUP BY toss_decision;")
run_query("4. No result matches", "SELECT COUNT(*) FROM t20_matches WHERE match_winner IS NULL;")
run_query("5. Top 5 batting teams", "SELECT batting_team, COUNT(*) FROM t20_matches GROUP BY batting_team ORDER BY COUNT(*) DESC LIMIT 5;")
run_query("6. Average runs per ball", "SELECT AVG(runs) FROM t20_matches;")
run_query("7. Powerplay deliveries", "SELECT COUNT(*) FROM t20_matches WHERE powerplay = 1;")


print("\nTEST QUERIES")
run_query("1. Top 5 venues", "SELECT venue, COUNT(*) FROM test_matches GROUP BY venue ORDER BY COUNT(*) DESC LIMIT 5;")
run_query("2. Top 5 teams by wins", "SELECT match_winner, COUNT(*) FROM test_matches WHERE match_winner IS NOT NULL GROUP BY match_winner ORDER BY COUNT(*) DESC LIMIT 5;")
run_query("3. Toss decisions", "SELECT toss_decision, COUNT(*) FROM test_matches GROUP BY toss_decision;")
run_query("4. Toss winner also being match winner", "SELECT COUNT(*) FROM test_matches WHERE toss_winner = match_winner;")
run_query("5. Top 5 batting teams", "SELECT batting_team, COUNT(*) FROM test_matches GROUP BY batting_team ORDER BY COUNT(*) DESC LIMIT 5;")
run_query("6. Average runs per ball", "SELECT AVG(runs) FROM test_matches;")

conn.close()
