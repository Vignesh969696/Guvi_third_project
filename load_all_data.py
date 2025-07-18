import sqlite3
import pandas as pd
import os

# Dataset Paths
odi_path = r"D:\GUVI_Second_Project\ODI_full_dataset.csv"
t20_path = r"D:\GUVI_Second_Project\T20_Match\T20_combined_data.csv"
test_path = r"D:\GUVI_Second_Project\Test_Match\Test_full_data.csv"

# SQLite database file
sqlite_db = r"D:\GUVI_Second_Project\cricket_data.db"

# Connecting to SQLite
conn = sqlite3.connect(sqlite_db)
cursor = conn.cursor()

# Loading and inserting ODI data
print("Loading ODI data...")
odi_df = pd.read_csv(odi_path)
odi_df.to_sql("odi_matches", conn, if_exists="replace", index=False)
print("ODI data loaded into 'odi_matches' table.")

# Loading and inserting T20 data
print("Loading T20 data...")
t20_df = pd.read_csv(t20_path)
t20_df.to_sql("t20_matches", conn, if_exists="replace", index=False)
print("T20 data loaded into 't20_matches' table.")

# Loading and inserting Test data
print("Loading Test data...")
test_df = pd.read_csv(test_path)
test_df.to_sql("test_matches", conn, if_exists="replace", index=False)
print("Test data loaded into 'test_matches' table.")

# Verifying table creation
print("\nTables in the database:")
tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
cursor.execute(tables_query)
tables = cursor.fetchall()
for table in tables:
    print(f"- {table[0]}")

# Closing the connection
conn.close()
print("\nAll data loaded into SQLite database successfully.")
