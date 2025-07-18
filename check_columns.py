import sqlite3

conn = sqlite3.connect("D:/GUVI_Second_Project/cricket_data.db")
cursor = conn.cursor()

tables = ['odi_matches', 't20_matches', 'test_matches']

for table in tables:
    print(f"\nColumns in table {table}:")
    cursor.execute(f"PRAGMA table_info({table})")
    for col in cursor.fetchall():
        print(col[1])

conn.close()
