import sqlite3

# Check trn_inventory table structure
conn = sqlite3.connect("tally.db")
cursor = conn.cursor()

print("=== trn_inventory table structure ===")
cursor.execute("PRAGMA table_info(trn_inventory);")
columns = cursor.fetchall()
for col in columns:
    print(f"{col[1]} ({col[2]})")

print("\n=== Sample data from trn_inventory ===")
cursor.execute("SELECT * FROM trn_inventory LIMIT 5;")
rows = cursor.fetchall()
col_names = [description[0] for description in cursor.description]
print("Columns:", col_names)

for row in rows:
    print(row)

conn.close()
