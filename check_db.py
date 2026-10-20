import sqlite3

db_path = "data/inventory.db"

connection = sqlite3.connect(db_path)
cursor = connection.cursor()

cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
)

tables = cursor.fetchall()

print("Database tables:", tables)

for (table_name,) in tables:
    print(f"\nTable: {table_name}")
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    print("Columns:", cursor.fetchall())

connection.close()
