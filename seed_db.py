import sqlite3
from pathlib import Path

# Create the data directory if it doesn't exist
db_path = Path("./data/startup_metrics.db")
db_path.parent.mkdir(exist_ok=True)

# Connect and create the schema exactly as the prompt expects
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS startup_metrics (
    startup_name TEXT,
    revenue_inr REAL,
    burn_rate_inr REAL,
    headcount INTEGER,
    foreign_remittance_usd REAL,
    cash_reserve_ratio REAL,
    month TEXT
)
''')

# Clear old data and insert a fresh mock startup record
cursor.execute('DELETE FROM startup_metrics')
cursor.execute('''
INSERT INTO startup_metrics VALUES
('BharatPay Solutions', 8500000, 3200000, 55, 12000, 0.20, '2025-03')
''')

conn.commit()
conn.close()

print(f"✅ Database successfully seeded at: {db_path.absolute()}")