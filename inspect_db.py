import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join("data", "nutrition.db")
USER_ID = "3012d790-122f-4ec3-a781-548ad9dd5320"
HALLUCINATED_DATE = "2024-01-26"
TODAY = "2026-01-27"

def dump_db():
    if not os.path.exists(DB_PATH):
        output = f"DB not found at {DB_PATH}\n"
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        import io
        f = io.StringIO()
        
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        log(f"Current Time: {datetime.now().isoformat()}")
        check_date_robust(cursor, HALLUCINATED_DATE, log)
        check_date_robust(cursor, TODAY, log)
        
        # Also check ANY data
        cursor.execute("SELECT COUNT(*) FROM meal_logs")
        log(f"\nTotal meal logs in DB: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(*) FROM daily_logs")
        log(f"\nTotal daily logs in DB: {cursor.fetchone()[0]}")

        conn.close()
        output = f.getvalue()

    with open("db_inspect_output.txt", "w") as out_file:
        out_file.write(output)

def check_date_robust(cursor, date, log):
    log(f"\n--- Records for User {USER_ID} on {date} ---")
    cursor.execute("SELECT * FROM daily_logs WHERE user_id = ? AND date = ?", (USER_ID, date))
    row = cursor.fetchone()
    if row:
        log("[daily_logs]:")
        for k in row.keys():
            log(f"  {k}: {row[k]}")
    else:
        log("[daily_logs]: No record found.")

    cursor.execute("SELECT * FROM meal_logs WHERE user_id = ? AND timestamp LIKE ?", (USER_ID, f"{date}%"))
    rows = cursor.fetchall()
    log(f"[meal_logs] ({len(rows)} entries):")
    for row in rows:
        log(f"  - {row['food_name']} (TS: {row['timestamp']})")

if __name__ == "__main__":
    dump_db()
