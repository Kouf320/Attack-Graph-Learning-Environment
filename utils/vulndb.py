import sqlite3
import os

def create_vulnerability_db(db_path='vulnerability.db'):
    """
    Creates the SQLite database and schema required by the GraphEnvironment.
    """
    if os.path.exists(db_path):
        print(f"Warning: '{db_path}' already exists.")
        user_input = input("Do you want to delete it and start fresh? (y/n): ")
        if user_input.lower() == 'y':
            os.remove(db_path)
            print(f"Deleted old '{db_path}'.")
        else:
            print("Operation cancelled.")
            return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Schema matches the query: SELECT cvss_string FROM vulnerability WHERE cve = ?
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS vulnerability (
            cve TEXT PRIMARY KEY,
            cvss_string TEXT NOT NULL,
            base_score REAL,
            severity TEXT,
            description TEXT
        );
        '''
        cursor.execute(create_table_query)
        
        # Create an index on the CVE column for faster lookups during graph initialization
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cve ON vulnerability(cve);')

        print(f"Successfully created database at: {db_path}")
        print("Schema initialized: Table 'vulnerability' with columns [cve, cvss_string, ...]")
        
        # --- Optional: Insert Dummy Data for Testing ---
        # cursor.execute("INSERT OR IGNORE INTO vulnerability (cve, cvss_string) VALUES (?, ?)", 
        #                ('CVE-2021-44228', 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H'))
        # conn.commit()
        # print("Inserted test record: CVE-2021-44228")

        conn.commit()
        conn.close()

    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}")

if __name__ == "__main__":
    create_vulnerability_db()
