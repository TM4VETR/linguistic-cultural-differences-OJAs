import pandas as pd
import sqlite3


# Assuming you have an SQLite DB file named 'jobs_data.db'
DATABASE_PATH = 'jobads_regional_try3.db'
TABLE_NAME = 'job_listings'



data_frame = {
   "url" : [],
   "title": [],
   "location" : [],
   "date":[],
   "company": [],
   "contractType" : [], # (regular position, apprenticeship, internship, etc.)
   "contractTerm":[],  # Full time / part time 
   "salary": [],
   "description":[],
   "country":[]
}

def saveData(data_frame):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data_frame)
    
    # Using a context manager to handle the database connection
    with sqlite3.connect(DATABASE_PATH) as conn:
        # Save the dataframe to SQLite table
        df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
    
    # After saving, you can clear the data_frame to free up memory
    data_frame.clear()
    # Reinitialize the dictionary structure after clearing
    for key in ['url', 'title', 'location', 'date', 'company', 'contractType', 'contractTerm', 'salary', 'description', 'country']:
        data_frame[key] = []



def initialize_database():
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            title TEXT,
            location TEXT,
            date TEXT,
            company TEXT,
            contractType TEXT,
            contractTerm TEXT,
            salary TEXT,
            description TEXT,
            country TEXT
        )''')
        conn.commit()