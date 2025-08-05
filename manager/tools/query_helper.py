import sqlite3
import google.generativeai as genai
from typing import Dict, List, Tuple

class DatabaseSchema:
    def __init__(self, db_path: str = r"C:\Users\manoj\tallydb.db"):
        self.db_path = db_path
        self.tables: Dict[str, List[Tuple[str, str]]] = {}
        self._load_schema()

    def _load_schema(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            self.tables[table] = [(col[1], col[2]) for col in columns]
        conn.close()

    def get_table_columns(self, table_name: str) -> List[str]:
        return [col[0] for col in self.tables.get(table_name, [])]

    def get_column_type(self, table_name: str, column_name: str) -> str:
        for col_name, col_type in self.tables.get(table_name, []):
            if col_name == column_name:
                return col_type
        return ""

def get_sqlite_schema(db_path: str = r"C:\Users\manoj\tallydb.db") -> str:
    schema = DatabaseSchema(db_path)
    schema_str = []
    for table, columns in schema.tables.items():
        schema_str.append(f"-- Table: {table}")
        for col_name, col_type in columns:
            schema_str.append(f"{col_name} {col_type}")
        schema_str.append("")
    return '\n'.join(schema_str)

def validate_query(sql: str, db_path: str = r"C:\Users\manoj\tallydb.db") -> str:
    """
    Validate if a SQL query is syntactically correct and return detailed error information.
    """
    import sqlite3  # Use direct sqlite3 for validation, avoid importing run_query
    try:
        print(f"[validate_query] Attempting to validate query: {sql}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = sql.strip()
        if not query.lower().startswith('select'):
            return "Error: Only SELECT queries are supported"
        if "limit" not in query.lower():
            query = f"{query} LIMIT 1"
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"[validate_query] Query executed successfully, got {len(results)} results")
        conn.close()
        return "valid"
    except sqlite3.Error as e:
        print(f"[validate_query] SQLite error: {str(e)}")
        return f"Error: SQLite error - {str(e)}"
    except Exception as e:
        print(f"[validate_query] Unexpected error: {str(e)}")
        return f"Error: Unexpected error - {str(e)}"

def suggest_query(intent: str, agent: str = None, api_key: str = None) -> str:
    schema = DatabaseSchema()
    if api_key is None:
        import os
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY not set."
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    schema_str = get_sqlite_schema()
    tables = list(schema.tables.keys())
    prompt = (
        "You are an expert SQLite query generator.\n"
        "Here is the database schema:\n\n"
        f"{schema_str}\n\n"
        "Tables in the database:\n"
        f"{', '.join(tables)}\n\n"
        "Instructions:\n"
        "1. Only use tables and columns that exist in the schema above\n"
        "2. Always use proper SQL syntax\n"
        "3. Include proper WHERE clauses for date ranges when needed\n"
        "4. Use appropriate JOINs when combining tables\n"
        "5. Format the query properly with proper indentation\n\n"
        "User request:\n"
        f"{intent}\n\n"
        "Generate a single, valid SQLite query that answers the user's request.\n"
        "Output ONLY the SQL query - no explanations or code blocks.\n"
        "If you cannot generate a valid query, output 'Error: Cannot generate query'."
    )
    try:
        response = model.generate_content(prompt)
        sql = response.text.strip()
        if sql.startswith('```'):
            lines = sql.split('\n')
            sql = '\n'.join([line for line in lines if not line.startswith('```')])
            sql = sql.strip()
        elif sql.startswith('`'):
            sql = sql[1:].strip()
            if sql.endswith('`'):
                sql = sql[:-1].strip()
        sql = '\n'.join(line for line in sql.split('\n') if line.strip())
        if not sql or sql.lower().startswith('error'):
            return "Error: No valid query generated"
        print(f"[suggest_query] Cleaned SQL query: {sql}")
        validation_result = validate_query(sql)
        if validation_result != "valid":
            print(f"[suggest_query] Validation failed: {validation_result}")
            return f"Error: {validation_result}"
        return sql
    except Exception as e:
        return f"Error: {str(e)}"

def construct_query(table_name, columns, conditions=None):
    query = f"SELECT {', '.join(columns)} FROM {table_name}"
    if conditions:
        query += f" WHERE {' AND '.join(conditions)}"
    return query

def execute_query(connection, query):
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    return results

def insert_query(table_name, data):
    columns = ', '.join(data.keys())
    placeholders = ', '.join(['%s'] * len(data))
    query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    return query, list(data.values())

def update_query(table_name, data, conditions):
    set_clause = ', '.join([f"{key} = %s" for key in data.keys()])
    query = f"UPDATE {table_name} SET {set_clause} WHERE {' AND '.join(conditions)}"
    return query, list(data.values())

def delete_query(table_name, conditions):
    query = f"DELETE FROM {table_name} WHERE {' AND '.join(conditions)}"
    return query

def get_table_names(db_path: str = r"C:\Users\manoj\tallydb.db"):
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]

def get_columns(table_name, db_path: str = r"C:\Users\manoj\tallydb.db"):
    query = f"PRAGMA table_info({table_name});"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[1] for row in rows]