import sqlite3
import google.generativeai as genai
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseSchema:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Get the path relative to the project root
            current_dir = os.path.dirname(__file__)  # tools directory
            project_root = os.path.join(current_dir, "..", "..")  # multiagents directory
            self.db_path = os.path.join(project_root, "tally.db")
        else:
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

def get_default_db_path():
    """Get the default database path relative to project root"""
    current_dir = os.path.dirname(__file__)  # tools directory
    project_root = os.path.join(current_dir, "..", "..")  # multiagents directory
    return os.path.join(project_root, "tally.db")

def get_sqlite_schema(db_path: str = None) -> str:
    if db_path is None:
        db_path = get_default_db_path()
    schema = DatabaseSchema(db_path)
    schema_str = []
    for table, columns in schema.tables.items():
        schema_str.append(f"-- Table: {table}")
        for col_name, col_type in columns:
            schema_str.append(f"{col_name} {col_type}")
        schema_str.append("")
    return '\n'.join(schema_str)

def validate_query(sql: str, db_path: str = None) -> str:
    """
    Validate if a SQL query is syntactically correct and return detailed error information.
    """
    if db_path is None:
        db_path = get_default_db_path()
    
    try:
        print(f"[validate_query] Query repr: {repr(sql)}")
        print(f"[validate_query] Query length: {len(sql)}")
        print(f"[validate_query] Contains semicolon: {';' in sql}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = sql.strip()
        
        # Remove any trailing semicolon
        if query.endswith(';'):
            query = query[:-1]
            
        if not query.lower().startswith('select'):
            return "Error: Only SELECT queries are supported"
            
        # Execute the query with a limit to test it
        if "limit" not in query.lower():
            test_query = f"{query} LIMIT 1"
        else:
            test_query = query
            
        cursor.execute(test_query)
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
        
        # Clean up the SQL query
        if sql.startswith('```'):
            lines = sql.split('\n')
            sql = '\n'.join([line for line in lines if not line.startswith('```')])
            sql = sql.strip()
        elif sql.startswith('`'):
            sql = sql[1:].strip()
            if sql.endswith('`'):
                sql = sql[:-1].strip()
        
        # Remove empty lines and join into a single statement
        sql_lines = [line.strip() for line in sql.split('\n') if line.strip()]
        sql = ' '.join(sql_lines)
        
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

def get_table_names(db_path: str = None):
    if db_path is None:
        db_path = get_default_db_path()
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]

def get_columns(table_name, db_path: str = None):
    if db_path is None:
        db_path = get_default_db_path()
    query = f"PRAGMA table_info({table_name});"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[1] for row in rows]

def get_sqlite_schema():
    """Return schema information based on actual database structure"""
    return """
    ACTUAL TALLY DATABASE SCHEMA:
    
    Key Tables and Columns:
    
    trn_accounting (Accounting Entries):
    - guid (TEXT) - Transaction ID
    - ledger (TEXT) - Account/Ledger name  
    - amount (FLOAT) - Transaction amount
    - amount_forex (FLOAT) - Foreign amount
    - currency (TEXT) - Currency symbol
    
    trn_voucher (Voucher Headers):
    - guid (TEXT) - Voucher ID
    - date (DATE) - Transaction date (YYYY-MM-DD format)
    - voucher_type (TEXT) - Type like 'GST Sales'
    - voucher_number (TEXT) - Document number
    - party_name (TEXT) - Customer/Supplier name
    - place_of_supply (TEXT) - Location
    
    mst_ledger (Chart of Accounts):
    - guid (TEXT) - Ledger ID
    - name (TEXT) - Account name
    - parent (TEXT) - Account group (Cash-in-hand, Bank Accounts, etc.)
    - opening_balance (FLOAT) - Opening balance
    - is_revenue (BIGINT) - Revenue account flag
    
    trn_bank (Bank Transactions):
    - guid (TEXT) - Transaction ID
    - ledger (TEXT) - Bank account name
    - transaction_type (TEXT) - Type like 'Cheque/DD'
    - instrument_date (DATE) - Transaction date
    - amount (FLOAT) - Amount
    
    trn_inventory (Inventory Transactions):
    - guid (TEXT) - Transaction ID
    - item (TEXT) - Product name
    - quantity (FLOAT) - Quantity (negative = outward/sales)
    - rate (FLOAT) - Unit rate
    - amount (FLOAT) - Total value
    - additional_amount (FLOAT) - Extra charges
    - discount_amount (FLOAT) - Discount amount
    - godown (TEXT) - Warehouse/location name
    - tracking_number (TEXT) - Tracking reference
    - order_number (TEXT) - Order reference
    - order_duedate (TEXT) - Due date
    
    Common GST Ledger Names:
    - 'Output CGST @ 09%'
    - 'Output SGST @ 09%' 
    - 'Output IGST @ 18%'
    - 'GST Sales @ 18%'
    
    Date Format: All dates are in 'YYYY-MM-DD' format
    Join Pattern: Use guid to join trn_accounting with trn_voucher
    """