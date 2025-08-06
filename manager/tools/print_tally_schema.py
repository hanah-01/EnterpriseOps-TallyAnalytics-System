import sqlite3
import json

def get_actual_table_structure(db_path="tally.db"):
    """Get the actual structure of all tables with sample data"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get all table names
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]
    
    schema = {}
    for table in tables:
        try:
            # Get column info
            cur.execute(f"PRAGMA table_info({table});")
            columns = cur.fetchall()
            
            # Get row count
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cur.fetchone()[0]
            
            # Get sample data (only if table has data)
            sample_data = []
            if row_count > 0:
                cur.execute(f"SELECT * FROM {table} LIMIT 3")
                sample_data = cur.fetchall()
            
            schema[table] = {
                "columns": [{"name": col[1], "type": col[2], "pk": col[5]} for col in columns],
                "row_count": row_count,
                "sample_data": sample_data[:2] if sample_data else []  # Just first 2 rows
            }
            
        except Exception as e:
            schema[table] = {"error": str(e)}
    
    conn.close()
    return schema

def print_relevant_tables(schema):
    """Print only tables that are relevant for financial analysis"""
    relevant_tables = [
        'trn_accounting', 'trn_voucher', 'trn_bank', 'trn_inventory',
        'mst_ledger', 'mst_stock_item', 'mst_cost_centre'
    ]
    
    print("RELEVANT TABLES FOR FINANCIAL ANALYSIS:")
    print("="*70)
    
    for table_name in relevant_tables:
        if table_name in schema and schema[table_name].get("row_count", 0) > 0:
            info = schema[table_name]
            print(f"\nTABLE: {table_name} ({info['row_count']} rows)")
            print("Columns:")
            for col in info["columns"]:
                pk_indicator = " (PK)" if col["pk"] else ""
                print(f"  - {col['name']} ({col['type']}){pk_indicator}")
            
            if info["sample_data"]:
                print("Sample data:")
                for i, row in enumerate(info["sample_data"]):
                    print(f"  Row {i+1}: {row}")
            print("-" * 50)

if __name__ == "__main__":
    print("ANALYZING ACTUAL TALLY DATABASE STRUCTURE")
    print("="*70)
    
    schema = get_actual_table_structure()
    
    # Save full schema to file
    with open("tally_actual_schema.json", "w") as f:
        json.dump(schema, f, indent=2, default=str)
    
    # Print relevant tables
    print("Run this first to understand your database structure!") 
    print(f"\nFull schema saved to 'tally_actual_schema.json'") 
    print_relevant_tables(schema)
    