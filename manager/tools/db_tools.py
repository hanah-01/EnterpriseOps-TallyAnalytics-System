import sqlite3
from typing import Any, Dict, List, Optional
from google.adk.tools import FunctionTool
from .query_helper import suggest_query

DB_PATH = r"C:\Users\manoj\tallydb.db" 

def run_query(query: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Run a SQL query against the Tally SQLite database and return the results as a list of dicts.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        if query.strip().lower().startswith("select"):
            rows = cur.fetchall()
            result = [dict(row) for row in rows]
            return {"status": "success", "data": result}
        else:
            conn.commit()
            return {"status": "success", "rows_affected": cur.rowcount}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        if 'conn' in locals():
            conn.close()

def get_financial_data(intent: str, description: str = "Get financial data from the Tally database", example: str = "Show me the total revenue for the last quarter") -> dict:
    sql = suggest_query(intent, agent="financial")
    if sql.lower().startswith("error"):
        return {"status": "error", "error": sql}
    return run_query(sql)

def get_cost_data(intent: str, description: str = "Get cost management data from the Tally database", example: str = "Show me the cost center analysis for last month") -> dict:
    sql = suggest_query(intent, agent="cost")
    if sql.lower().startswith("error"):
        return {"status": "error", "error": sql}
    return run_query(sql)

def get_banking_data(intent: str, description: str = "Get banking data from the Tally database", example: str = "Show me the cash flow for the last quarter") -> dict:
    sql = suggest_query(intent, agent="banking")
    if sql.lower().startswith("error"):
        return {"status": "error", "error": sql}
    return run_query(sql)

def get_tax_data(intent: str, description: str = "Get tax and compliance data from the Tally database", example: str = "Show me the GST payments for the last quarter") -> dict:
    sql = suggest_query(intent, agent="tax")
    if sql.lower().startswith("error"):
        return {"status": "error", "error": sql}
    return run_query(sql)

def get_inventory_data(intent: str) -> dict:
    sql = suggest_query(intent, agent="inventory")
    if sql.lower().startswith("error"):
        return {"status": "error", "error": sql}
    return run_query(sql)

def get_total_balance():
    query = "SELECT SUM(balance) as total_balance FROM mst_ledger;"
    result = run_query(query)
    return result["data"][0]['total_balance'] if result["status"] == "success" and result["data"] else None

# ADK FunctionTool definitions
get_financial_data_tool = FunctionTool(func=get_financial_data)
get_cost_data_tool = FunctionTool(func=get_cost_data)
get_banking_data_tool = FunctionTool(func=get_banking_data)
get_tax_data_tool = FunctionTool(func=get_tax_data)
get_inventory_data_tool = FunctionTool(func=get_inventory_data)