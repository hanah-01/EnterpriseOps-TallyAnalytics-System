import sqlite3
from typing import Any, Dict, List, Optional
from google.adk.tools import FunctionTool
from .query_helper import suggest_query
import os

DB_PATH = "tally.db"

def run_query(query: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Run a SQL query against the Tally SQLite database"""
    try:
        if not os.path.exists(DB_PATH):
            return {"status": "error", "error": f"Database file not found at {DB_PATH}"}
            
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
            return {"status": "success", "data": result, "row_count": len(result)}
        else:
            conn.commit()
            return {"status": "success", "rows_affected": cur.rowcount}
            
    except sqlite3.Error as e:
        return {"status": "error", "error": f"SQLite error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "error": f"Unexpected error: {str(e)}"}
    finally:
        if 'conn' in locals():
            conn.close()

def get_financial_data(intent: str, **kwargs) -> Dict[str, Any]:
    """Get financial data using AI-generated queries"""
    try:
        # Generate query using the corrected schema
        sql = suggest_query(intent, agent="financial")
        
        if sql.startswith("Error"):
            return {"status": "error", "error": sql}
        
        # Execute the query
        result = run_query(sql)
        
        if result["status"] == "success":
            return {
                "status": "success",
                "query": sql,
                "data": result["data"],
                "row_count": result["row_count"]
            }
        else:
            return result
            
    except Exception as e:
        return {"status": "error", "error": f"Financial data retrieval failed: {str(e)}"}

def get_banking_data(intent: str, **kwargs) -> Dict[str, Any]:
    """Get banking data using AI-generated queries"""
    try:
        sql = suggest_query(intent, agent="banking")
        
        if sql.startswith("Error"):
            return {"status": "error", "error": sql}
        
        result = run_query(sql)
        
        if result["status"] == "success":
            return {
                "status": "success", 
                "query": sql,
                "data": result["data"],
                "row_count": result["row_count"]
            }
        else:
            return result
            
    except Exception as e:
        return {"status": "error", "error": f"Banking data retrieval failed: {str(e)}"}

def get_tax_data(intent: str, **kwargs) -> Dict[str, Any]:
    """Get tax/GST data using AI-generated queries"""
    try:
        sql = suggest_query(intent, agent="tax")
        
        if sql.startswith("Error"):
            return {"status": "error", "error": sql}
        
        result = run_query(sql)
        
        if result["status"] == "success":
            return {
                "status": "success",
                "query": sql, 
                "data": result["data"],
                "row_count": result["row_count"]
            }
        else:
            return result
            
    except Exception as e:
        return {"status": "error", "error": f"Tax data retrieval failed: {str(e)}"}

def get_inventory_data(intent: str, **kwargs) -> Dict[str, Any]:
    """Get inventory data using AI-generated queries"""
    try:
        sql = suggest_query(intent, agent="inventory")
        
        if sql.startswith("Error"):
            return {"status": "error", "error": sql}
        
        result = run_query(sql)
        
        if result["status"] == "success":
            return {
                "status": "success",
                "query": sql,
                "data": result["data"], 
                "row_count": result["row_count"]
            }
        else:
            return result
            
    except Exception as e:
        return {"status": "error", "error": f"Inventory data retrieval failed: {str(e)}"}

def get_cost_data(intent: str, **kwargs) -> Dict[str, Any]:
    """Get cost management data using AI-generated queries"""
    try:
        sql = suggest_query(intent, agent="cost_management")
        
        if sql.startswith("Error"):
            return {"status": "error", "error": sql}
        
        result = run_query(sql)
        
        if result["status"] == "success":
            return {
                "status": "success",
                "query": sql,
                "data": result["data"],
                "row_count": result["row_count"]
            }
        else:
            return result
            
    except Exception as e:
        return {"status": "error", "error": f"Cost data retrieval failed: {str(e)}"}

# Create the tool objects
get_financial_data_tool = FunctionTool(func=get_financial_data)
get_banking_data_tool = FunctionTool(func=get_banking_data)
get_tax_data_tool = FunctionTool(func=get_tax_data)
get_inventory_data_tool = FunctionTool(func=get_inventory_data)
get_cost_data_tool = FunctionTool(func=get_cost_data)