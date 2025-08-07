"""
Database connection and query engine for Tally database
"""

import sqlite3
import os
from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


class QueryEngine:
    """Query engine for database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None, 
                     cache_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        try:
            if not self.connection:
                self._connect()
            
            cursor = self.connection.cursor()
            
            if params:
                # Convert named parameters to positional for SQLite
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Convert rows to dictionaries
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                row_dict = {}
                for i, value in enumerate(row):
                    row_dict[columns[i]] = value
                results.append(row_dict)
            
            cursor.close()
            return results
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            if params:
                logger.error(f"Parameters: {params}")
            raise
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information"""
        try:
            query = f"PRAGMA table_info({table_name})"
            result = self.execute_query(query)
            
            schema = {
                "table_name": table_name,
                "columns": []
            }
            
            for row in result:
                schema["columns"].append({
                    "name": row["name"],
                    "type": row["type"],
                    "nullable": not row["notnull"],
                    "primary_key": bool(row["pk"])
                })
            
            return schema
            
        except Exception as e:
            logger.error(f"Schema retrieval failed for table {table_name}: {e}")
            return {"table_name": table_name, "columns": [], "error": str(e)}
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            result = self.execute_query("SELECT 1 as test")
            return len(result) > 0 and result[0]["test"] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")


# Global query engine instance
_query_engine = None


def get_query_engine() -> QueryEngine:
    """Get global query engine instance"""
    global _query_engine
    
    if _query_engine is None:
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tally.db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        _query_engine = QueryEngine(db_path)
    
    return _query_engine


def close_query_engine():
    """Close global query engine"""
    global _query_engine
    if _query_engine:
        _query_engine.close()
        _query_engine = None
