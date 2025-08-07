"""
Enhanced Financial Agent with 4-Tier Analytics
Integration with your existing Tally system
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

try:
    from .base import BaseAgent
    from models.responses import QueryType
except ImportError:
    from base import BaseAgent
    from models.responses import QueryType
import logging

# Set up logging
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class FinancialAgent(BaseAgent):
    """
    Enhanced Financial Agent with Analytics Integration
    
    Capabilities:
    - Financial data analysis and reporting
    - Cash flow analysis
    - Profit & Loss analysis  
    - Account balance tracking
    - 4-Tier Analytics: Descriptive, Diagnostic, Predictive, Prescriptive
    """
    
    def __init__(self):
        super().__init__(
            name="Financial Agent",
            agent_type=QueryType.FINANCIAL,
            tables=["mst_ledger", "trn_accounting", "trn_voucher", "mst_group", "config"],
            model="gemini-2.5-flash"
        )
        logger.info("Enhanced Financial Agent initialized with analytics")
    
    def _initialize_tools(self) -> List[Dict[str, Any]]:
        """Initialize financial-specific tools"""
        return [
            {
                "name": "get_financial_summary",
                "description": "Get comprehensive financial summary with key metrics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                        "account_type": {"type": "string", "description": "Account type filter"}
                    }
                }
            },
            {
                "name": "get_cash_flow_analysis", 
                "description": "Analyze cash flow patterns and trends",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                        "period": {"type": "string", "description": "Aggregation period: daily, weekly, monthly"}
                    }
                }
            },
            {
                "name": "get_account_balances",
                "description": "Get account balances and ledger information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "account_name": {"type": "string", "description": "Specific account name filter"},
                        "as_of_date": {"type": "string", "description": "Balance as of date (YYYY-MM-DD)"},
                        "include_zero": {"type": "boolean", "description": "Include zero balance accounts"}
                    }
                }
            },
            {
                "name": "get_profit_loss_analysis",
                "description": "Generate profit and loss analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                        "group_by": {"type": "string", "description": "Group by period: monthly, quarterly"}
                    }
                }
            }
        ]
    
    def get_financial_summary(self, date_from: str = None, date_to: str = None, 
                            account_type: str = None) -> Dict[str, Any]:
        """Get comprehensive financial summary"""
        try:
            # Build query conditions
            where_conditions = []
            if date_from:
                where_conditions.append(f"v.date >= '{date_from}'")
            if date_to:
                where_conditions.append(f"v.date <= '{date_to}'")
            if account_type:
                where_conditions.append(f"l.parent LIKE '%{account_type}%'")
                
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Financial summary query
            query = f"""
            SELECT 
                COUNT(DISTINCT a.ledger) as total_accounts,
                COUNT(DISTINCT v.voucher_number) as total_transactions,
                SUM(CASE WHEN a.amount > 0 THEN a.amount ELSE 0 END) as total_credits,
                SUM(CASE WHEN a.amount < 0 THEN ABS(a.amount) ELSE 0 END) as total_debits,
                SUM(a.amount) as net_balance,
                MIN(v.date) as earliest_transaction,
                MAX(v.date) as latest_transaction,
                AVG(ABS(a.amount)) as avg_transaction_amount
            FROM trn_accounting a
            JOIN trn_voucher v ON a.guid = v.guid
            LEFT JOIN mst_ledger l ON a.ledger = l.name
            WHERE {where_clause}
            """
            
            result = self.execute_database_query(query)
            
            if result:
                summary = result[0]
                
                # Get account breakdown
                account_query = f"""
                SELECT 
                    l.parent as account_group,
                    COUNT(DISTINCT a.ledger) as account_count,
                    SUM(a.amount) as group_balance
                FROM trn_accounting a
                JOIN trn_voucher v ON a.guid = v.guid
                LEFT JOIN mst_ledger l ON a.ledger = l.name
                WHERE {where_clause}
                GROUP BY l.parent
                ORDER BY ABS(SUM(a.amount)) DESC
                """
                
                account_breakdown = self.execute_database_query(account_query)
                
                return {
                    "summary": {
                        "total_accounts": summary["total_accounts"],
                        "total_transactions": summary["total_transactions"],
                        "total_credits": round(summary["total_credits"] or 0, 2),
                        "total_debits": round(summary["total_debits"] or 0, 2),
                        "net_balance": round(summary["net_balance"] or 0, 2),
                        "avg_transaction_amount": round(summary["avg_transaction_amount"] or 0, 2),
                        "period": f"{summary['earliest_transaction']} to {summary['latest_transaction']}"
                    },
                    "account_breakdown": [
                        {
                            "group": row["account_group"] or "Unclassified",
                            "account_count": row["account_count"],
                            "balance": round(row["group_balance"] or 0, 2)
                        }
                        for row in account_breakdown[:10]  # Top 10 groups
                    ],
                    "filters_applied": {
                        "date_from": date_from,
                        "date_to": date_to,
                        "account_type": account_type
                    }
                }
            else:
                return {"error": "No financial data found"}
                
        except Exception as e:
            logger.error(f"Financial summary failed: {e}")
            return {"error": f"Financial summary failed: {str(e)}"}
    
    def get_cash_flow_analysis(self, date_from: str = None, date_to: str = None,
                              period: str = "monthly") -> Dict[str, Any]:
        """Analyze cash flow patterns"""
        try:
            where_conditions = []
            if date_from:
                where_conditions.append(f"v.date >= '{date_from}'")
            if date_to:
                where_conditions.append(f"v.date <= '{date_to}'")
                
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Determine grouping
            if period == "daily":
                date_group = "DATE(v.date)"
            elif period == "weekly":
                date_group = "strftime('%Y-W%W', v.date)"
            else:  # monthly
                date_group = "strftime('%Y-%m', v.date)"
            
            # Cash flow analysis (focusing on cash/bank accounts)
            query = f"""
            SELECT 
                {date_group} as period,
                SUM(CASE WHEN a.amount > 0 THEN a.amount ELSE 0 END) as cash_inflow,
                SUM(CASE WHEN a.amount < 0 THEN ABS(a.amount) ELSE 0 END) as cash_outflow,
                SUM(a.amount) as net_cash_flow,
                COUNT(DISTINCT v.voucher_number) as transaction_count
            FROM trn_accounting a
            JOIN trn_voucher v ON a.guid = v.guid  
            WHERE {where_clause}
            AND (a.ledger LIKE '%Cash%' OR a.ledger LIKE '%Bank%')
            GROUP BY {date_group}
            ORDER BY period
            """
            
            result = self.execute_database_query(query)
            
            if result:
                cash_flow_data = []
                total_inflow = 0
                total_outflow = 0
                
                for row in result:
                    inflow = row["cash_inflow"] or 0
                    outflow = row["cash_outflow"] or 0
                    net_flow = row["net_cash_flow"] or 0
                    
                    total_inflow += inflow
                    total_outflow += outflow
                    
                    cash_flow_data.append({
                        "period": row["period"],
                        "inflow": round(inflow, 2),
                        "outflow": round(outflow, 2),
                        "net_flow": round(net_flow, 2),
                        "transactions": row["transaction_count"]
                    })
                
                return {
                    "cash_flow_analysis": cash_flow_data,
                    "summary": {
                        "total_inflow": round(total_inflow, 2),
                        "total_outflow": round(total_outflow, 2),
                        "net_cash_flow": round(total_inflow - total_outflow, 2),
                        "periods_analyzed": len(cash_flow_data)
                    },
                    "period_type": period
                }
            else:
                return {"error": "No cash flow data found"}
                
        except Exception as e:
            logger.error(f"Cash flow analysis failed: {e}")
            return {"error": f"Cash flow analysis failed: {str(e)}"}
    
    def get_account_balances(self, account_name: str = None, as_of_date: str = None,
                           include_zero: bool = False) -> Dict[str, Any]:
        """Get account balances"""
        try:
            where_conditions = []
            if account_name:
                where_conditions.append(f"l.name LIKE '%{account_name}%'")
            if as_of_date:
                where_conditions.append(f"v.date <= '{as_of_date}'")
                
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query = f"""
            SELECT 
                l.name as account_name,
                l.parent as account_group,
                l.opening_balance,
                l.is_revenue,
                l.is_deemedpositive,
                COALESCE(SUM(a.amount), 0) as transaction_balance,
                COUNT(a.amount) as transaction_count,
                MAX(v.date) as last_transaction_date
            FROM mst_ledger l
            LEFT JOIN trn_accounting a ON l.name = a.ledger
            LEFT JOIN trn_voucher v ON a.guid = v.guid
            WHERE {where_clause}
            GROUP BY l.name, l.parent, l.opening_balance, l.is_revenue, l.is_deemedpositive
            """
            
            if not include_zero:
                query += " HAVING (l.opening_balance != 0 OR COALESCE(SUM(a.amount), 0) != 0)"
            
            query += " ORDER BY ABS(l.opening_balance + COALESCE(SUM(a.amount), 0)) DESC"
            
            result = self.execute_database_query(query)
            
            if result:
                accounts = []
                for row in result:
                    opening_balance = float(row["opening_balance"] or 0)
                    transaction_balance = float(row["transaction_balance"] or 0)
                    
                    # Calculate closing balance based on account type
                    if row["is_deemedpositive"] == 1:
                        # Asset/Expense accounts (Debit nature)
                        closing_balance = opening_balance + transaction_balance
                    else:
                        # Liability/Income accounts (Credit nature)
                        closing_balance = opening_balance + transaction_balance
                    
                    accounts.append({
                        "account_name": row["account_name"],
                        "account_group": row["account_group"] or "Unclassified",
                        "opening_balance": round(opening_balance, 2),
                        "transaction_balance": round(transaction_balance, 2),
                        "closing_balance": round(closing_balance, 2),
                        "transaction_count": row["transaction_count"] or 0,
                        "last_transaction": row["last_transaction_date"],
                        "account_type": "Asset/Expense" if row["is_deemedpositive"] == 1 else "Liability/Income"
                    })
                
                return {
                    "accounts": accounts,
                    "total_accounts": len(accounts),
                    "as_of_date": as_of_date or "Latest",
                    "include_zero_balance": include_zero
                }
            else:
                return {"error": "No account data found"}
                
        except Exception as e:
            logger.error(f"Account balance query failed: {e}")
            return {"error": f"Account balance query failed: {str(e)}"}
    
    def get_profit_loss_analysis(self, date_from: str = None, date_to: str = None,
                               group_by: str = "monthly") -> Dict[str, Any]:
        """Generate profit and loss analysis"""
        try:
            where_conditions = []
            if date_from:
                where_conditions.append(f"v.date >= '{date_from}'")
            if date_to:
                where_conditions.append(f"v.date <= '{date_to}'")
                
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Determine grouping
            if group_by == "quarterly":
                date_group = "strftime('%Y-Q', v.date) || CASE WHEN CAST(strftime('%m', v.date) AS INTEGER) <= 3 THEN '1' WHEN CAST(strftime('%m', v.date) AS INTEGER) <= 6 THEN '2' WHEN CAST(strftime('%m', v.date) AS INTEGER) <= 9 THEN '3' ELSE '4' END"
            else:  # monthly
                date_group = "strftime('%Y-%m', v.date)"
            
            # P&L Query focusing on revenue and expense accounts
            query = f"""
            SELECT 
                {date_group} as period,
                SUM(CASE WHEN l.is_revenue = 1 THEN a.amount ELSE 0 END) as total_revenue,
                SUM(CASE WHEN l.is_revenue = 0 THEN ABS(a.amount) ELSE 0 END) as total_expenses,
                COUNT(DISTINCT v.voucher_number) as transaction_count
            FROM trn_accounting a
            JOIN trn_voucher v ON a.guid = v.guid
            JOIN mst_ledger l ON a.ledger = l.name
            WHERE {where_clause}
            AND l.is_revenue IS NOT NULL
            GROUP BY {date_group}
            ORDER BY period
            """
            
            result = self.execute_database_query(query)
            
            if result:
                pl_data = []
                total_revenue = 0
                total_expenses = 0
                
                for row in result:
                    revenue = row["total_revenue"] or 0
                    expenses = row["total_expenses"] or 0
                    profit = revenue - expenses
                    
                    total_revenue += revenue
                    total_expenses += expenses
                    
                    pl_data.append({
                        "period": row["period"],
                        "revenue": round(revenue, 2),
                        "expenses": round(expenses, 2),
                        "profit": round(profit, 2),
                        "profit_margin": round((profit / revenue * 100) if revenue > 0 else 0, 2),
                        "transactions": row["transaction_count"]
                    })
                
                net_profit = total_revenue - total_expenses
                
                return {
                    "profit_loss_analysis": pl_data,
                    "summary": {
                        "total_revenue": round(total_revenue, 2),
                        "total_expenses": round(total_expenses, 2),
                        "net_profit": round(net_profit, 2),
                        "overall_margin": round((net_profit / total_revenue * 100) if total_revenue > 0 else 0, 2),
                        "periods_analyzed": len(pl_data)
                    },
                    "grouping": group_by
                }
            else:
                return {"error": "No profit/loss data found"}
                
        except Exception as e:
            logger.error(f"P&L analysis failed: {e}")
            return {"error": f"P&L analysis failed: {str(e)}"}
    
    def _get_analytics_data(self, date_from: str = None, date_to: str = None,
                           parameters: Dict[str, Any] = None) -> pd.DataFrame:
        """Get financial data for analytics"""
        try:
            where_conditions = []
            if date_from:
                where_conditions.append(f"v.date >= '{date_from}'")
            if date_to:
                where_conditions.append(f"v.date <= '{date_to}'")
                
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Comprehensive financial data query
            query = f"""
            SELECT 
                v.date,
                v.voucher_type,
                v.voucher_number,
                v.party_name,
                a.ledger,
                a.amount,
                l.parent as account_group,
                l.is_revenue,
                l.is_deemedpositive,
                l.opening_balance
            FROM trn_accounting a
            JOIN trn_voucher v ON a.guid = v.guid
            LEFT JOIN mst_ledger l ON a.ledger = l.name
            WHERE {where_clause}
            ORDER BY v.date DESC
            LIMIT 2000
            """
            
            result = self.execute_database_query(query)
            return pd.DataFrame(result) if result else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Financial analytics data retrieval failed: {e}")
            return pd.DataFrame()
    
    def get_summary(self, **kwargs) -> Dict[str, Any]:
        """Get financial summary - compatibility method"""
        return self.get_financial_summary(**kwargs)
