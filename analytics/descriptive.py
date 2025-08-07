"""
Descriptive Analytics - "What happened?"
Historical data analysis and current state reporting
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .base import AnalyticsBase, AnalyticsResponse


class DescriptiveAnalytics(AnalyticsBase):
    """Descriptive analytics for historical data analysis"""
    
    def __init__(self, agent_type: str = "financial"):
        super().__init__("DescriptiveAnalytics", agent_type)
        self.supported_queries = [
            'cash_flow_analysis',
            'account_balance_summary',
            'transaction_patterns',
            'financial_position',
            'trend_analysis',
            'period_comparison'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform descriptive analytics on the provided data"""
        start_time = datetime.now()
        
        # Determine specific analysis type
        analysis_type = self.classify_analysis_type(query)
        
        # Perform analysis based on type
        if analysis_type == 'cash_flow_analysis':
            results = self.analyze_cash_flow(data, params)
        elif analysis_type == 'account_balance_summary':
            results = self.analyze_account_balances(data, params)
        elif analysis_type == 'transaction_patterns':
            results = self.analyze_transaction_patterns(data, params)
        elif analysis_type == 'financial_position':
            results = self.analyze_financial_position(data, params)
        elif analysis_type == 'trend_analysis':
            results = self.analyze_trends(data, params)
        elif analysis_type == 'period_comparison':
            results = self.analyze_period_comparison(data, params)
        else:
            results = self.analyze_general_summary(data, params)
        
        return self.prepare_response(
            analytics_type=analysis_type,
            query=query,
            results=results,
            start_time=start_time
        )
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the specific type of descriptive analysis needed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['cash flow', 'cashflow']):
            return 'cash_flow_analysis'
        elif any(word in query_lower for word in ['balance', 'account']):
            return 'account_balance_summary'
        elif any(word in query_lower for word in ['transaction', 'payment', 'receipt']):
            return 'transaction_patterns'
        elif any(word in query_lower for word in ['position', 'snapshot', 'current']):
            return 'financial_position'
        elif any(word in query_lower for word in ['trend', 'pattern', 'movement']):
            return 'trend_analysis'
        elif any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            return 'period_comparison'
        else:
            return 'general_summary'
    
    def analyze_cash_flow(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cash flow patterns"""
        required_columns = ['date', 'amount', 'transaction_type']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for cash flow analysis'}
        
        # Convert date column to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Separate inflows and outflows
        inflows = data[data['amount'] > 0]['amount'].sum()
        outflows = data[data['amount'] < 0]['amount'].sum()
        net_flow = inflows + outflows
        
        # Calculate daily cash flow
        daily_flow = data.groupby(data['date'].dt.date)['amount'].sum()
        
        # Calculate running balance
        running_balance = daily_flow.cumsum()
        
        # Identify peak and trough periods
        peak_date = running_balance.idxmax()
        trough_date = running_balance.idxmin()
        
        # Calculate statistics
        avg_daily_flow = daily_flow.mean()
        volatility = daily_flow.std()
        
        return {
            'period': {
                'start_date': data['date'].min().strftime('%Y-%m-%d'),
                'end_date': data['date'].max().strftime('%Y-%m-%d'),
                'total_days': len(daily_flow)
            },
            'cash_flow': {
                'total_inflows': inflows,
                'total_outflows': abs(outflows),
                'net_cash_flow': net_flow,
                'avg_daily_flow': avg_daily_flow,
                'volatility': volatility
            },
            'peaks_troughs': {
                'peak_date': peak_date.strftime('%Y-%m-%d'),
                'peak_balance': running_balance[peak_date],
                'trough_date': trough_date.strftime('%Y-%m-%d'),
                'trough_balance': running_balance[trough_date]
            },
            'daily_flow': daily_flow.to_dict(),
            'running_balance': running_balance.to_dict()
        }
    
    def analyze_account_balances(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze account balance summary"""
        required_columns = ['account_name', 'balance']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for account balance analysis'}
        
        # Sort by balance magnitude
        data_sorted = data.reindex(data['balance'].abs().sort_values(ascending=False).index)
        
        # Calculate summary statistics
        total_assets = data[data['balance'] > 0]['balance'].sum()
        total_liabilities = abs(data[data['balance'] < 0]['balance'].sum())
        net_worth = total_assets - total_liabilities
        
        # Identify top accounts
        top_assets = data_sorted[data_sorted['balance'] > 0].head(5)
        top_liabilities = data_sorted[data_sorted['balance'] < 0].head(5)
        
        # Calculate account distribution
        account_types = self.classify_account_types(data['account_name'].tolist())
        
        return {
            'summary': {
                'total_accounts': len(data),
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'net_worth': net_worth
            },
            'top_assets': [
                {'account': row['account_name'], 'balance': row['balance']} 
                for _, row in top_assets.iterrows()
            ],
            'top_liabilities': [
                {'account': row['account_name'], 'balance': row['balance']} 
                for _, row in top_liabilities.iterrows()
            ],
            'account_distribution': account_types,
            'all_accounts': data_sorted.to_dict('records')
        }
    
    def analyze_transaction_patterns(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transaction patterns"""
        required_columns = ['date', 'amount', 'transaction_type']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for transaction analysis'}
        
        # Convert date column to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Add time-based features
        data['day_of_week'] = data['date'].dt.day_name()
        data['month'] = data['date'].dt.month_name()
        data['hour'] = data['date'].dt.hour
        
        # Analyze by transaction type
        type_summary = data.groupby('transaction_type').agg({
            'amount': ['count', 'sum', 'mean', 'std']
        }).round(2)
        
        # Analyze by day of week
        dow_summary = data.groupby('day_of_week').agg({
            'amount': ['count', 'sum', 'mean']
        }).round(2)
        
        # Analyze by month
        month_summary = data.groupby('month').agg({
            'amount': ['count', 'sum', 'mean']
        }).round(2)
        
        # Find most active periods
        most_active_day = dow_summary['amount']['count'].idxmax()
        most_active_month = month_summary['amount']['count'].idxmax()
        
        return {
            'summary': {
                'total_transactions': len(data),
                'date_range': {
                    'start': data['date'].min().strftime('%Y-%m-%d'),
                    'end': data['date'].max().strftime('%Y-%m-%d')
                },
                'most_active_day': most_active_day,
                'most_active_month': most_active_month
            },
            'by_type': type_summary.to_dict(),
            'by_day_of_week': dow_summary.to_dict(),
            'by_month': month_summary.to_dict()
        }
    
    def analyze_financial_position(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current financial position"""
        required_columns = ['account_name', 'balance']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for financial position analysis'}
        
        # Classify accounts
        account_types = self.classify_account_types(data['account_name'].tolist())
        
        # Calculate key metrics
        cash_accounts = data[data['account_name'].str.contains('cash|bank', case=False, na=False)]
        receivables = data[data['account_name'].str.contains('receivable|debtor', case=False, na=False)]
        payables = data[data['account_name'].str.contains('payable|creditor', case=False, na=False)]
        
        total_cash = cash_accounts['balance'].sum()
        total_receivables = receivables['balance'].sum()
        total_payables = payables['balance'].sum()
        
        # Calculate liquidity ratios
        current_assets = data[data['balance'] > 0]['balance'].sum()
        current_liabilities = abs(data[data['balance'] < 0]['balance'].sum())
        
        liquidity_ratio = current_assets / current_liabilities if current_liabilities > 0 else float('inf')
        
        return {
            'snapshot_date': datetime.now().strftime('%Y-%m-%d'),
            'cash_position': {
                'total_cash': total_cash,
                'cash_accounts': cash_accounts.to_dict('records')
            },
            'receivables': {
                'total_receivables': total_receivables,
                'receivable_accounts': receivables.to_dict('records')
            },
            'payables': {
                'total_payables': total_payables,
                'payable_accounts': payables.to_dict('records')
            },
            'liquidity_metrics': {
                'current_assets': current_assets,
                'current_liabilities': current_liabilities,
                'liquidity_ratio': liquidity_ratio,
                'net_working_capital': current_assets - current_liabilities
            },
            'account_distribution': account_types
        }
    
    def analyze_trends(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in financial data"""
        required_columns = ['date', 'amount']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for trend analysis'}
        
        # Convert date column to datetime
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
        # Calculate moving averages
        data['ma_7'] = data['amount'].rolling(window=7).mean()
        data['ma_30'] = data['amount'].rolling(window=30).mean()
        
        # Identify trend direction
        recent_trend = self.identify_trends(data['amount'].tail(30))
        overall_trend = self.identify_trends(data['amount'])
        
        # Calculate growth rates
        if len(data) >= 2:
            first_value = data['amount'].iloc[0]
            last_value = data['amount'].iloc[-1]
            total_growth = self.calculate_percentage_change(last_value, first_value)
        else:
            total_growth = 0
        
        # Detect seasonal patterns
        seasonal_pattern = self.detect_seasonal_patterns(data)
        
        return {
            'trend_analysis': {
                'recent_trend': recent_trend,
                'overall_trend': overall_trend,
                'total_growth_rate': total_growth,
                'data_points': len(data)
            },
            'moving_averages': {
                'ma_7_current': data['ma_7'].iloc[-1] if len(data) >= 7 else None,
                'ma_30_current': data['ma_30'].iloc[-1] if len(data) >= 30 else None
            },
            'seasonal_patterns': seasonal_pattern,
            'time_series_data': data[['date', 'amount', 'ma_7', 'ma_30']].to_dict('records')
        }
    
    def analyze_period_comparison(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across different periods"""
        required_columns = ['date', 'amount']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for period comparison'}
        
        # Convert date column to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Split data into periods
        total_days = (data['date'].max() - data['date'].min()).days
        mid_point = data['date'].min() + timedelta(days=total_days // 2)
        
        period1 = data[data['date'] < mid_point]
        period2 = data[data['date'] >= mid_point]
        
        # Calculate metrics for each period
        period1_metrics = self.calculate_period_metrics(period1)
        period2_metrics = self.calculate_period_metrics(period2)
        
        # Calculate changes
        changes = {}
        for metric in ['total', 'average', 'count']:
            if metric in period1_metrics and metric in period2_metrics:
                changes[metric] = self.calculate_percentage_change(
                    period2_metrics[metric], period1_metrics[metric]
                )
        
        return {
            'period_1': {
                'date_range': {
                    'start': period1['date'].min().strftime('%Y-%m-%d'),
                    'end': period1['date'].max().strftime('%Y-%m-%d')
                },
                'metrics': period1_metrics
            },
            'period_2': {
                'date_range': {
                    'start': period2['date'].min().strftime('%Y-%m-%d'),
                    'end': period2['date'].max().strftime('%Y-%m-%d')
                },
                'metrics': period2_metrics
            },
            'changes': changes
        }
    
    def analyze_general_summary(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Provide general summary statistics"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        summary = {
            'data_overview': {
                'total_records': len(data),
                'columns': list(data.columns),
                'numeric_columns': list(numeric_columns)
            }
        }
        
        # Calculate summary statistics for numeric columns
        for col in numeric_columns:
            summary[col] = {
                'count': data[col].count(),
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median(),
                'sum': data[col].sum()
            }
        
        return summary
    
    def calculate_period_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for a specific period"""
        return {
            'total': data['amount'].sum(),
            'average': data['amount'].mean(),
            'count': len(data),
            'std': data['amount'].std(),
            'min': data['amount'].min(),
            'max': data['amount'].max()
        }
    
    def classify_account_types(self, account_names: List[str]) -> Dict[str, int]:
        """Classify accounts by type"""
        account_types = {
            'cash_bank': 0,
            'receivables': 0,
            'payables': 0,
            'inventory': 0,
            'fixed_assets': 0,
            'other': 0
        }
        
        for account in account_names:
            account_lower = account.lower()
            if any(word in account_lower for word in ['cash', 'bank']):
                account_types['cash_bank'] += 1
            elif any(word in account_lower for word in ['receivable', 'debtor']):
                account_types['receivables'] += 1
            elif any(word in account_lower for word in ['payable', 'creditor']):
                account_types['payables'] += 1
            elif any(word in account_lower for word in ['inventory', 'stock']):
                account_types['inventory'] += 1
            elif any(word in account_lower for word in ['fixed', 'asset', 'equipment']):
                account_types['fixed_assets'] += 1
            else:
                account_types['other'] += 1
        
        return account_types
    
    def detect_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal patterns in time series data"""
        if len(data) < 12:
            return {'pattern': 'insufficient_data'}
        
        # Add month column
        data['month'] = data['date'].dt.month
        
        # Calculate monthly averages
        monthly_avg = data.groupby('month')['amount'].mean()
        
        # Find peak and low months
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        
        return {
            'pattern': 'seasonal_detected',
            'peak_month': peak_month,
            'peak_value': monthly_avg[peak_month],
            'low_month': low_month,
            'low_value': monthly_avg[low_month],
            'monthly_averages': monthly_avg.to_dict()
        }
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def generate_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate insights specific to descriptive analytics"""
        insights = []
        
        # Cash flow insights
        if 'cash_flow' in analysis_results:
            cf = analysis_results['cash_flow']
            net_flow = cf.get('net_cash_flow', 0)
            if net_flow > 0:
                insights.append(f"Positive cash flow of {self.format_currency(net_flow)} indicates healthy liquidity")
            elif net_flow < 0:
                insights.append(f"Negative cash flow of {self.format_currency(abs(net_flow))} requires attention")
        
        # Account balance insights
        if 'summary' in analysis_results:
            summary = analysis_results['summary']
            net_worth = summary.get('net_worth', 0)
            if net_worth > 0:
                insights.append(f"Net worth of {self.format_currency(net_worth)} shows positive financial position")
            else:
                insights.append(f"Negative net worth of {self.format_currency(abs(net_worth))} indicates financial stress")
        
        # Trend insights
        if 'trend_analysis' in analysis_results:
            trend = analysis_results['trend_analysis']
            recent_trend = trend.get('recent_trend', 'stable')
            insights.append(f"Recent trend shows {recent_trend} pattern in financial activity")
        
        return insights