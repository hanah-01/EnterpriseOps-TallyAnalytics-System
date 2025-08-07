"""
Diagnostic Analytics - "Why did it happen?"
Root cause analysis and correlation discovery
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .base import AnalyticsBase, AnalyticsResponse


class DiagnosticAnalytics(AnalyticsBase):
    """Diagnostic analytics for root cause analysis"""
    
    def __init__(self, agent_type: str = "financial"):
        super().__init__("DiagnosticAnalytics", agent_type)
        self.supported_queries = [
            'cash_flow_drivers',
            'payment_delay_analysis',
            'variance_analysis',
            'correlation_analysis',
            'anomaly_investigation',
            'performance_gaps'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform diagnostic analytics on the provided data"""
        start_time = datetime.now()
        
        analysis_type = self.classify_analysis_type(query)
        
        if analysis_type == 'cash_flow_drivers':
            results = self.analyze_cash_flow_drivers(data, params)
        elif analysis_type == 'payment_delay_analysis':
            results = self.analyze_payment_delays(data, params)
        elif analysis_type == 'variance_analysis':
            results = self.analyze_variance(data, params)
        elif analysis_type == 'correlation_analysis':
            results = self.analyze_correlations(data, params)
        elif analysis_type == 'anomaly_investigation':
            results = self.investigate_anomalies(data, params)
        elif analysis_type == 'performance_gaps':
            results = self.analyze_performance_gaps(data, params)
        else:
            results = self.analyze_general_causes(data, params)
        
        return self.prepare_response(
            analytics_type='diagnostic',
            query=query,
            results=results,
            start_time=start_time
        )
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the specific type of diagnostic analysis needed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['cash flow', 'cashflow', 'liquidity']):
            return 'cash_flow_drivers'
        elif any(word in query_lower for word in ['payment', 'delay', 'late']):
            return 'payment_delay_analysis'
        elif any(word in query_lower for word in ['variance', 'difference', 'budget']):
            return 'variance_analysis'
        elif any(word in query_lower for word in ['correlation', 'relationship', 'impact']):
            return 'correlation_analysis'
        elif any(word in query_lower for word in ['anomaly', 'unusual', 'spike']):
            return 'anomaly_investigation'
        elif any(word in query_lower for word in ['performance', 'gap', 'shortfall']):
            return 'performance_gaps'
        else:
            return 'general_causes'
    
    def analyze_cash_flow_drivers(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what's driving cash flow changes"""
        required_columns = ['date', 'amount', 'account_name', 'transaction_type']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for cash flow driver analysis'}
        
        data['date'] = pd.to_datetime(data['date'])
        
        # Calculate cash flow by account
        account_flow = data.groupby('account_name')['amount'].sum().sort_values(key=abs, ascending=False)
        
        # Identify major inflow and outflow drivers
        major_inflows = account_flow[account_flow > 0].head(5)
        major_outflows = account_flow[account_flow < 0].head(5)
        
        # Analyze timing patterns
        monthly_flow = data.groupby(data['date'].dt.to_period('M'))['amount'].sum()
        
        # Find months with extreme cash flow
        worst_month = monthly_flow.idxmin()
        best_month = monthly_flow.idxmax()
        
        # Analyze transaction frequency vs amount
        account_stats = data.groupby('account_name').agg({
            'amount': ['count', 'sum', 'mean', 'std']
        }).round(2)
        
        # Identify high-impact, low-frequency transactions
        high_impact = account_stats[account_stats['amount']['count'] < 5]
        high_impact = high_impact[abs(high_impact['amount']['sum']) > account_stats['amount']['sum'].quantile(0.8)]
        
        # Calculate contribution analysis
        total_flow = data['amount'].sum()
        contribution_analysis = {}
        for account in account_flow.index[:10]:  # Top 10 contributors
            contribution = (account_flow[account] / total_flow) * 100
            contribution_analysis[account] = {
                'amount': account_flow[account],
                'contribution_percent': contribution
            }
        
        return {
            'primary_drivers': {
                'major_inflows': major_inflows.to_dict(),
                'major_outflows': major_outflows.to_dict()
            },
            'temporal_analysis': {
                'worst_month': {
                    'period': str(worst_month),
                    'amount': monthly_flow[worst_month]
                },
                'best_month': {
                    'period': str(best_month),
                    'amount': monthly_flow[best_month]
                },
                'monthly_flow': monthly_flow.to_dict()
            },
            'high_impact_transactions': high_impact.to_dict(),
            'contribution_analysis': contribution_analysis,
            'summary_stats': {
                'total_accounts': len(account_flow),
                'net_flow': total_flow,
                'flow_concentration': (abs(account_flow.head(5)).sum() / abs(account_flow).sum()) * 100
            }
        }
    
    def analyze_payment_delays(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze causes of payment delays"""
        required_columns = ['date', 'due_date', 'amount', 'customer_name']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for payment delay analysis'}
        
        data['date'] = pd.to_datetime(data['date'])
        data['due_date'] = pd.to_datetime(data['due_date'])
        
        # Calculate delay days
        data['delay_days'] = (data['date'] - data['due_date']).dt.days
        
        # Separate on-time vs delayed payments
        on_time = data[data['delay_days'] <= 0]
        delayed = data[data['delay_days'] > 0]
        
        # Analyze by customer
        customer_delays = delayed.groupby('customer_name').agg({
            'delay_days': ['count', 'mean', 'max'],
            'amount': ['sum', 'mean']
        }).round(2)
        
        # Find patterns in delayed payments
        delay_patterns = {
            'by_amount': self.analyze_delay_by_amount(delayed),
            'by_customer_type': self.analyze_delay_by_customer_type(delayed),
            'by_day_of_week': self.analyze_delay_by_timing(delayed),
            'seasonal_patterns': self.analyze_delay_seasonal_patterns(delayed)
        }
        
        # Calculate impact
        total_delayed_amount = delayed['amount'].sum()
        avg_delay_days = delayed['delay_days'].mean()
        
        # Estimate financial impact (assuming interest rate)
        interest_rate = params.get('interest_rate', 0.12)  # 12% annual
        daily_rate = interest_rate / 365
        estimated_cost = total_delayed_amount * daily_rate * avg_delay_days
        
        return {
            'delay_summary': {
                'total_payments': len(data),
                'on_time_payments': len(on_time),
                'delayed_payments': len(delayed),
                'delay_rate': (len(delayed) / len(data)) * 100,
                'avg_delay_days': avg_delay_days,
                'total_delayed_amount': total_delayed_amount
            },
            'customer_analysis': customer_delays.to_dict(),
            'delay_patterns': delay_patterns,
            'financial_impact': {
                'estimated_cost': estimated_cost,
                'assumptions': {
                    'interest_rate': interest_rate,
                    'calculation_method': 'simple_interest'
                }
            },
            'root_causes': self.identify_delay_root_causes(delayed, customer_delays)
        }
    
    def analyze_variance(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze variance between actual and budgeted/expected values"""
        required_columns = ['account_name', 'actual', 'budget']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for variance analysis'}
        
        # Calculate variances
        data['variance'] = data['actual'] - data['budget']
        data['variance_percent'] = (data['variance'] / data['budget']) * 100
        
        # Identify significant variances
        significant_threshold = params.get('variance_threshold', 10)  # 10% threshold
        significant_variances = data[abs(data['variance_percent']) > significant_threshold]
        
        # Categorize variances
        favorable_variances = data[data['variance'] > 0]
        unfavorable_variances = data[data['variance'] < 0]
        
        # Analyze variance patterns
        variance_analysis = {
            'by_magnitude': self.analyze_variance_by_magnitude(data),
            'by_category': self.analyze_variance_by_category(data),
            'trend_analysis': self.analyze_variance_trends(data, params)
        }
        
        # Root cause analysis for significant variances
        root_causes = {}
        for _, row in significant_variances.iterrows():
            account = row['account_name']
            root_causes[account] = self.identify_variance_causes(row, data, params)
        
        return {
            'variance_summary': {
                'total_accounts': len(data),
                'significant_variances': len(significant_variances),
                'total_variance': data['variance'].sum(),
                'total_favorable': favorable_variances['variance'].sum(),
                'total_unfavorable': unfavorable_variances['variance'].sum()
            },
            'significant_variances': significant_variances.to_dict('records'),
            'variance_analysis': variance_analysis,
            'root_causes': root_causes,
            'recommendations': self.generate_variance_recommendations(significant_variances)
        }
    
    def analyze_correlations(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations between different variables"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return {'error': 'Insufficient numeric columns for correlation analysis'}
        
        # Calculate correlation matrix
        correlation_matrix = data[numeric_columns].corr()
        
        # Find strong correlations (excluding self-correlations)
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                    })
        
        # Analyze causation vs correlation
        causation_analysis = self.analyze_causation_indicators(data, strong_correlations)
        
        # Time-based correlation analysis
        if 'date' in data.columns:
            time_correlations = self.analyze_time_based_correlations(data)
        else:
            time_correlations = {}
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'causation_analysis': causation_analysis,
            'time_correlations': time_correlations,
            'insights': self.generate_correlation_insights(strong_correlations)
        }
    
    def investigate_anomalies(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Investigate anomalies and their causes"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return {'error': 'No numeric columns found for anomaly investigation'}
        
        anomaly_results = {}
        
        for column in numeric_columns:
            # Statistical anomaly detection
            z_scores = np.abs(stats.zscore(data[column].dropna()))
            anomalies = data[z_scores > 3]  # 3 standard deviations
            
            if len(anomalies) > 0:
                # Investigate causes
                anomaly_investigation = self.investigate_anomaly_causes(anomalies, data, column)
                
                anomaly_results[column] = {
                    'anomaly_count': len(anomalies),
                    'anomaly_values': anomalies[column].tolist(),
                    'investigation': anomaly_investigation
                }
        
        return {
            'anomaly_summary': {
                'columns_analyzed': len(numeric_columns),
                'columns_with_anomalies': len(anomaly_results)
            },
            'anomaly_details': anomaly_results,
            'investigation_summary': self.summarize_anomaly_investigations(anomaly_results)
        }
    
    def analyze_performance_gaps(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance gaps and their causes"""
        required_columns = ['metric_name', 'actual', 'target']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for performance gap analysis'}
        
        # Calculate gaps
        data['gap'] = data['actual'] - data['target']
        data['gap_percent'] = (data['gap'] / data['target']) * 100
        
        # Identify significant gaps
        gap_threshold = params.get('gap_threshold', 5)  # 5% threshold
        significant_gaps = data[abs(data['gap_percent']) > gap_threshold]
        
        # Analyze gap patterns
        gap_analysis = {
            'by_metric_type': self.analyze_gaps_by_metric_type(significant_gaps),
            'by_magnitude': self.analyze_gaps_by_magnitude(significant_gaps),
            'trend_analysis': self.analyze_gap_trends(data, params)
        }
        
        # Root cause analysis for each significant gap
        root_causes = {}
        for _, row in significant_gaps.iterrows():
            metric = row['metric_name']
            root_causes[metric] = self.identify_gap_causes(row, data, params)
        
        return {
            'gap_summary': {
                'total_metrics': len(data),
                'significant_gaps': len(significant_gaps),
                'underperformance_count': len(data[data['gap'] < 0]),
                'overperformance_count': len(data[data['gap'] > 0])
            },
            'significant_gaps': significant_gaps.to_dict('records'),
            'gap_analysis': gap_analysis,
            'root_causes': root_causes,
            'improvement_opportunities': self.identify_improvement_opportunities(significant_gaps)
        }
    
    def analyze_general_causes(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General cause analysis for any dataset"""
        return {
            'data_summary': {
                'total_records': len(data),
                'columns': list(data.columns),
                'missing_data': data.isnull().sum().to_dict()
            },
            'statistical_summary': data.describe().to_dict(),
            'potential_issues': self.identify_data_quality_issues(data)
        }
    
    # Helper methods for specific analysis types
    
    def analyze_delay_by_amount(self, delayed_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze delay patterns by transaction amount"""
        # Create amount bins
        delayed_data['amount_bin'] = pd.cut(delayed_data['amount'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        delay_by_amount = delayed_data.groupby('amount_bin')['delay_days'].agg(['count', 'mean', 'max']).round(2)
        
        return delay_by_amount.to_dict()
    
    def analyze_delay_by_customer_type(self, delayed_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze delay patterns by customer type"""
        # Simple customer type classification based on name patterns
        delayed_data['customer_type'] = delayed_data['customer_name'].apply(self.classify_customer_type)
        
        delay_by_type = delayed_data.groupby('customer_type')['delay_days'].agg(['count', 'mean', 'max']).round(2)
        
        return delay_by_type.to_dict()
    
    def analyze_delay_by_timing(self, delayed_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze delay patterns by timing"""
        delayed_data['payment_day'] = delayed_data['date'].dt.day_name()
        
        delay_by_day = delayed_data.groupby('payment_day')['delay_days'].agg(['count', 'mean']).round(2)
        
        return delay_by_day.to_dict()
    
    def analyze_delay_seasonal_patterns(self, delayed_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in payment delays"""
        delayed_data['month'] = delayed_data['date'].dt.month_name()
        
        seasonal_delays = delayed_data.groupby('month')['delay_days'].agg(['count', 'mean']).round(2)
        
        return seasonal_delays.to_dict()
    
    def identify_delay_root_causes(self, delayed_data: pd.DataFrame, customer_delays: pd.DataFrame) -> List[str]:
        """Identify root causes of payment delays"""
        causes = []
        
        # Check for customer concentration
        top_delayers = customer_delays.nlargest(3, ('delay_days', 'count'))
        if len(top_delayers) > 0:
            causes.append(f"Customer concentration: Top 3 customers account for {(top_delayers[('delay_days', 'count')].sum() / len(delayed_data)) * 100:.1f}% of delays")
        
        # Check for amount patterns
        high_amount_delays = delayed_data[delayed_data['amount'] > delayed_data['amount'].quantile(0.8)]
        if len(high_amount_delays) > 0:
            causes.append(f"High-value transactions: {len(high_amount_delays)} delays involve amounts > 80th percentile")
        
        return causes
    
    def classify_customer_type(self, customer_name: str) -> str:
        """Classify customer type based on name patterns"""
        name_lower = customer_name.lower()
        
        if any(word in name_lower for word in ['ltd', 'limited', 'corporation', 'corp', 'inc']):
            return 'Corporate'
        elif any(word in name_lower for word in ['shop', 'store', 'mart', 'retail']):
            return 'Retail'
        elif any(word in name_lower for word in ['mobile', 'electronics', 'tech']):
            return 'Electronics'
        else:
            return 'Individual'
    
    def analyze_variance_by_magnitude(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze variances by magnitude"""
        data['magnitude'] = abs(data['variance'])
        
        return {
            'large_variances': data[data['magnitude'] > data['magnitude'].quantile(0.9)].to_dict('records'),
            'variance_distribution': data['magnitude'].describe().to_dict()
        }
    
    def analyze_variance_by_category(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze variances by account category"""
        # Simple categorization
        data['category'] = data['account_name'].apply(self.categorize_account)
        
        category_variance = data.groupby('category')['variance'].agg(['sum', 'mean', 'count']).round(2)
        
        return category_variance.to_dict()
    
    def analyze_variance_trends(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze variance trends over time"""
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            monthly_variance = data.groupby(data['date'].dt.to_period('M'))['variance'].sum()
            
            return {
                'monthly_trends': monthly_variance.to_dict(),
                'trend_direction': self.identify_trends(monthly_variance)
            }
        else:
            return {'message': 'No date column available for trend analysis'}
    
    def categorize_account(self, account_name: str) -> str:
        """Categorize account based on name"""
        name_lower = account_name.lower()
        
        if any(word in name_lower for word in ['revenue', 'sales', 'income']):
            return 'Revenue'
        elif any(word in name_lower for word in ['expense', 'cost', 'expense']):
            return 'Expense'
        elif any(word in name_lower for word in ['asset', 'cash', 'bank']):
            return 'Asset'
        elif any(word in name_lower for word in ['liability', 'payable']):
            return 'Liability'
        else:
            return 'Other'
    
    def identify_variance_causes(self, row: pd.Series, data: pd.DataFrame, params: Dict[str, Any]) -> List[str]:
        """Identify causes of significant variance for a specific account"""
        causes = []
        
        variance_percent = abs(row['variance_percent'])
        
        if variance_percent > 50:
            causes.append("Major variance indicates significant budget deviation")
        elif variance_percent > 25:
            causes.append("Moderate variance suggests budget adjustment needed")
        
        return causes
    
    def generate_variance_recommendations(self, significant_variances: pd.DataFrame) -> List[str]:
        """Generate recommendations based on variance analysis"""
        recommendations = []
        
        if len(significant_variances) > 0:
            recommendations.append("Review budget assumptions for accounts with significant variances")
            recommendations.append("Implement monthly variance monitoring and reporting")
            recommendations.append("Investigate root causes of major variances")
        
        return recommendations
    
    def analyze_causation_indicators(self, data: pd.DataFrame, correlations: List[Dict]) -> Dict[str, Any]:
        """Analyze indicators of causation vs correlation"""
        causation_indicators = {}
        
        for corr in correlations:
            var1, var2 = corr['variable1'], corr['variable2']
            
            # Time-based analysis (if date column exists)
            if 'date' in data.columns:
                # Check for lead-lag relationships
                lead_lag = self.analyze_lead_lag_relationship(data, var1, var2)
                causation_indicators[f"{var1}_{var2}"] = {
                    'correlation': corr['correlation'],
                    'lead_lag_analysis': lead_lag,
                    'causation_likelihood': 'high' if lead_lag['significant_lag'] else 'low'
                }
        
        return causation_indicators
    
    def analyze_lead_lag_relationship(self, data: pd.DataFrame, var1: str, var2: str) -> Dict[str, Any]:
        """Analyze lead-lag relationship between two variables"""
        # Simple implementation - would need more sophisticated analysis in production
        return {
            'significant_lag': False,
            'lag_period': 0,
            'note': 'Advanced lead-lag analysis requires time series methods'
        }
    
    def analyze_time_based_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations over time"""
        if 'date' not in data.columns:
            return {}
        
        data['date'] = pd.to_datetime(data['date'])
        
        # Group by month and calculate correlations
        monthly_data = data.groupby(data['date'].dt.to_period('M')).mean()
        
        if len(monthly_data) > 3:
            monthly_corr = monthly_data.corr()
            return {
                'monthly_correlations': monthly_corr.to_dict(),
                'correlation_stability': 'stable' if monthly_corr.std().mean() < 0.3 else 'variable'
            }
        
        return {'message': 'Insufficient time periods for temporal correlation analysis'}
    
    def generate_correlation_insights(self, correlations: List[Dict]) -> List[str]:
        """Generate insights from correlation analysis"""
        insights = []
        
        strong_positive = [c for c in correlations if c['correlation'] > 0.7]
        strong_negative = [c for c in correlations if c['correlation'] < -0.7]
        
        if strong_positive:
            insights.append(f"Found {len(strong_positive)} strong positive correlations")
        if strong_negative:
            insights.append(f"Found {len(strong_negative)} strong negative correlations")
        
        return insights
    
    def investigate_anomaly_causes(self, anomalies: pd.DataFrame, full_data: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Investigate causes of anomalies in a specific column"""
        investigation = {
            'potential_causes': [],
            'data_quality_issues': [],
            'contextual_factors': []
        }
        
        # Check for data entry errors
        if anomalies[column].max() > full_data[column].quantile(0.99) * 10:
            investigation['data_quality_issues'].append("Potential data entry error - values much higher than normal")
        
        # Check for timing issues
        if 'date' in anomalies.columns:
            investigation['contextual_factors'].append("Anomalies distributed across time periods")
        
        return investigation
    
    def summarize_anomaly_investigations(self, anomaly_results: Dict[str, Any]) -> List[str]:
        """Summarize findings from anomaly investigations"""
        summary = []
        
        total_anomalies = sum(result['anomaly_count'] for result in anomaly_results.values())
        summary.append(f"Total anomalies detected: {total_anomalies}")
        
        return summary
    
    def identify_data_quality_issues(self, data: pd.DataFrame) -> List[str]:
        """Identify potential data quality issues"""
        issues = []
        
        # Check for missing data
        missing_pct = (data.isnull().sum() / len(data)) * 100
        high_missing = missing_pct[missing_pct > 20]
        if len(high_missing) > 0:
            issues.append(f"High missing data in columns: {high_missing.index.tolist()}")
        
        # Check for duplicate records
        if data.duplicated().sum() > 0:
            issues.append(f"Found {data.duplicated().sum()} duplicate records")
        
        return issues
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diagnostic analysis"""
        recommendations = []
        
        if 'primary_drivers' in analysis_results:
            recommendations.append("Focus on managing top cash flow drivers for maximum impact")
        
        if 'delay_summary' in analysis_results:
            delay_rate = analysis_results['delay_summary'].get('delay_rate', 0)
            if delay_rate > 20:
                recommendations.append("Implement stricter payment terms and follow-up procedures")
        
        if 'root_causes' in analysis_results:
            recommendations.append("Address identified root causes through targeted interventions")
        
        return recommendations