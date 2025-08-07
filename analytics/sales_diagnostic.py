"""
Sales Diagnostic Analytics - "Why did it happen?"
Root cause analysis for sales patterns and performance issues
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from .base import AnalyticsBase, AnalyticsResponse


class SalesDiagnosticAnalytics(AnalyticsBase):
    """Diagnostic analytics for sales root cause analysis"""
    
    def __init__(self, agent_type: str = "sales"):
        super().__init__("SalesDiagnosticAnalytics", agent_type)
        self.supported_queries = [
            'revenue_decline_analysis',
            'customer_churn_analysis',
            'sales_performance_issues',
            'transaction_anomaly_detection',
            'product_underperformance_analysis',
            'seasonal_variance_analysis',
            'customer_behavior_changes',
            'voucher_type_performance_issues'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform diagnostic analytics on sales data"""
        start_time = datetime.now()
        
        # Determine specific analysis type
        analysis_type = self.classify_analysis_type(query)
        
        # Perform analysis based on type
        if analysis_type == 'revenue_decline_analysis':
            results = self.analyze_revenue_decline(data, params)
        elif analysis_type == 'customer_churn_analysis':
            results = self.analyze_customer_churn(data, params)
        elif analysis_type == 'sales_performance_issues':
            results = self.analyze_sales_performance_issues(data, params)
        elif analysis_type == 'transaction_anomaly_detection':
            results = self.analyze_transaction_anomalies(data, params)
        elif analysis_type == 'product_underperformance_analysis':
            results = self.analyze_product_underperformance(data, params)
        elif analysis_type == 'seasonal_variance_analysis':
            results = self.analyze_seasonal_variance(data, params)
        elif analysis_type == 'customer_behavior_changes':
            results = self.analyze_customer_behavior_changes(data, params)
        elif analysis_type == 'voucher_type_performance_issues':
            results = self.analyze_voucher_performance_issues(data, params)
        else:
            results = self.analyze_general_sales_diagnostics(data, params)
        
        return self.prepare_response(
            analytics_type=analysis_type,
            query=query,
            results=results,
            start_time=start_time
        )
    
    def analyze_revenue_decline(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze causes of revenue decline"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Filter revenue data
            revenue_data = data[data['amount'] > 0].copy()
            revenue_data['date'] = pd.to_datetime(revenue_data['date'])
            
            # Monthly revenue trends
            monthly_revenue = revenue_data.groupby(revenue_data['date'].dt.to_period('M'))['amount'].sum()
            monthly_growth = monthly_revenue.pct_change().fillna(0) * 100
            
            # Identify declining periods
            declining_periods = monthly_growth[monthly_growth < -5].index.tolist()
            
            # Root cause analysis
            root_causes = {}
            
            # Customer activity analysis
            monthly_customers = revenue_data.groupby(revenue_data['date'].dt.to_period('M'))['party_name'].nunique()
            customer_decline = monthly_customers.pct_change().fillna(0) * 100
            
            # Transaction count analysis
            monthly_transactions = revenue_data.groupby(revenue_data['date'].dt.to_period('M')).size()
            transaction_decline = monthly_transactions.pct_change().fillna(0) * 100
            
            # Average transaction value analysis
            monthly_avg_value = revenue_data.groupby(revenue_data['date'].dt.to_period('M'))['amount'].mean()
            avg_value_decline = monthly_avg_value.pct_change().fillna(0) * 100
            
            # Identify primary causes
            root_causes['customer_loss'] = len(customer_decline[customer_decline < -10])
            root_causes['transaction_frequency_drop'] = len(transaction_decline[transaction_decline < -10])
            root_causes['avg_value_drop'] = len(avg_value_decline[avg_value_decline < -10])
            
            return {
                "revenue_decline_periods": len(declining_periods),
                "total_revenue_decline": round(monthly_growth.min(), 2),
                "root_causes": root_causes,
                "declining_periods": [str(period) for period in declining_periods],
                "decline_analysis": {
                    "customer_impact": round(customer_decline.min(), 2),
                    "transaction_impact": round(transaction_decline.min(), 2),
                    "avg_value_impact": round(avg_value_decline.min(), 2)
                },
                "recommendations": self._generate_revenue_decline_recommendations(root_causes)
            }
            
        except Exception as e:
            return {"error": f"Revenue decline analysis failed: {str(e)}"}
    
    def analyze_customer_churn(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze customer churn patterns and causes"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Customer transaction analysis
            customer_data = data[data['amount'] > 0].copy()
            customer_data['date'] = pd.to_datetime(customer_data['date'])
            
            # Calculate customer metrics
            customer_metrics = customer_data.groupby('party_name').agg({
                'date': ['min', 'max', 'count'],
                'amount': ['sum', 'mean'],
                'voucher_number': 'nunique'
            })
            
            # Flatten column names
            customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns]
            
            # Calculate days since last transaction
            current_date = customer_data['date'].max()
            customer_metrics['days_since_last'] = (current_date - pd.to_datetime(customer_metrics['date_max'])).dt.days
            
            # Identify churned customers (no activity in last 90 days)
            churn_threshold = params.get('churn_threshold', 90)
            churned_customers = customer_metrics[customer_metrics['days_since_last'] > churn_threshold]
            
            # Analyze churn patterns
            churn_analysis = {
                'high_value_churn': len(churned_customers[churned_customers['amount_sum'] > customer_metrics['amount_sum'].quantile(0.8)]),
                'frequent_customer_churn': len(churned_customers[churned_customers['voucher_number_nunique'] > customer_metrics['voucher_number_nunique'].quantile(0.8)]),
                'recent_customer_churn': len(churned_customers[churned_customers['days_since_last'] <= 120])
            }
            
            return {
                "total_customers": len(customer_metrics),
                "churned_customers": len(churned_customers),
                "churn_rate": round(len(churned_customers) / len(customer_metrics) * 100, 2),
                "churn_analysis": churn_analysis,
                "churn_patterns": {
                    "avg_days_since_last": round(churned_customers['days_since_last'].mean(), 2),
                    "lost_revenue": round(churned_customers['amount_sum'].sum(), 2),
                    "avg_lost_customer_value": round(churned_customers['amount_sum'].mean(), 2)
                },
                "recommendations": self._generate_churn_recommendations(churn_analysis)
            }
            
        except Exception as e:
            return {"error": f"Customer churn analysis failed: {str(e)}"}
    
    def analyze_sales_performance_issues(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sales performance issues and bottlenecks"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Performance metrics
            sales_data = data[data['amount'] > 0].copy()
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            
            # Daily performance analysis
            daily_performance = sales_data.groupby(sales_data['date'].dt.date).agg({
                'amount': ['sum', 'count', 'mean'],
                'party_name': 'nunique'
            })
            
            # Identify performance issues
            daily_revenue = daily_performance[('amount', 'sum')]
            daily_transactions = daily_performance[('amount', 'count')]
            
            # Statistical analysis
            revenue_mean = daily_revenue.mean()
            revenue_std = daily_revenue.std()
            
            # Low performance days
            low_performance_days = daily_revenue[daily_revenue < (revenue_mean - revenue_std)]
            
            # Analyze causes
            performance_issues = {
                'low_revenue_days': len(low_performance_days),
                'avg_low_day_revenue': round(low_performance_days.mean(), 2),
                'revenue_volatility': round(revenue_std / revenue_mean * 100, 2)
            }
            
            return {
                "performance_issues": performance_issues,
                "low_performance_analysis": {
                    "total_low_days": len(low_performance_days),
                    "percentage_low_days": round(len(low_performance_days) / len(daily_revenue) * 100, 2),
                    "revenue_impact": round((revenue_mean - low_performance_days.mean()) * len(low_performance_days), 2)
                },
                "recommendations": self._generate_performance_recommendations(performance_issues)
            }
            
        except Exception as e:
            return {"error": f"Sales performance analysis failed: {str(e)}"}
    
    def analyze_transaction_anomalies(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and analyze transaction anomalies"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Transaction anomaly detection
            transaction_data = data[data['amount'] > 0].copy()
            
            # Statistical anomaly detection using Z-score
            z_scores = np.abs(stats.zscore(transaction_data['amount']))
            anomaly_threshold = params.get('anomaly_threshold', 3)
            
            anomalies = transaction_data[z_scores > anomaly_threshold]
            
            # Analyze anomaly patterns
            anomaly_analysis = {
                'high_value_anomalies': len(anomalies[anomalies['amount'] > transaction_data['amount'].quantile(0.99)]),
                'low_value_anomalies': len(anomalies[anomalies['amount'] < transaction_data['amount'].quantile(0.01)]),
                'customer_anomalies': anomalies['party_name'].nunique()
            }
            
            return {
                "total_transactions": len(transaction_data),
                "anomalous_transactions": len(anomalies),
                "anomaly_rate": round(len(anomalies) / len(transaction_data) * 100, 2),
                "anomaly_analysis": anomaly_analysis,
                "anomaly_impact": {
                    "total_anomaly_value": round(anomalies['amount'].sum(), 2),
                    "avg_anomaly_value": round(anomalies['amount'].mean(), 2),
                    "max_anomaly_value": round(anomalies['amount'].max(), 2)
                },
                "recommendations": self._generate_anomaly_recommendations(anomaly_analysis)
            }
            
        except Exception as e:
            return {"error": f"Transaction anomaly analysis failed: {str(e)}"}
    
    def analyze_product_underperformance(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze underperforming products and their causes"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Product performance analysis
            product_data = data[(data['amount'] > 0) & (data['item'].notna())].copy()
            
            if len(product_data) == 0:
                return {"error": "No product data available"}
            
            # Product metrics
            product_metrics = product_data.groupby('item').agg({
                'amount': ['sum', 'count', 'mean'],
                'quantity': ['sum', 'mean'],
                'rate': ['mean', 'std']
            }).round(2)
            
            # Flatten column names
            product_metrics.columns = ['_'.join(col).strip() for col in product_metrics.columns]
            
            # Identify underperforming products
            revenue_threshold = product_metrics['amount_sum'].quantile(0.25)
            frequency_threshold = product_metrics['amount_count'].quantile(0.25)
            
            underperforming = product_metrics[
                (product_metrics['amount_sum'] <= revenue_threshold) |
                (product_metrics['amount_count'] <= frequency_threshold)
            ]
            
            # Analyze causes
            underperformance_causes = {
                'low_revenue_products': len(underperforming[underperforming['amount_sum'] <= revenue_threshold]),
                'low_frequency_products': len(underperforming[underperforming['amount_count'] <= frequency_threshold]),
                'high_price_volatility': len(underperforming[underperforming['rate_std'] > product_metrics['rate_std'].median()])
            }
            
            return {
                "total_products": len(product_metrics),
                "underperforming_products": len(underperforming),
                "underperformance_rate": round(len(underperforming) / len(product_metrics) * 100, 2),
                "underperformance_causes": underperformance_causes,
                "impact_analysis": {
                    "lost_revenue_potential": round((product_metrics['amount_sum'].median() - underperforming['amount_sum'].mean()) * len(underperforming), 2),
                    "avg_underperformer_revenue": round(underperforming['amount_sum'].mean(), 2)
                },
                "recommendations": self._generate_product_recommendations(underperformance_causes)
            }
            
        except Exception as e:
            return {"error": f"Product underperformance analysis failed: {str(e)}"}
    
    def analyze_seasonal_variance(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seasonal variances and their causes"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Seasonal analysis
            seasonal_data = data[data['amount'] > 0].copy()
            seasonal_data['date'] = pd.to_datetime(seasonal_data['date'])
            seasonal_data['month'] = seasonal_data['date'].dt.month
            seasonal_data['quarter'] = seasonal_data['date'].dt.quarter
            
            # Monthly variance
            monthly_revenue = seasonal_data.groupby('month')['amount'].sum()
            monthly_variance = monthly_revenue.std() / monthly_revenue.mean() * 100
            
            # Quarterly variance
            quarterly_revenue = seasonal_data.groupby('quarter')['amount'].sum()
            quarterly_variance = quarterly_revenue.std() / quarterly_revenue.mean() * 100
            
            # Identify seasonal patterns
            peak_month = monthly_revenue.idxmax()
            low_month = monthly_revenue.idxmin()
            
            variance_analysis = {
                'high_seasonal_variance': monthly_variance > 30,
                'peak_to_low_ratio': monthly_revenue.max() / monthly_revenue.min(),
                'consistent_quarters': len(quarterly_revenue[quarterly_revenue > quarterly_revenue.mean() * 0.8])
            }
            
            return {
                "seasonal_variance": {
                    "monthly_variance": round(monthly_variance, 2),
                    "quarterly_variance": round(quarterly_variance, 2),
                    "peak_month": peak_month,
                    "low_month": low_month
                },
                "variance_analysis": variance_analysis,
                "seasonal_patterns": {
                    "monthly_revenue": monthly_revenue.to_dict(),
                    "quarterly_revenue": quarterly_revenue.to_dict()
                },
                "recommendations": self._generate_seasonal_recommendations(variance_analysis)
            }
            
        except Exception as e:
            return {"error": f"Seasonal variance analysis failed: {str(e)}"}
    
    def analyze_customer_behavior_changes(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze changes in customer behavior patterns"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Customer behavior analysis
            customer_data = data[data['amount'] > 0].copy()
            customer_data['date'] = pd.to_datetime(customer_data['date'])
            
            # Split data into two periods for comparison
            mid_date = customer_data['date'].quantile(0.5)
            early_period = customer_data[customer_data['date'] <= mid_date]
            late_period = customer_data[customer_data['date'] > mid_date]
            
            # Calculate metrics for each period
            early_metrics = early_period.groupby('party_name')['amount'].agg(['sum', 'count', 'mean'])
            late_metrics = late_period.groupby('party_name')['amount'].agg(['sum', 'count', 'mean'])
            
            # Identify behavior changes
            common_customers = set(early_metrics.index) & set(late_metrics.index)
            
            if len(common_customers) == 0:
                return {"error": "No common customers found for comparison"}
            
            behavior_changes = {}
            for customer in common_customers:
                early_avg = early_metrics.loc[customer, 'mean']
                late_avg = late_metrics.loc[customer, 'mean']
                
                change_pct = (late_avg - early_avg) / early_avg * 100 if early_avg > 0 else 0
                
                if abs(change_pct) > 20:  # Significant change threshold
                    behavior_changes[customer] = change_pct
            
            # Categorize changes
            increasing_customers = len([c for c in behavior_changes.values() if c > 20])
            decreasing_customers = len([c for c in behavior_changes.values() if c < -20])
            
            return {
                "total_customers_analyzed": len(common_customers),
                "customers_with_behavior_changes": len(behavior_changes),
                "behavior_change_analysis": {
                    "increasing_spend_customers": increasing_customers,
                    "decreasing_spend_customers": decreasing_customers,
                    "avg_increase": round(np.mean([c for c in behavior_changes.values() if c > 0]), 2),
                    "avg_decrease": round(np.mean([c for c in behavior_changes.values() if c < 0]), 2)
                },
                "recommendations": self._generate_behavior_change_recommendations({
                    'increasing': increasing_customers,
                    'decreasing': decreasing_customers
                })
            }
            
        except Exception as e:
            return {"error": f"Customer behavior analysis failed: {str(e)}"}
    
    def analyze_voucher_performance_issues(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze voucher type performance issues"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Voucher performance analysis
            voucher_metrics = data[data['amount'] > 0].groupby('voucher_type').agg({
                'amount': ['sum', 'count', 'mean'],
                'party_name': 'nunique'
            }).round(2)
            
            # Flatten column names
            voucher_metrics.columns = ['_'.join(col).strip() for col in voucher_metrics.columns]
            
            # Identify underperforming voucher types
            avg_revenue = voucher_metrics['amount_sum'].mean()
            avg_frequency = voucher_metrics['amount_count'].mean()
            
            underperforming_vouchers = voucher_metrics[
                (voucher_metrics['amount_sum'] < avg_revenue * 0.5) |
                (voucher_metrics['amount_count'] < avg_frequency * 0.5)
            ]
            
            performance_issues = {
                'low_revenue_vouchers': len(underperforming_vouchers[underperforming_vouchers['amount_sum'] < avg_revenue * 0.5]),
                'low_frequency_vouchers': len(underperforming_vouchers[underperforming_vouchers['amount_count'] < avg_frequency * 0.5]),
                'total_underperforming': len(underperforming_vouchers)
            }
            
            return {
                "voucher_performance_issues": performance_issues,
                "underperforming_vouchers": underperforming_vouchers.to_dict('index'),
                "impact_analysis": {
                    "lost_revenue": round((avg_revenue - underperforming_vouchers['amount_sum'].mean()) * len(underperforming_vouchers), 2),
                    "efficiency_loss": round((avg_frequency - underperforming_vouchers['amount_count'].mean()) / avg_frequency * 100, 2)
                },
                "recommendations": self._generate_voucher_recommendations(performance_issues)
            }
            
        except Exception as e:
            return {"error": f"Voucher performance analysis failed: {str(e)}"}
    
    def analyze_general_sales_diagnostics(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General sales diagnostic analysis"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # General diagnostics
            sales_data = data[data['amount'] > 0]
            
            # Data quality issues
            quality_issues = {
                'missing_customer_names': data['party_name'].isna().sum(),
                'zero_amounts': len(data[data['amount'] == 0]),
                'negative_amounts': len(data[data['amount'] < 0]),
                'missing_dates': data['date'].isna().sum()
            }
            
            return {
                "total_records": len(data),
                "sales_records": len(sales_data),
                "data_quality_issues": quality_issues,
                "general_patterns": {
                    "avg_transaction_value": round(sales_data['amount'].mean(), 2),
                    "transaction_volatility": round(sales_data['amount'].std() / sales_data['amount'].mean() * 100, 2),
                    "customer_concentration": round(sales_data['party_name'].nunique() / len(sales_data) * 100, 2)
                }
            }
            
        except Exception as e:
            return {"error": f"General sales diagnostics failed: {str(e)}"}
    
    def generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diagnostic analysis results"""
        if 'recommendations' in analysis_results:
            return analysis_results['recommendations']
        return []
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the type of diagnostic analysis based on query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['revenue', 'decline', 'drop', 'decrease']):
            return 'revenue_decline_analysis'
        elif any(word in query_lower for word in ['churn', 'lost', 'leaving', 'retention']):
            return 'customer_churn_analysis'
        elif any(word in query_lower for word in ['performance', 'issues', 'problems', 'bottleneck']):
            return 'sales_performance_issues'
        elif any(word in query_lower for word in ['anomaly', 'unusual', 'outlier', 'irregular']):
            return 'transaction_anomaly_detection'
        elif any(word in query_lower for word in ['product', 'item', 'underperform', 'poor']):
            return 'product_underperformance_analysis'
        elif any(word in query_lower for word in ['seasonal', 'variance', 'fluctuation', 'inconsistent']):
            return 'seasonal_variance_analysis'
        elif any(word in query_lower for word in ['behavior', 'change', 'pattern', 'shift']):
            return 'customer_behavior_changes'
        elif any(word in query_lower for word in ['voucher', 'type', 'category', 'method']):
            return 'voucher_type_performance_issues'
        else:
            return 'general_diagnostics'
    
    # Recommendation generators
    def _generate_revenue_decline_recommendations(self, root_causes: Dict[str, int]) -> List[str]:
        """Generate recommendations for revenue decline"""
        recommendations = []
        
        if root_causes['customer_loss'] > 0:
            recommendations.append("Implement customer retention strategies and loyalty programs")
        
        if root_causes['transaction_frequency_drop'] > 0:
            recommendations.append("Increase marketing efforts and promotional campaigns")
        
        if root_causes['avg_value_drop'] > 0:
            recommendations.append("Review pricing strategy and upselling opportunities")
        
        recommendations.append("Conduct detailed customer feedback analysis")
        
        return recommendations
    
    def _generate_churn_recommendations(self, churn_analysis: Dict[str, int]) -> List[str]:
        """Generate recommendations for customer churn"""
        recommendations = []
        
        if churn_analysis['high_value_churn'] > 0:
            recommendations.append("Implement VIP customer retention program")
        
        if churn_analysis['frequent_customer_churn'] > 0:
            recommendations.append("Develop personalized re-engagement campaigns")
        
        recommendations.append("Implement customer health scoring system")
        recommendations.append("Create win-back campaigns for churned customers")
        
        return recommendations
    
    def _generate_performance_recommendations(self, issues: Dict[str, Any]) -> List[str]:
        """Generate recommendations for performance issues"""
        recommendations = []
        
        if issues['low_revenue_days'] > 0:
            recommendations.append("Analyze and address factors causing low-performance days")
        
        if issues['revenue_volatility'] > 30:
            recommendations.append("Implement revenue smoothing strategies")
        
        recommendations.append("Establish performance monitoring dashboards")
        recommendations.append("Create action plans for performance improvement")
        
        return recommendations
    
    def _generate_anomaly_recommendations(self, anomaly_analysis: Dict[str, int]) -> List[str]:
        """Generate recommendations for transaction anomalies"""
        recommendations = []
        
        if anomaly_analysis['high_value_anomalies'] > 0:
            recommendations.append("Implement fraud detection systems for high-value transactions")
        
        if anomaly_analysis['customer_anomalies'] > 0:
            recommendations.append("Review and validate unusual customer transactions")
        
        recommendations.append("Establish transaction monitoring alerts")
        recommendations.append("Create approval workflows for anomalous transactions")
        
        return recommendations
    
    def _generate_product_recommendations(self, causes: Dict[str, int]) -> List[str]:
        """Generate recommendations for product underperformance"""
        recommendations = []
        
        if causes['low_revenue_products'] > 0:
            recommendations.append("Review pricing and positioning of underperforming products")
        
        if causes['low_frequency_products'] > 0:
            recommendations.append("Increase marketing focus on slow-moving products")
        
        recommendations.append("Consider product portfolio optimization")
        recommendations.append("Implement product performance monitoring")
        
        return recommendations
    
    def _generate_seasonal_recommendations(self, variance: Dict[str, Any]) -> List[str]:
        """Generate recommendations for seasonal variance"""
        recommendations = []
        
        if variance['high_seasonal_variance']:
            recommendations.append("Develop seasonal demand planning strategies")
        
        recommendations.append("Implement inventory management for seasonal patterns")
        recommendations.append("Create targeted seasonal marketing campaigns")
        
        return recommendations
    
    def _generate_behavior_change_recommendations(self, changes: Dict[str, int]) -> List[str]:
        """Generate recommendations for behavior changes"""
        recommendations = []
        
        if changes['increasing'] > 0:
            recommendations.append("Leverage positive behavior changes to increase customer lifetime value")
        
        if changes['decreasing'] > 0:
            recommendations.append("Implement intervention strategies for declining customers")
        
        recommendations.append("Create behavior-based customer segmentation")
        recommendations.append("Develop personalized engagement strategies")
        
        return recommendations
    
    def _generate_voucher_recommendations(self, issues: Dict[str, int]) -> List[str]:
        """Generate recommendations for voucher performance"""
        recommendations = []
        
        if issues['low_revenue_vouchers'] > 0:
            recommendations.append("Optimize or discontinue underperforming voucher types")
        
        if issues['low_frequency_vouchers'] > 0:
            recommendations.append("Increase promotion of underutilized voucher types")
        
        recommendations.append("Review voucher type effectiveness and ROI")
        recommendations.append("Implement voucher performance tracking")
        
        return recommendations