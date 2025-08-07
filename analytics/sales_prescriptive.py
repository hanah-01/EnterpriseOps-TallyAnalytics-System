"""
Sales Prescriptive Analytics - "What should we do?"
Optimization strategies and actionable recommendations for sales performance
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from .base import AnalyticsBase, AnalyticsResponse


class SalesPrescriptiveAnalytics(AnalyticsBase):
    """Prescriptive analytics for sales optimization and strategic recommendations"""
    
    def __init__(self, agent_type: str = "sales"):
        super().__init__("SalesPrescriptiveAnalytics", agent_type)
        self.supported_queries = [
            'revenue_optimization',
            'customer_retention_strategy',
            'pricing_optimization',
            'sales_resource_allocation',
            'customer_segmentation_strategy',
            'product_portfolio_optimization',
            'sales_process_optimization',
            'market_expansion_strategy'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform prescriptive analytics on sales data"""
        start_time = datetime.now()
        
        # Determine specific analysis type
        analysis_type = self.classify_analysis_type(query)
        
        # Perform analysis based on type
        if analysis_type == 'revenue_optimization':
            results = self.optimize_revenue_strategies(data, params)
        elif analysis_type == 'customer_retention_strategy':
            results = self.optimize_customer_retention(data, params)
        elif analysis_type == 'pricing_optimization':
            results = self.optimize_pricing_strategy(data, params)
        elif analysis_type == 'sales_resource_allocation':
            results = self.optimize_sales_resources(data, params)
        elif analysis_type == 'customer_segmentation_strategy':
            results = self.optimize_customer_segmentation(data, params)
        elif analysis_type == 'product_portfolio_optimization':
            results = self.optimize_product_portfolio(data, params)
        elif analysis_type == 'sales_process_optimization':
            results = self.optimize_sales_processes(data, params)
        elif analysis_type == 'market_expansion_strategy':
            results = self.optimize_market_expansion(data, params)
        else:
            results = self.general_sales_optimization(data, params)
        
        return self.prepare_response(
            analytics_type=analysis_type,
            query=query,
            results=results,
            start_time=start_time
        )
    
    def optimize_revenue_strategies(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize revenue generation strategies"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Revenue optimization analysis
            revenue_data = data[data['amount'] > 0].copy()
            revenue_data['date'] = pd.to_datetime(revenue_data['date'])
            
            # Analyze revenue drivers
            customer_revenue = revenue_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean'],
                'voucher_number': 'nunique'
            })
            
            # Flatten column names
            customer_revenue.columns = ['_'.join(col).strip() for col in customer_revenue.columns]
            
            # Revenue optimization strategies
            optimization_strategies = {}
            
            # 1. Customer Value Optimization
            high_value_customers = customer_revenue[
                customer_revenue['amount_sum'] > customer_revenue['amount_sum'].quantile(0.8)
            ]
            
            # 2. Transaction Frequency Optimization
            high_frequency_customers = customer_revenue[
                customer_revenue['voucher_number_nunique'] > customer_revenue['voucher_number_nunique'].quantile(0.8)
            ]
            
            # 3. Average Transaction Value Optimization
            high_avg_value_customers = customer_revenue[
                customer_revenue['amount_mean'] > customer_revenue['amount_mean'].quantile(0.8)
            ]
            
            # Calculate optimization potential
            current_revenue = customer_revenue['amount_sum'].sum()
            
            # Potential from increasing transaction frequency
            freq_optimization_potential = len(customer_revenue) * customer_revenue['amount_mean'].mean() * 0.2  # 20% increase
            
            # Potential from increasing average transaction value
            value_optimization_potential = current_revenue * 0.15  # 15% increase
            
            # Potential from customer retention
            retention_optimization_potential = current_revenue * 0.1  # 10% increase
            
            optimization_strategies = {
                'customer_value_focus': {
                    'target_customers': len(high_value_customers),
                    'current_contribution': round(high_value_customers['amount_sum'].sum(), 2),
                    'optimization_potential': round(high_value_customers['amount_sum'].sum() * 0.25, 2),
                    'strategy': 'VIP programs and personalized service'
                },
                'frequency_optimization': {
                    'target_customers': len(customer_revenue) - len(high_frequency_customers),
                    'potential_revenue': round(freq_optimization_potential, 2),
                    'strategy': 'Loyalty programs and regular engagement'
                },
                'value_optimization': {
                    'target_customers': len(customer_revenue) - len(high_avg_value_customers),
                    'potential_revenue': round(value_optimization_potential, 2),
                    'strategy': 'Upselling and cross-selling programs'
                },
                'retention_optimization': {
                    'potential_revenue': round(retention_optimization_potential, 2),
                    'strategy': 'Churn prevention and win-back campaigns'
                }
            }
            
            # Overall optimization potential
            total_optimization_potential = (
                optimization_strategies['customer_value_focus']['optimization_potential'] +
                optimization_strategies['frequency_optimization']['potential_revenue'] +
                optimization_strategies['value_optimization']['potential_revenue'] +
                optimization_strategies['retention_optimization']['potential_revenue']
            )
            
            return {
                "current_revenue": round(current_revenue, 2),
                "optimization_strategies": optimization_strategies,
                "total_optimization_potential": round(total_optimization_potential, 2),
                "roi_potential": round(total_optimization_potential / current_revenue * 100, 2),
                "recommended_actions": self._generate_revenue_optimization_actions(optimization_strategies)
            }
            
        except Exception as e:
            return {"error": f"Revenue optimization failed: {str(e)}"}
    
    def optimize_customer_retention(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize customer retention strategies"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Customer retention analysis
            customer_data = data[data['amount'] > 0].copy()
            customer_data['date'] = pd.to_datetime(customer_data['date'])
            
            # Calculate customer metrics
            current_date = customer_data['date'].max()
            customer_metrics = customer_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean'],
                'date': ['min', 'max'],
                'voucher_number': 'nunique'
            })
            
            # Flatten column names
            customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns]
            
            # Calculate retention metrics
            customer_metrics['days_since_last'] = (current_date - pd.to_datetime(customer_metrics['date_max'])).dt.days
            customer_metrics['customer_lifetime'] = (
                pd.to_datetime(customer_metrics['date_max']) - 
                pd.to_datetime(customer_metrics['date_min'])
            ).dt.days
            
            # Retention risk segments
            churn_threshold = params.get('churn_threshold', 90)
            at_risk_customers = customer_metrics[customer_metrics['days_since_last'] > churn_threshold]
            
            # High-value at-risk customers
            high_value_at_risk = at_risk_customers[
                at_risk_customers['amount_sum'] > customer_metrics['amount_sum'].quantile(0.8)
            ]
            
            # Retention optimization strategies
            retention_strategies = {
                'immediate_intervention': {
                    'target_customers': len(high_value_at_risk),
                    'revenue_at_risk': round(high_value_at_risk['amount_sum'].sum(), 2),
                    'strategy': 'Personal outreach and exclusive offers',
                    'expected_retention_rate': 0.7,
                    'expected_revenue_saved': round(high_value_at_risk['amount_sum'].sum() * 0.7, 2)
                },
                'proactive_engagement': {
                    'target_customers': len(at_risk_customers) - len(high_value_at_risk),
                    'revenue_at_risk': round(at_risk_customers['amount_sum'].sum() - high_value_at_risk['amount_sum'].sum(), 2),
                    'strategy': 'Automated re-engagement campaigns',
                    'expected_retention_rate': 0.4,
                    'expected_revenue_saved': round((at_risk_customers['amount_sum'].sum() - high_value_at_risk['amount_sum'].sum()) * 0.4, 2)
                },
                'loyalty_program': {
                    'target_customers': len(customer_metrics) - len(at_risk_customers),
                    'strategy': 'Points-based loyalty program',
                    'expected_retention_improvement': 0.15,
                    'expected_revenue_increase': round(customer_metrics['amount_sum'].sum() * 0.15, 2)
                }
            }
            
            # Calculate overall retention ROI
            total_revenue_at_risk = at_risk_customers['amount_sum'].sum()
            total_expected_savings = (
                retention_strategies['immediate_intervention']['expected_revenue_saved'] +
                retention_strategies['proactive_engagement']['expected_revenue_saved'] +
                retention_strategies['loyalty_program']['expected_revenue_increase']
            )
            
            return {
                "retention_analysis": {
                    "total_customers": len(customer_metrics),
                    "at_risk_customers": len(at_risk_customers),
                    "high_value_at_risk": len(high_value_at_risk),
                    "total_revenue_at_risk": round(total_revenue_at_risk, 2)
                },
                "retention_strategies": retention_strategies,
                "optimization_impact": {
                    "total_expected_savings": round(total_expected_savings, 2),
                    "retention_roi": round(total_expected_savings / total_revenue_at_risk * 100, 2) if total_revenue_at_risk > 0 else 0
                },
                "recommended_actions": self._generate_retention_actions(retention_strategies)
            }
            
        except Exception as e:
            return {"error": f"Customer retention optimization failed: {str(e)}"}
    
    def optimize_pricing_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize pricing strategies"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Pricing optimization analysis
            pricing_data = data[(data['amount'] > 0) & (data['rate'].notna())].copy()
            
            if len(pricing_data) == 0:
                return {"error": "No pricing data available"}
            
            # Product pricing analysis
            product_pricing = pricing_data.groupby('item').agg({
                'rate': ['mean', 'min', 'max', 'std'],
                'amount': ['sum', 'count'],
                'quantity': 'sum'
            }).round(2)
            
            # Flatten column names
            product_pricing.columns = ['_'.join(col).strip() for col in product_pricing.columns]
            
            # Price elasticity analysis (simplified)
            product_pricing['price_volatility'] = product_pricing['rate_std'] / product_pricing['rate_mean']
            product_pricing['revenue_per_unit'] = product_pricing['amount_sum'] / product_pricing['quantity_sum']
            
            # Pricing optimization opportunities
            pricing_opportunities = {}
            
            # 1. Price standardization opportunities
            high_volatility_products = product_pricing[product_pricing['price_volatility'] > 0.2]
            
            # 2. Premium pricing opportunities
            high_demand_products = product_pricing[
                product_pricing['amount_count'] > product_pricing['amount_count'].quantile(0.8)
            ]
            
            # 3. Value pricing opportunities
            high_revenue_products = product_pricing[
                product_pricing['amount_sum'] > product_pricing['amount_sum'].quantile(0.8)
            ]
            
            # Calculate optimization potential
            pricing_opportunities = {
                'price_standardization': {
                    'target_products': len(high_volatility_products),
                    'current_volatility': round(high_volatility_products['price_volatility'].mean(), 2),
                    'potential_revenue_stabilization': round(high_volatility_products['amount_sum'].sum() * 0.05, 2),
                    'strategy': 'Implement consistent pricing policies'
                },
                'premium_pricing': {
                    'target_products': len(high_demand_products),
                    'current_avg_price': round(high_demand_products['rate_mean'].mean(), 2),
                    'potential_price_increase': round(high_demand_products['rate_mean'].mean() * 0.1, 2),
                    'potential_revenue_increase': round(high_demand_products['amount_sum'].sum() * 0.1, 2),
                    'strategy': 'Increase prices for high-demand products'
                },
                'value_optimization': {
                    'target_products': len(high_revenue_products),
                    'optimization_potential': round(high_revenue_products['amount_sum'].sum() * 0.08, 2),
                    'strategy': 'Optimize value-based pricing'
                }
            }
            
            # Overall pricing optimization potential
            total_pricing_potential = (
                pricing_opportunities['price_standardization']['potential_revenue_stabilization'] +
                pricing_opportunities['premium_pricing']['potential_revenue_increase'] +
                pricing_opportunities['value_optimization']['optimization_potential']
            )
            
            return {
                "current_pricing_analysis": {
                    "total_products": len(product_pricing),
                    "avg_price": round(product_pricing['rate_mean'].mean(), 2),
                    "price_range": {
                        "min": round(product_pricing['rate_min'].min(), 2),
                        "max": round(product_pricing['rate_max'].max(), 2)
                    },
                    "avg_volatility": round(product_pricing['price_volatility'].mean(), 2)
                },
                "pricing_opportunities": pricing_opportunities,
                "total_optimization_potential": round(total_pricing_potential, 2),
                "recommended_actions": self._generate_pricing_actions(pricing_opportunities)
            }
            
        except Exception as e:
            return {"error": f"Pricing optimization failed: {str(e)}"}
    
    def optimize_sales_resources(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize sales resource allocation"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Sales resource optimization
            sales_data = data[data['amount'] > 0].copy()
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            
            # Customer profitability analysis
            customer_profitability = sales_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean'],
                'voucher_number': 'nunique'
            })
            
            # Flatten column names
            customer_profitability.columns = ['_'.join(col).strip() for col in customer_profitability.columns]
            
            # Customer segmentation for resource allocation
            high_value_customers = customer_profitability[
                customer_profitability['amount_sum'] > customer_profitability['amount_sum'].quantile(0.8)
            ]
            
            medium_value_customers = customer_profitability[
                (customer_profitability['amount_sum'] > customer_profitability['amount_sum'].quantile(0.4)) &
                (customer_profitability['amount_sum'] <= customer_profitability['amount_sum'].quantile(0.8))
            ]
            
            low_value_customers = customer_profitability[
                customer_profitability['amount_sum'] <= customer_profitability['amount_sum'].quantile(0.4)
            ]
            
            # Resource allocation optimization
            resource_allocation = {
                'high_value_segment': {
                    'customers': len(high_value_customers),
                    'revenue_contribution': round(high_value_customers['amount_sum'].sum(), 2),
                    'recommended_resource_allocation': 0.5,  # 50% of resources
                    'strategy': 'Dedicated account management and personalized service',
                    'expected_growth': 0.25  # 25% growth potential
                },
                'medium_value_segment': {
                    'customers': len(medium_value_customers),
                    'revenue_contribution': round(medium_value_customers['amount_sum'].sum(), 2),
                    'recommended_resource_allocation': 0.3,  # 30% of resources
                    'strategy': 'Regular engagement and upselling',
                    'expected_growth': 0.15  # 15% growth potential
                },
                'low_value_segment': {
                    'customers': len(low_value_customers),
                    'revenue_contribution': round(low_value_customers['amount_sum'].sum(), 2),
                    'recommended_resource_allocation': 0.2,  # 20% of resources
                    'strategy': 'Automated marketing and self-service',
                    'expected_growth': 0.05  # 5% growth potential
                }
            }
            
            # Calculate optimization impact
            current_total_revenue = customer_profitability['amount_sum'].sum()
            optimized_revenue = (
                resource_allocation['high_value_segment']['revenue_contribution'] * 
                (1 + resource_allocation['high_value_segment']['expected_growth']) +
                resource_allocation['medium_value_segment']['revenue_contribution'] * 
                (1 + resource_allocation['medium_value_segment']['expected_growth']) +
                resource_allocation['low_value_segment']['revenue_contribution'] * 
                (1 + resource_allocation['low_value_segment']['expected_growth'])
            )
            
            return {
                "current_resource_analysis": {
                    "total_customers": len(customer_profitability),
                    "total_revenue": round(current_total_revenue, 2),
                    "avg_customer_value": round(customer_profitability['amount_sum'].mean(), 2)
                },
                "resource_allocation": resource_allocation,
                "optimization_impact": {
                    "current_revenue": round(current_total_revenue, 2),
                    "optimized_revenue": round(optimized_revenue, 2),
                    "additional_revenue": round(optimized_revenue - current_total_revenue, 2),
                    "roi_improvement": round((optimized_revenue - current_total_revenue) / current_total_revenue * 100, 2)
                },
                "recommended_actions": self._generate_resource_allocation_actions(resource_allocation)
            }
            
        except Exception as e:
            return {"error": f"Sales resource optimization failed: {str(e)}"}
    
    def optimize_customer_segmentation(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize customer segmentation strategy"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Customer segmentation optimization
            customer_data = data[data['amount'] > 0].copy()
            customer_data['date'] = pd.to_datetime(customer_data['date'])
            
            # RFM Analysis (Recency, Frequency, Monetary)
            current_date = customer_data['date'].max()
            rfm_data = customer_data.groupby('party_name').agg({
                'date': lambda x: (current_date - x.max()).days,  # Recency
                'voucher_number': 'nunique',  # Frequency
                'amount': 'sum'  # Monetary
            })
            
            rfm_data.columns = ['Recency', 'Frequency', 'Monetary']
            
            # Create RFM scores
            rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], 5, labels=[5, 4, 3, 2, 1])
            rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'], 5, labels=[1, 2, 3, 4, 5])
            rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'], 5, labels=[1, 2, 3, 4, 5])
            
            # Combined RFM score
            rfm_data['RFM_Score'] = (
                rfm_data['R_Score'].astype(int) * 100 +
                rfm_data['F_Score'].astype(int) * 10 +
                rfm_data['M_Score'].astype(int)
            )
            
            # Define customer segments
            def rfm_segment(score):
                if score >= 544:
                    return 'Champions'
                elif score >= 444:
                    return 'Loyal Customers'
                elif score >= 343:
                    return 'Potential Loyalists'
                elif score >= 242:
                    return 'New Customers'
                elif score >= 155:
                    return 'Promising'
                elif score >= 155:
                    return 'Need Attention'
                elif score >= 144:
                    return 'About to Sleep'
                elif score >= 114:
                    return 'At Risk'
                elif score >= 111:
                    return 'Cannot Lose Them'
                else:
                    return 'Lost'
            
            rfm_data['Segment'] = rfm_data['RFM_Score'].apply(rfm_segment)
            
            # Segment analysis
            segment_analysis = rfm_data.groupby('Segment').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': ['mean', 'sum', 'count']
            }).round(2)
            
            # Optimization strategies for each segment
            segment_strategies = {
                'Champions': {
                    'customers': len(rfm_data[rfm_data['Segment'] == 'Champions']),
                    'strategy': 'Reward them, ask for referrals',
                    'expected_growth': 0.1
                },
                'Loyal Customers': {
                    'customers': len(rfm_data[rfm_data['Segment'] == 'Loyal Customers']),
                    'strategy': 'Upsell higher value products',
                    'expected_growth': 0.15
                },
                'Potential Loyalists': {
                    'customers': len(rfm_data[rfm_data['Segment'] == 'Potential Loyalists']),
                    'strategy': 'Offer membership or loyalty program',
                    'expected_growth': 0.2
                },
                'At Risk': {
                    'customers': len(rfm_data[rfm_data['Segment'] == 'At Risk']),
                    'strategy': 'Send personalized campaigns, offer discounts',
                    'expected_retention': 0.6
                },
                'Lost': {
                    'customers': len(rfm_data[rfm_data['Segment'] == 'Lost']),
                    'strategy': 'Revive interest with relevant offers',
                    'expected_winback': 0.2
                }
            }
            
            return {
                "rfm_analysis": {
                    "total_customers": len(rfm_data),
                    "segment_distribution": rfm_data['Segment'].value_counts().to_dict(),
                    "avg_rfm_score": round(rfm_data['RFM_Score'].mean(), 2)
                },
                "segment_strategies": segment_strategies,
                "segment_performance": segment_analysis.to_dict(),
                "recommended_actions": self._generate_segmentation_actions(segment_strategies)
            }
            
        except Exception as e:
            return {"error": f"Customer segmentation optimization failed: {str(e)}"}
    
    def optimize_product_portfolio(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize product portfolio strategy"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Product portfolio optimization
            product_data = data[(data['amount'] > 0) & (data['item'].notna())].copy()
            
            if len(product_data) == 0:
                return {"error": "No product data available"}
            
            # Product performance analysis
            product_performance = product_data.groupby('item').agg({
                'amount': ['sum', 'count', 'mean'],
                'quantity': 'sum',
                'rate': 'mean'
            }).round(2)
            
            # Flatten column names
            product_performance.columns = ['_'.join(col).strip() for col in product_performance.columns]
            
            # Portfolio optimization using BCG matrix approach
            # Growth = transaction frequency, Market Share = revenue contribution
            total_revenue = product_performance['amount_sum'].sum()
            total_transactions = product_performance['amount_count'].sum()
            
            product_performance['revenue_share'] = product_performance['amount_sum'] / total_revenue
            product_performance['transaction_share'] = product_performance['amount_count'] / total_transactions
            
            # Categorize products
            high_revenue_threshold = product_performance['revenue_share'].quantile(0.5)
            high_frequency_threshold = product_performance['transaction_share'].quantile(0.5)
            
            def categorize_product(revenue_share, frequency_share):
                if revenue_share >= high_revenue_threshold and frequency_share >= high_frequency_threshold:
                    return 'Stars'
                elif revenue_share >= high_revenue_threshold and frequency_share < high_frequency_threshold:
                    return 'Cash Cows'
                elif revenue_share < high_revenue_threshold and frequency_share >= high_frequency_threshold:
                    return 'Question Marks'
                else:
                    return 'Dogs'
            
            product_performance['category'] = product_performance.apply(
                lambda row: categorize_product(row['revenue_share'], row['transaction_share']), 
                axis=1
            )
            
            # Portfolio optimization strategies
            portfolio_strategies = {
                'Stars': {
                    'products': len(product_performance[product_performance['category'] == 'Stars']),
                    'revenue_contribution': round(product_performance[product_performance['category'] == 'Stars']['amount_sum'].sum(), 2),
                    'strategy': 'Invest heavily to maintain market position',
                    'expected_growth': 0.2
                },
                'Cash Cows': {
                    'products': len(product_performance[product_performance['category'] == 'Cash Cows']),
                    'revenue_contribution': round(product_performance[product_performance['category'] == 'Cash Cows']['amount_sum'].sum(), 2),
                    'strategy': 'Maintain and harvest profits',
                    'expected_growth': 0.05
                },
                'Question Marks': {
                    'products': len(product_performance[product_performance['category'] == 'Question Marks']),
                    'revenue_contribution': round(product_performance[product_performance['category'] == 'Question Marks']['amount_sum'].sum(), 2),
                    'strategy': 'Selective investment or divestment',
                    'expected_growth': 0.3
                },
                'Dogs': {
                    'products': len(product_performance[product_performance['category'] == 'Dogs']),
                    'revenue_contribution': round(product_performance[product_performance['category'] == 'Dogs']['amount_sum'].sum(), 2),
                    'strategy': 'Divest or reposition',
                    'expected_growth': -0.1
                }
            }
            
            return {
                "portfolio_analysis": {
                    "total_products": len(product_performance),
                    "category_distribution": product_performance['category'].value_counts().to_dict(),
                    "total_revenue": round(total_revenue, 2)
                },
                "portfolio_strategies": portfolio_strategies,
                "recommended_actions": self._generate_portfolio_actions(portfolio_strategies)
            }
            
        except Exception as e:
            return {"error": f"Product portfolio optimization failed: {str(e)}"}
    
    def optimize_sales_processes(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize sales processes and efficiency"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Sales process optimization
            sales_data = data[data['amount'] > 0].copy()
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            
            # Process efficiency analysis
            daily_performance = sales_data.groupby(sales_data['date'].dt.date).agg({
                'amount': ['sum', 'count', 'mean'],
                'party_name': 'nunique',
                'voucher_type': 'nunique'
            })
            
            # Flatten column names
            daily_performance.columns = ['_'.join(col).strip() for col in daily_performance.columns]
            
            # Identify process optimization opportunities
            process_metrics = {
                'avg_daily_revenue': round(daily_performance['amount_sum'].mean(), 2),
                'avg_daily_transactions': round(daily_performance['amount_count'].mean(), 2),
                'avg_transaction_value': round(daily_performance['amount_mean'].mean(), 2),
                'avg_daily_customers': round(daily_performance['party_name_nunique'].mean(), 2)
            }
            
            # Process optimization strategies
            optimization_strategies = {
                'transaction_efficiency': {
                    'current_avg_transaction_value': process_metrics['avg_transaction_value'],
                    'target_improvement': 0.15,  # 15% improvement
                    'strategy': 'Streamline transaction process and reduce friction',
                    'expected_revenue_increase': round(process_metrics['avg_daily_revenue'] * 0.15 * 365, 2)
                },
                'customer_throughput': {
                    'current_daily_customers': process_metrics['avg_daily_customers'],
                    'target_improvement': 0.2,  # 20% improvement
                    'strategy': 'Optimize customer service and reduce wait times',
                    'expected_revenue_increase': round(process_metrics['avg_daily_revenue'] * 0.2 * 365, 2)
                },
                'process_automation': {
                    'automation_potential': 0.25,  # 25% time saving
                    'strategy': 'Implement automated workflows and systems',
                    'expected_cost_reduction': round(process_metrics['avg_daily_revenue'] * 0.1 * 365, 2)
                }
            }
            
            return {
                "current_process_metrics": process_metrics,
                "optimization_strategies": optimization_strategies,
                "total_optimization_potential": round(
                    optimization_strategies['transaction_efficiency']['expected_revenue_increase'] +
                    optimization_strategies['customer_throughput']['expected_revenue_increase'] +
                    optimization_strategies['process_automation']['expected_cost_reduction'], 2
                ),
                "recommended_actions": self._generate_process_optimization_actions(optimization_strategies)
            }
            
        except Exception as e:
            return {"error": f"Sales process optimization failed: {str(e)}"}
    
    def optimize_market_expansion(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize market expansion strategies"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Market expansion analysis
            market_data = data[data['amount'] > 0].copy()
            
            # Current market analysis
            current_market_metrics = {
                'total_customers': market_data['party_name'].nunique(),
                'total_revenue': round(market_data['amount'].sum(), 2),
                'avg_customer_value': round(market_data.groupby('party_name')['amount'].sum().mean(), 2)
            }
            
            # Expansion opportunities
            expansion_strategies = {
                'customer_acquisition': {
                    'current_customers': current_market_metrics['total_customers'],
                    'target_acquisition': round(current_market_metrics['total_customers'] * 0.3),  # 30% growth
                    'expected_revenue_per_customer': current_market_metrics['avg_customer_value'],
                    'expected_revenue_increase': round(current_market_metrics['total_customers'] * 0.3 * current_market_metrics['avg_customer_value'], 2),
                    'strategy': 'Targeted marketing campaigns and referral programs'
                },
                'geographic_expansion': {
                    'expansion_potential': 0.4,  # 40% market expansion
                    'expected_revenue_increase': round(current_market_metrics['total_revenue'] * 0.4, 2),
                    'strategy': 'Expand to new geographic markets'
                },
                'product_line_expansion': {
                    'cross_sell_potential': 0.25,  # 25% revenue increase
                    'expected_revenue_increase': round(current_market_metrics['total_revenue'] * 0.25, 2),
                    'strategy': 'Introduce complementary products and services'
                }
            }
            
            # Calculate total expansion potential
            total_expansion_potential = (
                expansion_strategies['customer_acquisition']['expected_revenue_increase'] +
                expansion_strategies['geographic_expansion']['expected_revenue_increase'] +
                expansion_strategies['product_line_expansion']['expected_revenue_increase']
            )
            
            return {
                "current_market_metrics": current_market_metrics,
                "expansion_strategies": expansion_strategies,
                "total_expansion_potential": round(total_expansion_potential, 2),
                "market_growth_potential": round(total_expansion_potential / current_market_metrics['total_revenue'] * 100, 2),
                "recommended_actions": self._generate_expansion_actions(expansion_strategies)
            }
            
        except Exception as e:
            return {"error": f"Market expansion optimization failed: {str(e)}"}
    
    def general_sales_optimization(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General sales optimization"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # General optimization
            sales_data = data[data['amount'] > 0]
            
            return {
                "total_sales": round(sales_data['amount'].sum(), 2),
                "total_customers": sales_data['party_name'].nunique(),
                "avg_transaction_value": round(sales_data['amount'].mean(), 2),
                "optimization_potential": round(sales_data['amount'].sum() * 0.15, 2),
                "general_recommendations": [
                    "Focus on high-value customers",
                    "Improve transaction frequency",
                    "Optimize pricing strategy",
                    "Enhance customer retention"
                ]
            }
            
        except Exception as e:
            return {"error": f"General sales optimization failed: {str(e)}"}
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the type of prescriptive analysis based on query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['revenue', 'optimize', 'maximize', 'increase']):
            return 'revenue_optimization'
        elif any(word in query_lower for word in ['retention', 'churn', 'keep', 'loyal']):
            return 'customer_retention_strategy'
        elif any(word in query_lower for word in ['pricing', 'price', 'cost', 'value']):
            return 'pricing_optimization'
        elif any(word in query_lower for word in ['resource', 'allocation', 'team', 'staff']):
            return 'sales_resource_allocation'
        elif any(word in query_lower for word in ['segment', 'customer', 'rfm', 'group']):
            return 'customer_segmentation_strategy'
        elif any(word in query_lower for word in ['product', 'portfolio', 'mix', 'catalog']):
            return 'product_portfolio_optimization'
        elif any(word in query_lower for word in ['process', 'efficiency', 'workflow', 'automation']):
            return 'sales_process_optimization'
        elif any(word in query_lower for word in ['expansion', 'growth', 'market', 'new']):
            return 'market_expansion_strategy'
        else:
            return 'general_optimization'
    
    # Action generators
    def _generate_revenue_optimization_actions(self, strategies: Dict[str, Any]) -> List[str]:
        """Generate revenue optimization actions"""
        actions = []
        
        actions.append(f"Implement VIP program for {strategies['customer_value_focus']['target_customers']} high-value customers")
        actions.append("Develop loyalty program to increase transaction frequency")
        actions.append("Create upselling and cross-selling campaigns")
        actions.append("Implement churn prevention strategies")
        actions.append("Personalize customer experiences based on value segments")
        
        return actions
    
    def _generate_retention_actions(self, strategies: Dict[str, Any]) -> List[str]:
        """Generate retention optimization actions"""
        actions = []
        
        actions.append(f"Immediate outreach to {strategies['immediate_intervention']['target_customers']} high-value at-risk customers")
        actions.append("Launch automated re-engagement email campaigns")
        actions.append("Implement points-based loyalty program")
        actions.append("Create personalized offers based on customer history")
        actions.append("Establish customer health scoring system")
        
        return actions
    
    def _generate_pricing_actions(self, opportunities: Dict[str, Any]) -> List[str]:
        """Generate pricing optimization actions"""
        actions = []
        
        actions.append(f"Standardize pricing for {opportunities['price_standardization']['target_products']} volatile products")
        actions.append(f"Implement 10% price increase for {opportunities['premium_pricing']['target_products']} high-demand products")
        actions.append("Develop value-based pricing strategy")
        actions.append("Create dynamic pricing models")
        actions.append("Implement price testing and optimization")
        
        return actions
    
    def _generate_resource_allocation_actions(self, allocation: Dict[str, Any]) -> List[str]:
        """Generate resource allocation actions"""
        actions = []
        
        actions.append(f"Assign dedicated account managers to {allocation['high_value_segment']['customers']} high-value customers")
        actions.append("Implement tiered service levels based on customer value")
        actions.append("Automate low-value customer interactions")
        actions.append("Create customer success programs")
        actions.append("Optimize sales team territories and assignments")
        
        return actions
    
    def _generate_segmentation_actions(self, strategies: Dict[str, Any]) -> List[str]:
        """Generate segmentation optimization actions"""
        actions = []
        
        actions.append(f"Create exclusive rewards program for {strategies['Champions']['customers']} Champions")
        actions.append("Develop upselling campaigns for Loyal Customers")
        actions.append("Implement membership program for Potential Loyalists")
        actions.append("Launch win-back campaigns for At Risk customers")
        actions.append("Create targeted offers for Lost customers")
        
        return actions
    
    def _generate_portfolio_actions(self, strategies: Dict[str, Any]) -> List[str]:
        """Generate portfolio optimization actions"""
        actions = []
        
        actions.append(f"Increase investment in {strategies['Stars']['products']} Star products")
        actions.append(f"Maintain and harvest {strategies['Cash Cows']['products']} Cash Cow products")
        actions.append(f"Evaluate {strategies['Question Marks']['products']} Question Mark products for investment")
        actions.append(f"Consider divesting {strategies['Dogs']['products']} Dog products")
        actions.append("Implement product performance monitoring")
        
        return actions
    
    def _generate_process_optimization_actions(self, strategies: Dict[str, Any]) -> List[str]:
        """Generate process optimization actions"""
        actions = []
        
        actions.append("Streamline transaction processes to reduce friction")
        actions.append("Implement customer service optimization")
        actions.append("Deploy automation tools for routine tasks")
        actions.append("Create performance dashboards and KPIs")
        actions.append("Establish continuous improvement processes")
        
        return actions
    
    def _generate_expansion_actions(self, strategies: Dict[str, Any]) -> List[str]:
        """Generate market expansion actions"""
        actions = []
        
        actions.append("Launch targeted customer acquisition campaigns")
        actions.append("Develop referral program to leverage existing customers")
        actions.append("Research and enter new geographic markets")
        actions.append("Introduce complementary products and services")
        actions.append("Create strategic partnerships for market expansion")
        
        return actions