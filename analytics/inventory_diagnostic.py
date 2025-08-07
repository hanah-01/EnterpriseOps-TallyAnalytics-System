"""
Inventory Diagnostic Analytics - "Why did it happen?"
Root cause analysis for inventory patterns and anomalies
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from .base import AnalyticsBase, AnalyticsResponse


class InventoryDiagnosticAnalytics(AnalyticsBase):
    """Diagnostic analytics for inventory root cause analysis"""
    
    def __init__(self, agent_type: str = "inventory"):
        super().__init__("InventoryDiagnosticAnalytics", agent_type)
        self.supported_queries = [
            'stockout_analysis',
            'slow_moving_causes',
            'inventory_variance_analysis',
            'demand_volatility_analysis',
            'supplier_performance_impact',
            'seasonal_anomaly_detection',
            'price_fluctuation_analysis',
            'storage_efficiency_analysis'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform diagnostic analytics on inventory data"""
        start_time = datetime.now()
        
        # Determine specific analysis type
        analysis_type = self.classify_analysis_type(query)
        
        # Perform analysis based on type
        if analysis_type == 'stockout_analysis':
            results = self.analyze_stockout_causes(data, params)
        elif analysis_type == 'slow_moving_causes':
            results = self.analyze_slow_moving_causes(data, params)
        elif analysis_type == 'inventory_variance_analysis':
            results = self.analyze_inventory_variance(data, params)
        elif analysis_type == 'demand_volatility_analysis':
            results = self.analyze_demand_volatility(data, params)
        elif analysis_type == 'supplier_performance_impact':
            results = self.analyze_supplier_impact(data, params)
        elif analysis_type == 'seasonal_anomaly_detection':
            results = self.analyze_seasonal_anomalies(data, params)
        elif analysis_type == 'price_fluctuation_analysis':
            results = self.analyze_price_fluctuations(data, params)
        elif analysis_type == 'storage_efficiency_analysis':
            results = self.analyze_storage_efficiency(data, params)
        else:
            results = self.analyze_general_diagnostics(data, params)
        
        return self.prepare_response(
            analytics_type=analysis_type,
            query=query,
            results=results,
            start_time=start_time
        )
    
    def analyze_stockout_causes(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze causes of stockouts and low inventory levels"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            # Calculate stock levels per item
            item_stock = data.groupby('item').agg({
                'quantity': ['sum', 'count', 'std'],
                'amount': 'sum',
                'date': ['min', 'max']
            })
            
            # Identify low stock items (bottom 10% by quantity)
            low_stock_threshold = item_stock[('quantity', 'sum')].quantile(0.1)
            low_stock_items = item_stock[item_stock[('quantity', 'sum')] <= low_stock_threshold]
            
            # Analyze patterns in low stock items
            low_stock_analysis = {}
            for item in low_stock_items.index:
                item_data = data[data['item'] == item]
                
                # Calculate transaction frequency
                date_range = (item_data['date'].max() - item_data['date'].min()).days
                transaction_frequency = len(item_data) / max(date_range, 1)
                
                # Calculate demand volatility
                demand_volatility = item_data['quantity'].std() / abs(item_data['quantity'].mean()) if item_data['quantity'].mean() != 0 else 0
                
                # Check for negative quantities (outbound movements)
                outbound_ratio = len(item_data[item_data['quantity'] < 0]) / len(item_data)
                
                low_stock_analysis[item] = {
                    'current_stock': item_stock.loc[item, ('quantity', 'sum')],
                    'transaction_frequency': round(transaction_frequency, 4),
                    'demand_volatility': round(demand_volatility, 4),
                    'outbound_ratio': round(outbound_ratio, 4),
                    'last_transaction': item_data['date'].max(),
                    'avg_rate': round(item_data['rate'].mean(), 2)
                }
            
            # Identify root causes
            root_causes = {
                'high_demand_volatility': len([k for k, v in low_stock_analysis.items() if v['demand_volatility'] > 1.0]),
                'low_transaction_frequency': len([k for k, v in low_stock_analysis.items() if v['transaction_frequency'] < 0.01]),
                'high_outbound_ratio': len([k for k, v in low_stock_analysis.items() if v['outbound_ratio'] > 0.7]),
                'stale_items': len([k for k, v in low_stock_analysis.items() if 
                                  (datetime.now() - pd.to_datetime(v['last_transaction'])).days > 90])
            }
            
            return {
                'low_stock_items_count': len(low_stock_items),
                'low_stock_threshold': round(low_stock_threshold, 2),
                'low_stock_analysis': low_stock_analysis,
                'root_causes': root_causes,
                'recommendations': self._generate_stockout_recommendations(root_causes)
            }
            
        except Exception as e:
            return {"error": f"Stockout analysis failed: {str(e)}"}
    
    def analyze_slow_moving_causes(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze causes of slow-moving inventory"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            # Calculate movement frequency per item
            item_movement = data.groupby('item').agg({
                'quantity': ['count', 'sum', 'mean'],
                'amount': 'sum',
                'rate': ['mean', 'std'],
                'date': ['min', 'max']
            })
            
            # Calculate days between first and last transaction
            item_movement['days_active'] = (
                pd.to_datetime(item_movement[('date', 'max')]) - 
                pd.to_datetime(item_movement[('date', 'min')])
            ).dt.days
            
            # Calculate movement velocity
            item_movement['movement_velocity'] = item_movement[('quantity', 'count')] / item_movement['days_active'].replace(0, 1)
            
            # Identify slow movers (bottom 20% by movement velocity)
            slow_threshold = item_movement['movement_velocity'].quantile(0.2)
            slow_movers = item_movement[item_movement['movement_velocity'] <= slow_threshold]
            
            # Analyze slow mover characteristics
            slow_mover_analysis = {}
            for item in slow_movers.index:
                item_data = data[data['item'] == item]
                
                # Price analysis
                price_mean = item_data['rate'].mean()
                price_std = item_data['rate'].std()
                price_cv = price_std / price_mean if price_mean != 0 else 0
                
                # Volume analysis
                volume_mean = abs(item_data['quantity'].mean())
                volume_std = item_data['quantity'].std()
                
                # Seasonality check
                item_data['month'] = pd.to_datetime(item_data['date']).dt.month
                monthly_counts = item_data.groupby('month').size()
                seasonality_score = monthly_counts.std() / monthly_counts.mean() if monthly_counts.mean() != 0 else 0
                
                slow_mover_analysis[item] = {
                    'movement_velocity': round(slow_movers.loc[item, 'movement_velocity'], 4),
                    'price_mean': round(price_mean, 2),
                    'price_volatility': round(price_cv, 4),
                    'volume_mean': round(volume_mean, 2),
                    'volume_std': round(volume_std, 2),
                    'seasonality_score': round(seasonality_score, 4),
                    'total_transactions': slow_movers.loc[item, ('quantity', 'count')],
                    'days_active': slow_movers.loc[item, 'days_active']
                }
            
            # Categorize slow movers by probable causes
            causes = {
                'high_price_items': len([k for k, v in slow_mover_analysis.items() if v['price_mean'] > item_movement[('rate', 'mean')].quantile(0.8)]),
                'volatile_pricing': len([k for k, v in slow_mover_analysis.items() if v['price_volatility'] > 0.5]),
                'low_volume_items': len([k for k, v in slow_mover_analysis.items() if v['volume_mean'] < 1.0]),
                'seasonal_items': len([k for k, v in slow_mover_analysis.items() if v['seasonality_score'] > 1.0])
            }
            
            return {
                'slow_movers_count': len(slow_movers),
                'slow_threshold': round(slow_threshold, 4),
                'slow_mover_analysis': slow_mover_analysis,
                'probable_causes': causes,
                'recommendations': self._generate_slow_mover_recommendations(causes)
            }
            
        except Exception as e:
            return {"error": f"Slow moving analysis failed: {str(e)}"}
    
    def analyze_inventory_variance(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze inventory variance and discrepancies"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            # Calculate variance metrics per item
            item_variance = data.groupby('item').agg({
                'quantity': ['count', 'sum', 'mean', 'std', 'var'],
                'amount': ['sum', 'mean', 'std'],
                'rate': ['mean', 'std']
            })
            
            # Calculate coefficient of variation for quantities
            item_variance['quantity_cv'] = item_variance[('quantity', 'std')] / abs(item_variance[('quantity', 'mean')])
            item_variance['quantity_cv'] = item_variance['quantity_cv'].replace([np.inf, -np.inf], 0)
            
            # Calculate coefficient of variation for rates
            item_variance['rate_cv'] = item_variance[('rate', 'std')] / item_variance[('rate', 'mean')]
            item_variance['rate_cv'] = item_variance['rate_cv'].replace([np.inf, -np.inf], 0)
            
            # Identify high variance items
            high_variance_items = item_variance[item_variance['quantity_cv'] > item_variance['quantity_cv'].quantile(0.8)]
            
            # Analyze variance patterns
            variance_analysis = {}
            for item in high_variance_items.index:
                item_data = data[data['item'] == item]
                
                # Statistical analysis
                quantity_stats = stats.describe(item_data['quantity'])
                rate_stats = stats.describe(item_data['rate'])
                
                # Outlier detection using IQR method
                Q1_qty = item_data['quantity'].quantile(0.25)
                Q3_qty = item_data['quantity'].quantile(0.75)
                IQR_qty = Q3_qty - Q1_qty
                outliers_qty = item_data[(item_data['quantity'] < (Q1_qty - 1.5 * IQR_qty)) | 
                                        (item_data['quantity'] > (Q3_qty + 1.5 * IQR_qty))]
                
                variance_analysis[item] = {
                    'quantity_cv': round(high_variance_items.loc[item, 'quantity_cv'], 4),
                    'rate_cv': round(high_variance_items.loc[item, 'rate_cv'], 4),
                    'quantity_skewness': round(quantity_stats.skewness, 4),
                    'quantity_kurtosis': round(quantity_stats.kurtosis, 4),
                    'outlier_count': len(outliers_qty),
                    'outlier_percentage': round(len(outliers_qty) / len(item_data) * 100, 2),
                    'transaction_count': len(item_data)
                }
            
            return {
                'high_variance_items_count': len(high_variance_items),
                'average_quantity_cv': round(item_variance['quantity_cv'].mean(), 4),
                'average_rate_cv': round(item_variance['rate_cv'].mean(), 4),
                'variance_analysis': variance_analysis,
                'recommendations': self._generate_variance_recommendations(variance_analysis)
            }
            
        except Exception as e:
            return {"error": f"Inventory variance analysis failed: {str(e)}"}
    
    def analyze_demand_volatility(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze demand volatility patterns"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            # Convert date to datetime
            data['date'] = pd.to_datetime(data['date'])
            
            # Calculate weekly demand patterns
            data['week'] = data['date'].dt.isocalendar().week
            data['year'] = data['date'].dt.year
            data['year_week'] = data['year'].astype(str) + '-W' + data['week'].astype(str).str.zfill(2)
            
            # Weekly demand by item
            weekly_demand = data.groupby(['item', 'year_week']).agg({
                'quantity': 'sum',
                'amount': 'sum'
            }).reset_index()
            
            # Calculate volatility metrics
            volatility_analysis = {}
            for item in weekly_demand['item'].unique():
                item_weeks = weekly_demand[weekly_demand['item'] == item]
                
                if len(item_weeks) > 1:
                    # Calculate standard deviation and coefficient of variation
                    demand_std = item_weeks['quantity'].std()
                    demand_mean = item_weeks['quantity'].mean()
                    demand_cv = demand_std / abs(demand_mean) if demand_mean != 0 else 0
                    
                    # Calculate demand pattern consistency
                    positive_weeks = len(item_weeks[item_weeks['quantity'] > 0])
                    negative_weeks = len(item_weeks[item_weeks['quantity'] < 0])
                    zero_weeks = len(item_weeks[item_weeks['quantity'] == 0])
                    
                    # Trend analysis
                    if len(item_weeks) > 2:
                        x = np.arange(len(item_weeks))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, item_weeks['quantity'])
                        trend_strength = abs(r_value)
                    else:
                        trend_strength = 0
                    
                    volatility_analysis[item] = {
                        'demand_cv': round(demand_cv, 4),
                        'demand_std': round(demand_std, 2),
                        'demand_mean': round(demand_mean, 2),
                        'positive_weeks': positive_weeks,
                        'negative_weeks': negative_weeks,
                        'zero_weeks': zero_weeks,
                        'trend_strength': round(trend_strength, 4),
                        'total_weeks': len(item_weeks)
                    }
            
            # Classify items by volatility
            volatility_categories = {
                'low_volatility': len([k for k, v in volatility_analysis.items() if v['demand_cv'] < 0.5]),
                'medium_volatility': len([k for k, v in volatility_analysis.items() if 0.5 <= v['demand_cv'] < 1.0]),
                'high_volatility': len([k for k, v in volatility_analysis.items() if v['demand_cv'] >= 1.0])
            }
            
            return {
                'volatility_analysis': volatility_analysis,
                'volatility_categories': volatility_categories,
                'average_demand_cv': round(np.mean([v['demand_cv'] for v in volatility_analysis.values()]), 4),
                'recommendations': self._generate_volatility_recommendations(volatility_categories)
            }
            
        except Exception as e:
            return {"error": f"Demand volatility analysis failed: {str(e)}"}
    
    def analyze_supplier_impact(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze supplier performance impact on inventory"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            # Use party_name as supplier indicator if available
            if 'party_name' in data.columns:
                supplier_analysis = data.groupby(['party_name', 'item']).agg({
                    'quantity': ['sum', 'count', 'mean'],
                    'amount': ['sum', 'mean'],
                    'rate': ['mean', 'std'],
                    'date': ['min', 'max']
                })
                
                # Calculate supplier performance metrics
                supplier_metrics = {}
                for supplier in data['party_name'].unique():
                    supplier_data = data[data['party_name'] == supplier]
                    
                    # Calculate consistency metrics
                    rate_cv = supplier_data['rate'].std() / supplier_data['rate'].mean() if supplier_data['rate'].mean() != 0 else 0
                    quantity_cv = supplier_data['quantity'].std() / abs(supplier_data['quantity'].mean()) if supplier_data['quantity'].mean() != 0 else 0
                    
                    # Calculate frequency metrics
                    date_range = (supplier_data['date'].max() - supplier_data['date'].min()).days
                    supply_frequency = len(supplier_data) / max(date_range, 1)
                    
                    supplier_metrics[supplier] = {
                        'total_transactions': len(supplier_data),
                        'unique_items': supplier_data['item'].nunique(),
                        'rate_consistency': round(1 - rate_cv, 4),  # Higher is better
                        'quantity_consistency': round(1 - quantity_cv, 4),  # Higher is better
                        'supply_frequency': round(supply_frequency, 4),
                        'average_rate': round(supplier_data['rate'].mean(), 2),
                        'total_value': round(supplier_data['amount'].sum(), 2)
                    }
                
                # Identify problematic suppliers
                problematic_suppliers = {
                    'inconsistent_pricing': [k for k, v in supplier_metrics.items() if v['rate_consistency'] < 0.7],
                    'inconsistent_quantity': [k for k, v in supplier_metrics.items() if v['quantity_consistency'] < 0.7],
                    'low_frequency': [k for k, v in supplier_metrics.items() if v['supply_frequency'] < 0.01]
                }
                
                # Calculate performance ratings
                performance_ratings = {}
                for supplier, metrics in supplier_metrics.items():
                    # Overall performance score (0-1)
                    score = (metrics['rate_consistency'] + metrics['quantity_consistency'] + 
                            min(metrics['supply_frequency'] * 100, 1)) / 3
                    performance_ratings[supplier] = round(score, 4)
                
                return {
                    'supplier_metrics': supplier_metrics,
                    'problematic_suppliers': problematic_suppliers,
                    'total_suppliers': len(supplier_metrics),
                    'performance_rating': performance_ratings,
                    'recommendations': self._generate_supplier_recommendations(problematic_suppliers)
                }
            else:
                return {"error": "Supplier information not available in data"}
                
        except Exception as e:
            return {"error": f"Supplier impact analysis failed: {str(e)}"}
    
    def analyze_seasonal_anomalies(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect seasonal anomalies in inventory patterns"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            data['date'] = pd.to_datetime(data['date'])
            data['month'] = data['date'].dt.month
            data['quarter'] = data['date'].dt.quarter
            
            # Calculate monthly patterns for each item
            monthly_patterns = data.groupby(['item', 'month']).agg({
                'quantity': 'sum',
                'amount': 'sum'
            }).reset_index()
            
            # Detect anomalies using statistical methods
            anomaly_analysis = {}
            for item in monthly_patterns['item'].unique():
                item_monthly = monthly_patterns[monthly_patterns['item'] == item]
                
                if len(item_monthly) > 3:  # Need at least 4 months for analysis
                    # Calculate z-scores for quantity
                    mean_qty = item_monthly['quantity'].mean()
                    std_qty = item_monthly['quantity'].std()
                    
                    if std_qty > 0:
                        item_monthly['z_score'] = (item_monthly['quantity'] - mean_qty) / std_qty
                        
                        # Identify anomalies (z-score > 2 or < -2)
                        anomalies = item_monthly[abs(item_monthly['z_score']) > 2]
                        
                        if len(anomalies) > 0:
                            anomaly_analysis[item] = {
                                'anomaly_months': anomalies['month'].tolist(),
                                'anomaly_quantities': anomalies['quantity'].tolist(),
                                'anomaly_z_scores': anomalies['z_score'].round(2).tolist(),
                                'normal_mean': round(mean_qty, 2),
                                'normal_std': round(std_qty, 2),
                                'anomaly_count': len(anomalies)
                            }
            
            return {
                'items_with_anomalies': len(anomaly_analysis),
                'anomaly_analysis': anomaly_analysis,
                'recommendations': self._generate_anomaly_recommendations(anomaly_analysis)
            }
            
        except Exception as e:
            return {"error": f"Seasonal anomaly analysis failed: {str(e)}"}
    
    def analyze_price_fluctuations(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price fluctuation patterns and impacts"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            # Calculate price metrics per item
            price_analysis = data.groupby('item').agg({
                'rate': ['mean', 'std', 'min', 'max', 'count'],
                'quantity': 'sum',
                'amount': 'sum'
            })
            
            # Calculate price volatility
            price_analysis['price_cv'] = price_analysis[('rate', 'std')] / price_analysis[('rate', 'mean')]
            price_analysis['price_cv'] = price_analysis['price_cv'].replace([np.inf, -np.inf], 0)
            
            # Calculate price range
            price_analysis['price_range'] = price_analysis[('rate', 'max')] - price_analysis[('rate', 'min')]
            price_analysis['price_range_pct'] = (price_analysis['price_range'] / price_analysis[('rate', 'mean')]) * 100
            
            # Identify high fluctuation items
            high_fluctuation_items = price_analysis[price_analysis['price_cv'] > price_analysis['price_cv'].quantile(0.8)]
            
            # Analyze impact of price fluctuations
            fluctuation_impact = {}
            for item in high_fluctuation_items.index:
                item_data = data[data['item'] == item]
                
                # Correlation between price and quantity
                if len(item_data) > 1:
                    price_qty_corr = item_data['rate'].corr(item_data['quantity'])
                    
                    # Calculate elasticity approximation
                    if item_data['rate'].std() > 0 and item_data['quantity'].std() > 0:
                        elasticity = (item_data['quantity'].std() / item_data['quantity'].mean()) / (item_data['rate'].std() / item_data['rate'].mean())
                    else:
                        elasticity = 0
                    
                    fluctuation_impact[item] = {
                        'price_cv': round(high_fluctuation_items.loc[item, 'price_cv'], 4),
                        'price_range_pct': round(high_fluctuation_items.loc[item, 'price_range_pct'], 2),
                        'price_qty_correlation': round(price_qty_corr, 4),
                        'demand_elasticity': round(elasticity, 4),
                        'min_price': round(high_fluctuation_items.loc[item, ('rate', 'min')], 2),
                        'max_price': round(high_fluctuation_items.loc[item, ('rate', 'max')], 2),
                        'avg_price': round(high_fluctuation_items.loc[item, ('rate', 'mean')], 2)
                    }
            
            return {
                'high_fluctuation_items': len(high_fluctuation_items),
                'average_price_cv': round(price_analysis['price_cv'].mean(), 4),
                'fluctuation_impact': fluctuation_impact,
                'recommendations': self._generate_price_fluctuation_recommendations(fluctuation_impact)
            }
            
        except Exception as e:
            return {"error": f"Price fluctuation analysis failed: {str(e)}"}
    
    def analyze_storage_efficiency(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze storage efficiency and godown utilization issues"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            if 'godown' not in data.columns:
                return {"error": "Godown information not available"}
            
            # Godown utilization analysis
            godown_analysis = data.groupby('godown').agg({
                'item': 'nunique',
                'quantity': ['sum', 'count'],
                'amount': 'sum'
            })
            
            # Calculate efficiency metrics
            total_value = data['amount'].sum()
            total_items = data['item'].nunique()
            
            efficiency_metrics = {}
            for godown in godown_analysis.index:
                godown_data = data[data['godown'] == godown]
                
                # Calculate space utilization proxy
                unique_items = godown_data['item'].nunique()
                total_transactions = len(godown_data)
                avg_transactions_per_item = total_transactions / unique_items if unique_items > 0 else 0
                
                # Calculate value density
                godown_value = godown_data['amount'].sum()
                value_density = godown_value / unique_items if unique_items > 0 else 0
                
                # Calculate turnover frequency
                date_range = (godown_data['date'].max() - godown_data['date'].min()).days
                turnover_frequency = total_transactions / max(date_range, 1)
                
                efficiency_metrics[godown] = {
                    'unique_items': unique_items,
                    'item_percentage': round(unique_items / total_items * 100, 2),
                    'value_percentage': round(godown_value / total_value * 100, 2),
                    'value_density': round(value_density, 2),
                    'turnover_frequency': round(turnover_frequency, 4),
                    'avg_transactions_per_item': round(avg_transactions_per_item, 2),
                    'total_value': round(godown_value, 2)
                }
            
            # Identify efficiency issues
            efficiency_issues = {
                'underutilized_godowns': [k for k, v in efficiency_metrics.items() if v['turnover_frequency'] < 0.01],
                'low_value_density': [k for k, v in efficiency_metrics.items() if v['value_density'] < np.mean([m['value_density'] for m in efficiency_metrics.values()]) * 0.5],
                'imbalanced_distribution': [k for k, v in efficiency_metrics.items() if v['item_percentage'] > 80 or v['item_percentage'] < 10]
            }
            
            return {
                'godown_efficiency_metrics': efficiency_metrics,
                'efficiency_issues': efficiency_issues,
                'recommendations': self._generate_storage_recommendations(efficiency_issues)
            }
            
        except Exception as e:
            return {"error": f"Storage efficiency analysis failed: {str(e)}"}
    
    def analyze_general_diagnostics(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General diagnostic analysis"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            return {
                'total_records': len(data),
                'analysis_period': {
                    'start': data['date'].min(),
                    'end': data['date'].max()
                },
                'data_quality_issues': self._check_data_quality(data),
                'general_patterns': self._identify_general_patterns(data)
            }
            
        except Exception as e:
            return {"error": f"General diagnostics failed: {str(e)}"}
    
    def _check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data quality issues"""
        issues = {}
        
        # Check for missing values
        issues['missing_values'] = data.isnull().sum().to_dict()
        
        # Check for zero quantities
        issues['zero_quantities'] = len(data[data['quantity'] == 0])
        
        # Check for negative rates
        issues['negative_rates'] = len(data[data['rate'] < 0])
        
        # Check for duplicate records
        issues['potential_duplicates'] = len(data) - len(data.drop_duplicates())
        
        return issues
    
    def _identify_general_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify general patterns in the data"""
        patterns = {}
        
        # Transaction patterns
        patterns['transaction_frequency'] = len(data) / data['item'].nunique()
        patterns['average_transaction_value'] = data['amount'].mean()
        patterns['value_concentration'] = data['amount'].std() / data['amount'].mean()
        
        return patterns
    
    def _generate_stockout_recommendations(self, root_causes: Dict[str, int]) -> List[str]:
        """Generate recommendations for stockout issues"""
        recommendations = []
        
        if root_causes['high_demand_volatility'] > 0:
            recommendations.append("Implement buffer stock for high volatility items")
        
        if root_causes['low_transaction_frequency'] > 0:
            recommendations.append("Review inventory policies for slow-moving items")
        
        if root_causes['high_outbound_ratio'] > 0:
            recommendations.append("Improve demand forecasting for high-outbound items")
        
        if root_causes['stale_items'] > 0:
            recommendations.append("Review and liquidate stale inventory")
        
        return recommendations
    
    def _generate_slow_mover_recommendations(self, causes: Dict[str, int]) -> List[str]:
        """Generate recommendations for slow-moving items"""
        recommendations = []
        
        if causes['high_price_items'] > 0:
            recommendations.append("Review pricing strategy for high-value items")
        
        if causes['volatile_pricing'] > 0:
            recommendations.append("Stabilize pricing for volatile items")
        
        if causes['low_volume_items'] > 0:
            recommendations.append("Consider bundling or promotional strategies")
        
        if causes['seasonal_items'] > 0:
            recommendations.append("Implement seasonal inventory planning")
        
        return recommendations
    
    def _generate_variance_recommendations(self, variance_analysis: Dict[str, Dict]) -> List[str]:
        """Generate recommendations for variance issues"""
        recommendations = []
        
        high_outlier_items = len([k for k, v in variance_analysis.items() if v['outlier_percentage'] > 10])
        
        if high_outlier_items > 0:
            recommendations.append("Investigate and address outlier transactions")
        
        recommendations.append("Implement better inventory control procedures")
        recommendations.append("Regular inventory audits for high-variance items")
        
        return recommendations
    
    def _generate_volatility_recommendations(self, volatility_categories: Dict[str, int]) -> List[str]:
        """Generate recommendations for demand volatility"""
        recommendations = []
        
        if volatility_categories['high_volatility'] > 0:
            recommendations.append("Implement dynamic inventory planning for high-volatility items")
        
        recommendations.append("Use demand forecasting models for volatile items")
        recommendations.append("Establish safety stock levels based on volatility")
        
        return recommendations
    
    def _generate_supplier_recommendations(self, problematic_suppliers: Dict[str, List]) -> List[str]:
        """Generate recommendations for supplier issues"""
        recommendations = []
        
        if problematic_suppliers['inconsistent_pricing']:
            recommendations.append("Negotiate stable pricing agreements with suppliers")
        
        if problematic_suppliers['inconsistent_quantity']:
            recommendations.append("Work with suppliers to improve quantity consistency")
        
        if problematic_suppliers['low_frequency']:
            recommendations.append("Diversify supplier base or improve supplier relationships")
        
        return recommendations
    
    def _generate_anomaly_recommendations(self, anomaly_analysis: Dict[str, Dict]) -> List[str]:
        """Generate recommendations for seasonal anomalies"""
        recommendations = []
        
        if anomaly_analysis:
            recommendations.append("Investigate causes of seasonal anomalies")
            recommendations.append("Adjust inventory planning for seasonal variations")
            recommendations.append("Implement early warning systems for anomalies")
        
        return recommendations
    
    def _generate_price_fluctuation_recommendations(self, fluctuation_impact: Dict[str, Dict]) -> List[str]:
        """Generate recommendations for price fluctuation issues"""
        recommendations = []
        
        if fluctuation_impact:
            recommendations.append("Implement price stabilization strategies")
            recommendations.append("Review supplier agreements for price volatility")
            recommendations.append("Consider hedging strategies for volatile items")
        
        return recommendations
    
    def _generate_storage_recommendations(self, efficiency_issues: Dict[str, List]) -> List[str]:
        """Generate recommendations for storage efficiency"""
        recommendations = []
        
        if efficiency_issues['underutilized_godowns']:
            recommendations.append("Optimize godown utilization by redistributing inventory")
        
        if efficiency_issues['low_value_density']:
            recommendations.append("Improve space utilization for low-density storage")
        
        if efficiency_issues['imbalanced_distribution']:
            recommendations.append("Rebalance inventory distribution across godowns")
        
        return recommendations
    
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
        
        if any(word in query_lower for word in ['stockout', 'out of stock', 'shortage']):
            return 'stockout_analysis'
        elif any(word in query_lower for word in ['slow', 'moving', 'stagnant']):
            return 'slow_moving_causes'
        elif any(word in query_lower for word in ['variance', 'discrepancy', 'difference']):
            return 'inventory_variance_analysis'
        elif any(word in query_lower for word in ['price', 'rate', 'cost', 'fluctuation']):
            return 'price_fluctuation_analysis'
        elif any(word in query_lower for word in ['volatility', 'variation']):
            return 'demand_volatility_analysis'
        elif any(word in query_lower for word in ['supplier', 'vendor', 'party']):
            return 'supplier_performance_impact'
        elif any(word in query_lower for word in ['seasonal', 'anomaly', 'unusual']):
            return 'seasonal_anomaly_detection'
        elif any(word in query_lower for word in ['storage', 'godown', 'warehouse', 'space']):
            return 'storage_efficiency_analysis'
        else:
            return 'general_diagnostics'