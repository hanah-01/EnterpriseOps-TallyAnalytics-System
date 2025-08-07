"""
Inventory Prescriptive Analytics - "What should we do?"
Optimization and recommendation engine for inventory management decisions
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from .base import AnalyticsBase, AnalyticsResponse


class InventoryPrescriptiveAnalytics(AnalyticsBase):
    """Prescriptive analytics for inventory optimization and recommendations"""
    
    def __init__(self, agent_type: str = "inventory"):
        super().__init__("InventoryPrescriptiveAnalytics", agent_type)
        self.supported_queries = [
            'optimal_stock_levels',
            'reorder_optimization',
            'inventory_allocation',
            'procurement_strategy',
            'storage_optimization',
            'cost_minimization',
            'service_level_optimization',
            'supplier_mix_optimization'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform prescriptive analytics on inventory data"""
        start_time = datetime.now()
        
        # Determine specific analysis type
        analysis_type = self.classify_analysis_type(query)
        
        # Perform analysis based on type
        if analysis_type == 'optimal_stock_levels':
            results = self.optimize_stock_levels(data, params)
        elif analysis_type == 'reorder_optimization':
            results = self.optimize_reorder_strategy(data, params)
        elif analysis_type == 'inventory_allocation':
            results = self.optimize_inventory_allocation(data, params)
        elif analysis_type == 'procurement_strategy':
            results = self.optimize_procurement_strategy(data, params)
        elif analysis_type == 'storage_optimization':
            results = self.optimize_storage_allocation(data, params)
        elif analysis_type == 'cost_minimization':
            results = self.minimize_inventory_costs(data, params)
        elif analysis_type == 'service_level_optimization':
            results = self.optimize_service_levels(data, params)
        elif analysis_type == 'supplier_mix_optimization':
            results = self.optimize_supplier_mix(data, params)
        else:
            results = self.general_optimization(data, params)
        
        return self.prepare_response(
            analytics_type=analysis_type,
            query=query,
            results=results,
            start_time=start_time
        )
    
    def optimize_stock_levels(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize stock levels using EOQ and other methods"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            # Calculate demand statistics
            demand_stats = data.groupby('item').agg({
                'quantity': ['sum', 'count', 'mean', 'std'],
                'amount': ['sum', 'mean'],
                'rate': 'mean'
            })
            
            # Parameters for optimization
            holding_cost_rate = params.get('holding_cost_rate', 0.25)  # 25% per year
            order_cost = params.get('order_cost', 100)  # Cost per order
            stockout_cost = params.get('stockout_cost', 500)  # Cost per stockout
            
            optimizations = {}
            
            for item in demand_stats.index:
                # Basic parameters
                annual_demand = abs(demand_stats.loc[item, ('quantity', 'sum')]) * 365 / 30  # Annualized
                unit_cost = demand_stats.loc[item, ('rate', 'mean')]
                demand_std = demand_stats.loc[item, ('quantity', 'std')]
                
                if annual_demand > 0 and unit_cost > 0:
                    # EOQ Calculation
                    eoq = np.sqrt((2 * annual_demand * order_cost) / (holding_cost_rate * unit_cost))
                    
                    # Safety stock calculation
                    service_level = params.get('service_level', 0.95)
                    z_score = 1.65  # 95% service level
                    lead_time = params.get('lead_time', 7)  # days
                    
                    safety_stock = z_score * demand_std * np.sqrt(lead_time)
                    
                    # Reorder point
                    daily_demand = annual_demand / 365
                    reorder_point = (daily_demand * lead_time) + safety_stock
                    
                    # Total annual cost
                    holding_cost = (eoq / 2 + safety_stock) * holding_cost_rate * unit_cost
                    ordering_cost = (annual_demand / eoq) * order_cost
                    total_cost = holding_cost + ordering_cost
                    
                    # Calculate current performance
                    current_stock = demand_stats.loc[item, ('quantity', 'sum')]
                    current_cost = current_stock * holding_cost_rate * unit_cost
                    
                    # Recommendations
                    recommendations = []
                    if current_stock < reorder_point:
                        recommendations.append(f"Reorder immediately - current stock below reorder point")
                    if current_stock > eoq * 2:
                        recommendations.append(f"Reduce stock level - current stock exceeds optimal level")
                    if abs(current_stock - eoq) > eoq * 0.5:
                        recommendations.append(f"Adjust stock to optimal level of {eoq:.0f}")
                    
                    optimizations[item] = {
                        'current_stock': round(current_stock, 2),
                        'optimal_stock_level': round(eoq, 2),
                        'reorder_point': round(max(0, reorder_point), 2),
                        'safety_stock': round(max(0, safety_stock), 2),
                        'annual_demand': round(annual_demand, 2),
                        'total_annual_cost': round(total_cost, 2),
                        'current_annual_cost': round(current_cost, 2),
                        'potential_savings': round(current_cost - total_cost, 2),
                        'recommendations': recommendations
                    }
            
            # Summary statistics
            total_current_cost = sum(opt['current_annual_cost'] for opt in optimizations.values())
            total_optimal_cost = sum(opt['total_annual_cost'] for opt in optimizations.values())
            total_savings = total_current_cost - total_optimal_cost
            
            return {
                'stock_optimizations': optimizations,
                'summary': {
                    'total_current_cost': round(total_current_cost, 2),
                    'total_optimal_cost': round(total_optimal_cost, 2),
                    'total_potential_savings': round(total_savings, 2),
                    'items_optimized': len(optimizations)
                },
                'parameters_used': {
                    'holding_cost_rate': holding_cost_rate,
                    'order_cost': order_cost,
                    'service_level': service_level,
                    'lead_time': lead_time
                }
            }
            
        except Exception as e:
            return {"error": f"Stock level optimization failed: {str(e)}"}
    
    def optimize_reorder_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize reorder strategy and timing"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            # Calculate demand patterns
            data['date'] = pd.to_datetime(data['date'])
            
            reorder_strategies = {}
            
            for item in data['item'].unique():
                item_data = data[data['item'] == item].sort_values('date')
                
                if len(item_data) >= 10:  # Need sufficient data
                    # Calculate demand patterns
                    daily_demand = item_data.groupby(item_data['date'].dt.date)['quantity'].sum()
                    avg_daily_demand = abs(daily_demand.mean())
                    demand_volatility = daily_demand.std()
                    
                    # Calculate transaction frequency
                    date_range = (item_data['date'].max() - item_data['date'].min()).days
                    transaction_frequency = len(item_data) / max(date_range, 1)
                    
                    # Optimize reorder parameters
                    lead_time = params.get('lead_time', 7)
                    service_level = params.get('service_level', 0.95)
                    
                    # Reorder point optimization
                    z_score = 1.65 if service_level == 0.95 else 2.33 if service_level == 0.99 else 1.28
                    safety_stock = z_score * demand_volatility * np.sqrt(lead_time)
                    reorder_point = (avg_daily_demand * lead_time) + safety_stock
                    
                    # Reorder quantity optimization (based on demand pattern)
                    if transaction_frequency > 0.1:  # Frequent transactions
                        reorder_strategy = 'frequent_small_orders'
                        reorder_quantity = avg_daily_demand * 7  # Weekly supply
                    elif transaction_frequency > 0.03:  # Moderate transactions
                        reorder_strategy = 'moderate_orders'
                        reorder_quantity = avg_daily_demand * 14  # Bi-weekly supply
                    else:  # Infrequent transactions
                        reorder_strategy = 'large_orders'
                        reorder_quantity = avg_daily_demand * 30  # Monthly supply
                    
                    # Calculate optimal review period
                    if demand_volatility > avg_daily_demand:
                        review_period = 'daily'  # High volatility needs daily review
                    elif demand_volatility > avg_daily_demand * 0.5:
                        review_period = 'weekly'  # Moderate volatility
                    else:
                        review_period = 'monthly'  # Low volatility
                    
                    reorder_strategies[item] = {
                        'reorder_point': round(max(0, reorder_point), 2),
                        'reorder_quantity': round(reorder_quantity, 2),
                        'safety_stock': round(max(0, safety_stock), 2),
                        'avg_daily_demand': round(avg_daily_demand, 2),
                        'demand_volatility': round(demand_volatility, 2),
                        'reorder_strategy': reorder_strategy,
                        'review_period': review_period,
                        'lead_time': lead_time,
                        'service_level': service_level
                    }
            
            return {
                'reorder_strategies': reorder_strategies,
                'items_analyzed': len(reorder_strategies),
                'strategy_distribution': self._analyze_strategy_distribution(reorder_strategies)
            }
            
        except Exception as e:
            return {"error": f"Reorder strategy optimization failed: {str(e)}"}
    
    def optimize_inventory_allocation(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize inventory allocation across godowns"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            if 'godown' not in data.columns:
                return {"error": "Godown information not available"}
            
            # Current allocation
            current_allocation = data.groupby(['godown', 'item']).agg({
                'quantity': 'sum',
                'amount': 'sum'
            }).reset_index()
            
            # Calculate demand patterns by godown
            godown_demand = data.groupby('godown').agg({
                'quantity': ['sum', 'count', 'mean'],
                'amount': 'sum'
            })
            
            # Total capacity constraints (if provided)
            total_capacity = params.get('total_capacity', {})
            demand_weights = params.get('demand_weights', {})
            
            allocation_optimization = {}
            
            for item in data['item'].unique():
                item_data = data[data['item'] == item]
                item_godown_data = item_data.groupby('godown').agg({
                    'quantity': ['sum', 'count'],
                    'amount': 'sum'
                })
                
                # Calculate demand intensity by godown
                total_item_demand = item_data['quantity'].sum()
                
                optimal_allocation = {}
                for godown in item_godown_data.index:
                    godown_demand_ratio = abs(item_godown_data.loc[godown, ('quantity', 'sum')] / total_item_demand)
                    transaction_frequency = item_godown_data.loc[godown, ('quantity', 'count')]
                    
                    # Apply demand weights if provided
                    if godown in demand_weights:
                        godown_demand_ratio *= demand_weights[godown]
                    
                    optimal_allocation[godown] = {
                        'current_quantity': item_godown_data.loc[godown, ('quantity', 'sum')],
                        'demand_ratio': round(godown_demand_ratio, 4),
                        'transaction_frequency': transaction_frequency,
                        'recommended_allocation': round(total_item_demand * godown_demand_ratio, 2)
                    }
                
                allocation_optimization[item] = optimal_allocation
            
            return {
                'allocation_optimization': allocation_optimization,
                'current_allocation_summary': self._summarize_current_allocation(current_allocation),
                'optimization_recommendations': self._generate_allocation_recommendations(allocation_optimization)
            }
            
        except Exception as e:
            return {"error": f"Inventory allocation optimization failed: {str(e)}"}
    
    def optimize_procurement_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize procurement strategy"""
        try:
            # Analyze procurement patterns
            procurement_analysis = data.groupby('item').agg({
                'quantity': ['sum', 'count', 'mean', 'std'],
                'amount': ['sum', 'mean'],
                'rate': ['mean', 'std', 'min', 'max']
            })
            
            # Parameters
            budget_constraint = params.get('budget_constraint', float('inf'))
            min_order_quantity = params.get('min_order_quantity', 1)
            max_order_quantity = params.get('max_order_quantity', 1000)
            
            procurement_strategies = {}
            
            for item in procurement_analysis.index:
                # Basic metrics
                total_demand = abs(procurement_analysis.loc[item, ('quantity', 'sum')])
                avg_price = procurement_analysis.loc[item, ('rate', 'mean')]
                price_volatility = procurement_analysis.loc[item, ('rate', 'std')] / avg_price if avg_price > 0 else 0
                
                # Procurement strategy based on demand and price patterns
                if total_demand > procurement_analysis[('quantity', 'sum')].quantile(0.8):
                    # High demand items
                    if price_volatility < 0.1:  # Stable prices
                        strategy = 'bulk_purchase'
                        recommended_quantity = min(total_demand * 0.5, max_order_quantity)
                    else:  # Volatile prices
                        strategy = 'opportunistic_purchase'
                        recommended_quantity = min(total_demand * 0.25, max_order_quantity)
                elif total_demand > procurement_analysis[('quantity', 'sum')].quantile(0.4):
                    # Medium demand items
                    strategy = 'regular_purchase'
                    recommended_quantity = min(total_demand * 0.3, max_order_quantity)
                else:
                    # Low demand items
                    strategy = 'just_in_time'
                    recommended_quantity = min(total_demand * 0.2, max_order_quantity)
                
                # Apply constraints
                recommended_quantity = max(min_order_quantity, recommended_quantity)
                
                # Calculate timing
                transaction_frequency = procurement_analysis.loc[item, ('quantity', 'count')]
                if transaction_frequency > 20:
                    purchase_frequency = 'weekly'
                elif transaction_frequency > 10:
                    purchase_frequency = 'bi-weekly'
                else:
                    purchase_frequency = 'monthly'
                
                procurement_strategies[item] = {
                    'strategy': strategy,
                    'recommended_quantity': round(recommended_quantity, 2),
                    'purchase_frequency': purchase_frequency,
                    'total_demand': round(total_demand, 2),
                    'avg_price': round(avg_price, 2),
                    'price_volatility': round(price_volatility, 4),
                    'estimated_cost': round(recommended_quantity * avg_price, 2)
                }
            
            # Budget optimization
            if budget_constraint < float('inf'):
                procurement_strategies = self._optimize_within_budget(procurement_strategies, budget_constraint)
            
            return {
                'procurement_strategies': procurement_strategies,
                'strategy_summary': self._summarize_procurement_strategies(procurement_strategies),
                'total_estimated_cost': sum(s['estimated_cost'] for s in procurement_strategies.values())
            }
            
        except Exception as e:
            return {"error": f"Procurement strategy optimization failed: {str(e)}"}
    
    def optimize_storage_allocation(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize storage allocation and warehouse management"""
        try:
            if 'godown' not in data.columns:
                return {"error": "Storage information not available"}
            
            # Storage capacity constraints
            storage_capacities = params.get('storage_capacities', {})
            storage_costs = params.get('storage_costs', {})
            
            # Current storage utilization
            storage_utilization = data.groupby('godown').agg({
                'item': 'nunique',
                'quantity': 'sum',
                'amount': 'sum'
            })
            
            # Calculate value density (value per unit space)
            item_value_density = data.groupby('item').agg({
                'amount': 'sum',
                'quantity': 'sum'
            })
            item_value_density['value_density'] = item_value_density['amount'] / abs(item_value_density['quantity'])
            
            # Optimization recommendations
            storage_optimization = {}
            
            for godown in storage_utilization.index:
                godown_data = data[data['godown'] == godown]
                
                # Calculate current metrics
                current_items = godown_data['item'].nunique()
                current_quantity = godown_data['quantity'].sum()
                current_value = godown_data['amount'].sum()
                
                # Get capacity if available
                capacity = storage_capacities.get(godown, current_quantity * 1.5)
                utilization_rate = abs(current_quantity) / capacity if capacity > 0 else 0
                
                # Recommendations based on utilization
                if utilization_rate > 0.9:
                    recommendation = 'overutilized_redistribute'
                    action = 'Move low-value items to other locations'
                elif utilization_rate < 0.3:
                    recommendation = 'underutilized_consolidate'
                    action = 'Consolidate items from other locations'
                else:
                    recommendation = 'optimal_utilization'
                    action = 'Maintain current allocation'
                
                # Calculate optimal item mix
                godown_items = godown_data.groupby('item').agg({
                    'quantity': 'sum',
                    'amount': 'sum'
                })
                
                # Rank items by value density
                item_rankings = []
                for item in godown_items.index:
                    if item in item_value_density.index:
                        value_density = item_value_density.loc[item, 'value_density']
                        item_rankings.append((item, value_density, godown_items.loc[item, 'quantity']))
                
                item_rankings.sort(key=lambda x: x[1], reverse=True)
                
                storage_optimization[godown] = {
                    'current_utilization': round(utilization_rate, 4),
                    'current_items': current_items,
                    'current_quantity': round(current_quantity, 2),
                    'current_value': round(current_value, 2),
                    'capacity': capacity,
                    'recommendation': recommendation,
                    'action': action,
                    'high_value_items': [item[0] for item in item_rankings[:5]],
                    'low_value_items': [item[0] for item in item_rankings[-5:]]
                }
            
            return {
                'storage_optimization': storage_optimization,
                'overall_recommendations': self._generate_storage_recommendations(storage_optimization),
                'value_density_analysis': item_value_density.to_dict('index')
            }
            
        except Exception as e:
            return {"error": f"Storage optimization failed: {str(e)}"}
    
    def minimize_inventory_costs(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Minimize total inventory costs"""
        try:
            # Cost parameters
            holding_cost_rate = params.get('holding_cost_rate', 0.25)
            order_cost = params.get('order_cost', 100)
            stockout_cost = params.get('stockout_cost', 500)
            storage_cost_per_unit = params.get('storage_cost_per_unit', 1)
            
            # Calculate current costs
            current_costs = {}
            cost_optimizations = {}
            
            item_stats = data.groupby('item').agg({
                'quantity': ['sum', 'count', 'std'],
                'amount': ['sum', 'mean'],
                'rate': 'mean'
            })
            
            for item in item_stats.index:
                # Current metrics
                current_stock = item_stats.loc[item, ('quantity', 'sum')]
                unit_cost = item_stats.loc[item, ('rate', 'mean')]
                annual_demand = abs(current_stock) * 12  # Approximate annual demand
                
                # Current costs
                current_holding_cost = abs(current_stock) * holding_cost_rate * unit_cost
                current_storage_cost = abs(current_stock) * storage_cost_per_unit
                
                # Optimize for minimum cost
                if annual_demand > 0 and unit_cost > 0:
                    # EOQ for minimum cost
                    optimal_order_quantity = np.sqrt((2 * annual_demand * order_cost) / (holding_cost_rate * unit_cost))
                    optimal_stock_level = optimal_order_quantity / 2
                    
                    # Optimized costs
                    optimal_holding_cost = optimal_stock_level * holding_cost_rate * unit_cost
                    optimal_storage_cost = optimal_stock_level * storage_cost_per_unit
                    optimal_ordering_cost = (annual_demand / optimal_order_quantity) * order_cost
                    
                    total_current_cost = current_holding_cost + current_storage_cost
                    total_optimal_cost = optimal_holding_cost + optimal_storage_cost + optimal_ordering_cost
                    
                    cost_optimizations[item] = {
                        'current_stock': round(current_stock, 2),
                        'optimal_stock': round(optimal_stock_level, 2),
                        'current_total_cost': round(total_current_cost, 2),
                        'optimal_total_cost': round(total_optimal_cost, 2),
                        'cost_savings': round(total_current_cost - total_optimal_cost, 2),
                        'savings_percentage': round((total_current_cost - total_optimal_cost) / total_current_cost * 100, 2) if total_current_cost > 0 else 0
                    }
            
            # Summary
            total_current_cost = sum(opt['current_total_cost'] for opt in cost_optimizations.values())
            total_optimal_cost = sum(opt['optimal_total_cost'] for opt in cost_optimizations.values())
            total_savings = total_current_cost - total_optimal_cost
            
            return {
                'cost_optimizations': cost_optimizations,
                'cost_summary': {
                    'total_current_cost': round(total_current_cost, 2),
                    'total_optimal_cost': round(total_optimal_cost, 2),
                    'total_savings': round(total_savings, 2),
                    'savings_percentage': round(total_savings / total_current_cost * 100, 2) if total_current_cost > 0 else 0
                },
                'top_cost_savers': self._identify_top_cost_savers(cost_optimizations)
            }
            
        except Exception as e:
            return {"error": f"Cost minimization failed: {str(e)}"}
    
    def optimize_service_levels(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize service levels vs. inventory investment"""
        try:
            # Service level targets
            target_service_levels = params.get('service_levels', [0.90, 0.95, 0.99])
            
            service_optimizations = {}
            
            for service_level in target_service_levels:
                z_score = 1.28 if service_level == 0.90 else 1.65 if service_level == 0.95 else 2.33
                
                item_optimizations = {}
                
                for item in data['item'].unique():
                    item_data = data[data['item'] == item]
                    
                    # Calculate demand statistics
                    demand_mean = abs(item_data['quantity'].mean())
                    demand_std = item_data['quantity'].std()
                    unit_cost = item_data['rate'].mean()
                    
                    # Calculate required safety stock
                    lead_time = params.get('lead_time', 7)
                    safety_stock = z_score * demand_std * np.sqrt(lead_time)
                    
                    # Calculate investment required
                    investment = safety_stock * unit_cost
                    
                    item_optimizations[item] = {
                        'service_level': service_level,
                        'safety_stock': round(max(0, safety_stock), 2),
                        'investment_required': round(investment, 2),
                        'demand_mean': round(demand_mean, 2),
                        'demand_std': round(demand_std, 2)
                    }
                
                total_investment = sum(opt['investment_required'] for opt in item_optimizations.values())
                
                service_optimizations[f'{service_level*100:.0f}%'] = {
                    'item_optimizations': item_optimizations,
                    'total_investment': round(total_investment, 2),
                    'items_covered': len(item_optimizations)
                }
            
            return {
                'service_level_analysis': service_optimizations,
                'investment_comparison': self._compare_service_investments(service_optimizations),
                'recommendations': self._recommend_service_level(service_optimizations)
            }
            
        except Exception as e:
            return {"error": f"Service level optimization failed: {str(e)}"}
    
    def optimize_supplier_mix(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize supplier mix and sourcing strategy"""
        try:
            if 'party_name' not in data.columns:
                return {"error": "Supplier information not available"}
            
            # Supplier performance metrics
            supplier_metrics = data.groupby('party_name').agg({
                'quantity': ['sum', 'count', 'std'],
                'amount': ['sum', 'mean'],
                'rate': ['mean', 'std'],
                'item': 'nunique'
            })
            
            # Calculate supplier scores
            supplier_scores = {}
            
            for supplier in supplier_metrics.index:
                # Performance metrics
                total_volume = abs(supplier_metrics.loc[supplier, ('quantity', 'sum')])
                price_stability = 1 - (supplier_metrics.loc[supplier, ('rate', 'std')] / supplier_metrics.loc[supplier, ('rate', 'mean')])
                reliability = supplier_metrics.loc[supplier, ('quantity', 'count')] / 30  # Transactions per day
                diversity = supplier_metrics.loc[supplier, ('item', 'nunique')]
                
                # Calculate composite score
                score = (total_volume * 0.3 + price_stability * 0.25 + reliability * 0.25 + diversity * 0.2)
                
                supplier_scores[supplier] = {
                    'total_volume': round(total_volume, 2),
                    'price_stability': round(max(0, price_stability), 4),
                    'reliability_score': round(reliability, 4),
                    'item_diversity': diversity,
                    'composite_score': round(score, 4),
                    'performance_rating': self._rate_supplier(score)
                }
            
            # Optimize supplier allocation
            total_business = sum(s['total_volume'] for s in supplier_scores.values())
            
            optimized_allocation = {}
            for supplier, metrics in supplier_scores.items():
                # Allocate based on performance score
                optimal_share = metrics['composite_score'] / sum(s['composite_score'] for s in supplier_scores.values())
                current_share = metrics['total_volume'] / total_business
                
                optimized_allocation[supplier] = {
                    'current_share': round(current_share, 4),
                    'optimal_share': round(optimal_share, 4),
                    'recommended_action': self._get_supplier_action(current_share, optimal_share),
                    'performance_rating': metrics['performance_rating']
                }
            
            return {
                'supplier_analysis': supplier_scores,
                'optimized_allocation': optimized_allocation,
                'diversification_recommendations': self._generate_diversification_recommendations(supplier_scores)
            }
            
        except Exception as e:
            return {"error": f"Supplier mix optimization failed: {str(e)}"}
    
    def general_optimization(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General inventory optimization recommendations"""
        try:
            # Overall inventory health
            total_items = data['item'].nunique()
            total_value = data['amount'].sum()
            total_quantity = data['quantity'].sum()
            
            # Key recommendations
            recommendations = [
                "Implement ABC analysis for inventory categorization",
                "Establish reorder points based on lead time and demand",
                "Review slow-moving items for potential liquidation",
                "Optimize storage allocation based on item velocity",
                "Implement regular inventory audits"
            ]
            
            return {
                'inventory_overview': {
                    'total_items': total_items,
                    'total_value': round(total_value, 2),
                    'total_quantity': round(total_quantity, 2),
                    'average_item_value': round(total_value / total_items, 2) if total_items > 0 else 0
                },
                'key_recommendations': recommendations,
                'optimization_priorities': [
                    'Cost reduction through optimal stock levels',
                    'Service level improvement',
                    'Storage efficiency optimization',
                    'Supplier relationship optimization'
                ]
            }
            
        except Exception as e:
            return {"error": f"General optimization failed: {str(e)}"}
    
    def _analyze_strategy_distribution(self, strategies: Dict[str, Dict]) -> Dict[str, int]:
        """Analyze distribution of reorder strategies"""
        distribution = {}
        for strategy_data in strategies.values():
            strategy_type = strategy_data['reorder_strategy']
            distribution[strategy_type] = distribution.get(strategy_type, 0) + 1
        return distribution
    
    def _summarize_current_allocation(self, allocation: pd.DataFrame) -> Dict[str, Any]:
        """Summarize current inventory allocation"""
        return {
            'total_godowns': allocation['godown'].nunique(),
            'total_items': allocation['item'].nunique(),
            'total_quantity': round(allocation['quantity'].sum(), 2),
            'total_value': round(allocation['amount'].sum(), 2)
        }
    
    def _generate_allocation_recommendations(self, optimization: Dict[str, Dict]) -> List[str]:
        """Generate inventory allocation recommendations"""
        recommendations = []
        
        # Analyze allocation patterns
        imbalanced_items = 0
        for item, godown_data in optimization.items():
            max_allocation = max(data['recommended_allocation'] for data in godown_data.values())
            min_allocation = min(data['recommended_allocation'] for data in godown_data.values())
            if max_allocation > min_allocation * 3:  # Highly imbalanced
                imbalanced_items += 1
        
        if imbalanced_items > 0:
            recommendations.append(f"Rebalance {imbalanced_items} items across godowns")
        
        recommendations.append("Implement demand-based allocation strategy")
        recommendations.append("Regular review of allocation patterns")
        
        return recommendations
    
    def _summarize_procurement_strategies(self, strategies: Dict[str, Dict]) -> Dict[str, int]:
        """Summarize procurement strategies"""
        summary = {}
        for strategy_data in strategies.values():
            strategy_type = strategy_data['strategy']
            summary[strategy_type] = summary.get(strategy_type, 0) + 1
        return summary
    
    def _optimize_within_budget(self, strategies: Dict[str, Dict], budget: float) -> Dict[str, Dict]:
        """Optimize procurement within budget constraints"""
        # Sort items by priority (could be based on demand, profitability, etc.)
        sorted_items = sorted(strategies.items(), key=lambda x: x[1]['total_demand'], reverse=True)
        
        remaining_budget = budget
        optimized_strategies = {}
        
        for item, strategy in sorted_items:
            if strategy['estimated_cost'] <= remaining_budget:
                optimized_strategies[item] = strategy
                remaining_budget -= strategy['estimated_cost']
            else:
                # Adjust quantity to fit budget
                max_quantity = remaining_budget / strategy['estimated_cost'] * strategy['recommended_quantity']
                if max_quantity > 0:
                    strategy['recommended_quantity'] = round(max_quantity, 2)
                    strategy['estimated_cost'] = round(max_quantity * strategy['avg_price'], 2)
                    optimized_strategies[item] = strategy
                    remaining_budget = 0
                break
        
        return optimized_strategies
    
    def _generate_storage_recommendations(self, optimization: Dict[str, Dict]) -> List[str]:
        """Generate storage optimization recommendations"""
        recommendations = []
        
        overutilized = [k for k, v in optimization.items() if v['recommendation'] == 'overutilized_redistribute']
        underutilized = [k for k, v in optimization.items() if v['recommendation'] == 'underutilized_consolidate']
        
        if overutilized:
            recommendations.append(f"Redistribute inventory from overutilized locations: {', '.join(overutilized)}")
        
        if underutilized:
            recommendations.append(f"Consolidate inventory to underutilized locations: {', '.join(underutilized)}")
        
        recommendations.append("Implement value-based storage allocation")
        recommendations.append("Regular storage utilization audits")
        
        return recommendations
    
    def _identify_top_cost_savers(self, optimizations: Dict[str, Dict]) -> List[Tuple[str, float]]:
        """Identify items with highest cost savings potential"""
        cost_savers = [(item, data['cost_savings']) for item, data in optimizations.items()]
        return sorted(cost_savers, key=lambda x: x[1], reverse=True)[:5]
    
    def _compare_service_investments(self, service_optimizations: Dict[str, Dict]) -> Dict[str, float]:
        """Compare investments required for different service levels"""
        return {level: data['total_investment'] for level, data in service_optimizations.items()}
    
    def _recommend_service_level(self, service_optimizations: Dict[str, Dict]) -> List[str]:
        """Recommend optimal service level"""
        investments = self._compare_service_investments(service_optimizations)
        
        recommendations = []
        recommendations.append("Consider 95% service level for balanced cost and service")
        recommendations.append("Implement tiered service levels based on item importance")
        recommendations.append("Regular review of service level performance")
        
        return recommendations
    
    def _rate_supplier(self, score: float) -> str:
        """Rate supplier based on composite score"""
        if score > 0.8:
            return 'excellent'
        elif score > 0.6:
            return 'good'
        elif score > 0.4:
            return 'average'
        else:
            return 'poor'
    
    def _get_supplier_action(self, current_share: float, optimal_share: float) -> str:
        """Get recommended action for supplier"""
        if optimal_share > current_share * 1.2:
            return 'increase_allocation'
        elif optimal_share < current_share * 0.8:
            return 'decrease_allocation'
        else:
            return 'maintain_current'
    
    def _generate_diversification_recommendations(self, supplier_scores: Dict[str, Dict]) -> List[str]:
        """Generate supplier diversification recommendations"""
        recommendations = []
        
        total_suppliers = len(supplier_scores)
        excellent_suppliers = len([s for s in supplier_scores.values() if s['performance_rating'] == 'excellent'])
        poor_suppliers = len([s for s in supplier_scores.values() if s['performance_rating'] == 'poor'])
        
        if total_suppliers < 3:
            recommendations.append("Consider adding more suppliers for better risk management")
        
        if poor_suppliers > 0:
            recommendations.append(f"Review and potentially replace {poor_suppliers} poor-performing suppliers")
        
        recommendations.append("Maintain balanced supplier portfolio")
        recommendations.append("Regular supplier performance reviews")
        
        return recommendations
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the type of prescriptive analysis based on query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['optimal stock', 'stock level', 'inventory level']):
            return 'optimal_stock_levels'
        elif any(word in query_lower for word in ['reorder', 'order point', 'ordering']):
            return 'reorder_optimization'
        elif any(word in query_lower for word in ['allocation', 'distribute', 'assign']):
            return 'inventory_allocation'
        elif any(word in query_lower for word in ['procurement', 'purchasing', 'buying']):
            return 'procurement_strategy'
        elif any(word in query_lower for word in ['storage', 'warehouse', 'godown']):
            return 'storage_optimization'
        elif any(word in query_lower for word in ['cost', 'minimize', 'reduce cost']):
            return 'cost_minimization'
        elif any(word in query_lower for word in ['service level', 'availability', 'stockout']):
            return 'service_level_optimization'
        elif any(word in query_lower for word in ['supplier', 'vendor', 'sourcing']):
            return 'supplier_mix_optimization'
        else:
            return 'general_optimization'