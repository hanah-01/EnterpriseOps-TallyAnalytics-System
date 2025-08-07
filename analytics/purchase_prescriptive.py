"""
Purchase Prescriptive Analytics - "What should we do?"
Strategic recommendations and optimization strategies for procurement management
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .base import AnalyticsBase, AnalyticsResponse


class PurchasePrescriptiveAnalytics(AnalyticsBase):
    """Prescriptive analytics for purchase optimization and strategic recommendations"""
    
    def __init__(self, agent_type: str = "purchase"):
        super().__init__("PurchasePrescriptiveAnalytics", agent_type)
        self.supported_queries = [
            'cost_optimization_strategies',
            'supplier_consolidation_plan',
            'procurement_process_optimization',
            'inventory_planning_recommendations',
            'contract_negotiation_strategies',
            'risk_mitigation_strategies',
            'budget_allocation_optimization',
            'strategic_sourcing_recommendations'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform prescriptive analytics on purchase data"""
        start_time = datetime.now()
        
        # Determine specific analysis type
        analysis_type = self.classify_analysis_type(query)
        
        # Perform analysis based on type
        if analysis_type == 'cost_optimization_strategies':
            results = self.generate_cost_optimization_strategies(data, params)
        elif analysis_type == 'supplier_consolidation_plan':
            results = self.generate_supplier_consolidation_plan(data, params)
        elif analysis_type == 'procurement_process_optimization':
            results = self.generate_process_optimization_strategies(data, params)
        elif analysis_type == 'inventory_planning_recommendations':
            results = self.generate_inventory_planning_recommendations(data, params)
        elif analysis_type == 'contract_negotiation_strategies':
            results = self.generate_contract_negotiation_strategies(data, params)
        elif analysis_type == 'risk_mitigation_strategies':
            results = self.generate_risk_mitigation_strategies(data, params)
        elif analysis_type == 'budget_allocation_optimization':
            results = self.generate_budget_allocation_optimization(data, params)
        elif analysis_type == 'strategic_sourcing_recommendations':
            results = self.generate_strategic_sourcing_recommendations(data, params)
        else:
            results = self.generate_general_purchase_recommendations(data, params)
        
        return self.prepare_response(
            analytics_type=analysis_type,
            query=query,
            results=results,
            start_time=start_time
        )
    
    def generate_cost_optimization_strategies(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cost optimization strategies"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Cost optimization analysis
            cost_data = data[data['amount'] > 0].copy()
            cost_data['date'] = pd.to_datetime(cost_data['date'])
            
            # Identify cost optimization opportunities
            supplier_costs = cost_data.groupby('party_name')['amount'].agg(['sum', 'count', 'mean', 'std'])
            
            # High-cost suppliers
            high_cost_suppliers = supplier_costs.nlargest(10, 'sum')
            
            # Price variance analysis
            item_price_variance = cost_data.groupby('item').agg({
                'rate': ['mean', 'std', 'count'],
                'amount': 'sum'
            })
            item_price_variance.columns = ['_'.join(col).strip() for col in item_price_variance.columns]
            item_price_variance['price_variance'] = (item_price_variance['rate_std'] / item_price_variance['rate_mean']) * 100
            
            # High variance items
            high_variance_items = item_price_variance[
                (item_price_variance['price_variance'] > 20) & 
                (item_price_variance['rate_count'] > 3)
            ].nlargest(10, 'amount_sum')
            
            # Volume consolidation opportunities
            monthly_spending = cost_data.groupby(cost_data['date'].dt.to_period('M'))['amount'].sum()
            avg_monthly_spending = monthly_spending.mean()
            
            # Generate strategies
            strategies = []
            
            # Supplier consolidation strategy
            if len(high_cost_suppliers) > 3:
                top_3_spending = high_cost_suppliers.head(3)['sum'].sum()
                total_spending = supplier_costs['sum'].sum()
                consolidation_potential = (total_spending - top_3_spending) / total_spending * 100
                
                strategies.append({
                    "strategy": "Supplier Consolidation",
                    "description": f"Consolidate {consolidation_potential:.1f}% of spending with top 3 suppliers",
                    "potential_savings": round(consolidation_potential * 0.05 * total_spending / 100, 2),  # 5% savings estimate
                    "implementation_priority": "High",
                    "timeframe": "3-6 months"
                })
            
            # Price standardization strategy
            if len(high_variance_items) > 0:
                variance_impact = high_variance_items['amount_sum'].sum()
                strategies.append({
                    "strategy": "Price Standardization",
                    "description": f"Standardize pricing for {len(high_variance_items)} high-variance items",
                    "potential_savings": round(variance_impact * 0.10, 2),  # 10% savings estimate
                    "implementation_priority": "Medium",
                    "timeframe": "2-4 months"
                })
            
            # Volume-based discounts
            strategies.append({
                "strategy": "Volume-Based Negotiation",
                "description": "Negotiate volume discounts based on consolidated purchasing",
                "potential_savings": round(avg_monthly_spending * 12 * 0.03, 2),  # 3% annual savings
                "implementation_priority": "High",
                "timeframe": "1-3 months"
            })
            
            # Payment term optimization
            strategies.append({
                "strategy": "Payment Term Optimization",
                "description": "Negotiate extended payment terms for better cash flow",
                "potential_savings": round(avg_monthly_spending * 12 * 0.02, 2),  # 2% cost of capital savings
                "implementation_priority": "Medium",
                "timeframe": "2-3 months"
            })
            
            return {
                "cost_optimization_strategies": strategies,
                "total_potential_savings": round(sum([s['potential_savings'] for s in strategies]), 2),
                "high_impact_opportunities": {
                    "supplier_consolidation": len(high_cost_suppliers),
                    "price_standardization": len(high_variance_items),
                    "volume_optimization": round(avg_monthly_spending * 12, 2)
                },
                "implementation_roadmap": self._create_implementation_roadmap(strategies)
            }
            
        except Exception as e:
            return {"error": f"Cost optimization strategies failed: {str(e)}"}
    
    def generate_supplier_consolidation_plan(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate supplier consolidation plan"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Supplier consolidation analysis
            supplier_data = data[data['amount'] > 0].copy()
            
            # Calculate supplier metrics
            supplier_metrics = supplier_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean'],
                'item': 'nunique',
                'date': ['min', 'max']
            })
            
            # Flatten column names
            supplier_metrics.columns = ['_'.join(col).strip() for col in supplier_metrics.columns]
            
            # Calculate relationship duration and frequency
            supplier_metrics['relationship_duration'] = (
                pd.to_datetime(supplier_metrics['date_max']) - 
                pd.to_datetime(supplier_metrics['date_min'])
            ).dt.days
            
            # Supplier categorization
            total_spending = supplier_metrics['amount_sum'].sum()
            supplier_metrics['spending_share'] = supplier_metrics['amount_sum'] / total_spending * 100
            
            # Create ABC segmentation
            supplier_metrics_sorted = supplier_metrics.sort_values('amount_sum', ascending=False)
            cumulative_share = supplier_metrics_sorted['spending_share'].cumsum()
            
            supplier_metrics_sorted['category'] = 'C'
            supplier_metrics_sorted.loc[cumulative_share <= 80, 'category'] = 'A'
            supplier_metrics_sorted.loc[(cumulative_share > 80) & (cumulative_share <= 95), 'category'] = 'B'
            
            # Consolidation opportunities
            category_a_suppliers = supplier_metrics_sorted[supplier_metrics_sorted['category'] == 'A']
            category_b_suppliers = supplier_metrics_sorted[supplier_metrics_sorted['category'] == 'B']
            category_c_suppliers = supplier_metrics_sorted[supplier_metrics_sorted['category'] == 'C']
            
            # Generate consolidation plan
            consolidation_plan = {
                "strategic_suppliers": {
                    "count": len(category_a_suppliers),
                    "spending_share": round(category_a_suppliers['spending_share'].sum(), 2),
                    "recommendation": "Strengthen partnerships, negotiate strategic agreements"
                },
                "tactical_suppliers": {
                    "count": len(category_b_suppliers),
                    "spending_share": round(category_b_suppliers['spending_share'].sum(), 2),
                    "recommendation": "Consolidate or develop into strategic suppliers"
                },
                "transactional_suppliers": {
                    "count": len(category_c_suppliers),
                    "spending_share": round(category_c_suppliers['spending_share'].sum(), 2),
                    "recommendation": "Consolidate or eliminate low-value suppliers"
                }
            }
            
            # Consolidation recommendations
            recommendations = []
            
            if len(category_c_suppliers) > 10:
                recommendations.append({
                    "action": "Eliminate Low-Value Suppliers",
                    "target": f"Reduce {len(category_c_suppliers)} suppliers to top 10",
                    "impact": "Reduced administrative overhead",
                    "savings_potential": round(category_c_suppliers['amount_sum'].sum() * 0.05, 2)
                })
            
            if len(category_b_suppliers) > 5:
                recommendations.append({
                    "action": "Consolidate Mid-Tier Suppliers",
                    "target": f"Reduce {len(category_b_suppliers)} suppliers to 3-5 key suppliers",
                    "impact": "Better negotiation power and service levels",
                    "savings_potential": round(category_b_suppliers['amount_sum'].sum() * 0.08, 2)
                })
            
            recommendations.append({
                "action": "Develop Strategic Partnerships",
                "target": f"Enhance relationships with top {len(category_a_suppliers)} suppliers",
                "impact": "Long-term cost stability and innovation",
                "savings_potential": round(category_a_suppliers['amount_sum'].sum() * 0.05, 2)
            })
            
            return {
                "supplier_segmentation": consolidation_plan,
                "consolidation_recommendations": recommendations,
                "current_supplier_base": {
                    "total_suppliers": len(supplier_metrics),
                    "recommended_optimal_count": len(category_a_suppliers) + min(5, len(category_b_suppliers)) + min(10, len(category_c_suppliers))
                },
                "expected_benefits": {
                    "cost_savings": round(sum([r['savings_potential'] for r in recommendations]), 2),
                    "administrative_efficiency": "30-50% reduction in supplier management overhead",
                    "quality_improvement": "Better supplier relationships and service levels"
                }
            }
            
        except Exception as e:
            return {"error": f"Supplier consolidation plan failed: {str(e)}"}
    
    def generate_process_optimization_strategies(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate procurement process optimization strategies"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Process efficiency analysis
            process_data = data[data['amount'] > 0].copy()
            process_data['date'] = pd.to_datetime(process_data['date'])
            
            # Transaction patterns
            daily_transactions = process_data.groupby(process_data['date'].dt.date).agg({
                'voucher_number': 'nunique',
                'amount': ['sum', 'count'],
                'party_name': 'nunique'
            })
            
            # Small transaction analysis
            transaction_sizes = process_data.groupby('voucher_number')['amount'].sum()
            small_transactions = transaction_sizes[transaction_sizes < transaction_sizes.quantile(0.25)]
            
            # Frequency analysis
            supplier_frequency = process_data.groupby('party_name')['voucher_number'].nunique()
            low_frequency_suppliers = supplier_frequency[supplier_frequency < 3]
            
            # Generate optimization strategies
            strategies = []
            
            # Small order consolidation
            if len(small_transactions) > 0:
                small_order_value = small_transactions.sum()
                strategies.append({
                    "strategy": "Small Order Consolidation",
                    "current_state": f"{len(small_transactions)} small orders totaling ₹{small_order_value:,.2f}",
                    "recommendation": "Implement minimum order values and batch ordering",
                    "expected_improvement": "20-30% reduction in processing costs",
                    "implementation_steps": [
                        "Set minimum order thresholds",
                        "Implement batch ordering schedules",
                        "Negotiate consolidated delivery terms"
                    ]
                })
            
            # Supplier rationalization
            if len(low_frequency_suppliers) > 0:
                strategies.append({
                    "strategy": "Supplier Rationalization",
                    "current_state": f"{len(low_frequency_suppliers)} suppliers with <3 orders",
                    "recommendation": "Consolidate or eliminate low-frequency suppliers",
                    "expected_improvement": "40-50% reduction in supplier management overhead",
                    "implementation_steps": [
                        "Identify alternative suppliers for low-frequency items",
                        "Negotiate expanded service offerings with existing suppliers",
                        "Implement supplier performance scorecards"
                    ]
                })
            
            # Digital automation
            strategies.append({
                "strategy": "Digital Process Automation",
                "current_state": "Manual procurement processes",
                "recommendation": "Implement e-procurement and automated workflows",
                "expected_improvement": "60-70% reduction in processing time",
                "implementation_steps": [
                    "Deploy e-procurement platform",
                    "Automate approval workflows",
                    "Implement electronic invoicing",
                    "Set up automated reordering for standard items"
                ]
            })
            
            # Strategic sourcing
            strategies.append({
                "strategy": "Strategic Sourcing Framework",
                "current_state": "Reactive purchasing approach",
                "recommendation": "Implement category-based strategic sourcing",
                "expected_improvement": "10-15% cost reduction through better negotiations",
                "implementation_steps": [
                    "Categorize spend into strategic categories",
                    "Develop category-specific sourcing strategies",
                    "Implement supplier development programs",
                    "Establish long-term contracts with key suppliers"
                ]
            })
            
            return {
                "process_optimization_strategies": strategies,
                "current_inefficiencies": {
                    "small_orders": len(small_transactions),
                    "low_frequency_suppliers": len(low_frequency_suppliers),
                    "average_daily_transactions": round(daily_transactions[('voucher_number', 'nunique')].mean(), 2)
                },
                "optimization_priorities": [
                    "Automate routine purchasing processes",
                    "Consolidate supplier base",
                    "Implement strategic sourcing",
                    "Enhance supplier relationships"
                ],
                "expected_roi": {
                    "cost_reduction": "15-25%",
                    "efficiency_gain": "50-70%",
                    "quality_improvement": "30-40%"
                }
            }
            
        except Exception as e:
            return {"error": f"Process optimization strategies failed: {str(e)}"}
    
    def generate_inventory_planning_recommendations(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate inventory planning recommendations"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Inventory planning analysis
            inventory_data = data[(data['amount'] > 0) & (data['item'].notna())].copy()
            
            if len(inventory_data) == 0:
                return {"error": "No inventory data available"}
            
            inventory_data['date'] = pd.to_datetime(inventory_data['date'])
            
            # Item analysis
            item_metrics = inventory_data.groupby('item').agg({
                'quantity': ['sum', 'mean', 'count'],
                'amount': ['sum', 'mean'],
                'rate': ['mean', 'std'],
                'date': ['min', 'max']
            })
            
            # Flatten column names
            item_metrics.columns = ['_'.join(col).strip() for col in item_metrics.columns]
            
            # ABC analysis for inventory
            total_value = item_metrics['amount_sum'].sum()
            item_metrics['value_share'] = item_metrics['amount_sum'] / total_value * 100
            item_metrics_sorted = item_metrics.sort_values('amount_sum', ascending=False)
            cumulative_share = item_metrics_sorted['value_share'].cumsum()
            
            # Categorize items
            item_metrics_sorted['abc_category'] = 'C'
            item_metrics_sorted.loc[cumulative_share <= 80, 'abc_category'] = 'A'
            item_metrics_sorted.loc[(cumulative_share > 80) & (cumulative_share <= 95), 'abc_category'] = 'B'
            
            # Purchase frequency analysis
            item_metrics_sorted['purchase_frequency'] = item_metrics_sorted['quantity_count'] / (
                (pd.to_datetime(item_metrics_sorted['date_max']) - pd.to_datetime(item_metrics_sorted['date_min'])).dt.days + 1
            ) * 30  # Monthly frequency
            
            # Generate recommendations
            recommendations = []
            
            # A-category items
            a_items = item_metrics_sorted[item_metrics_sorted['abc_category'] == 'A']
            recommendations.append({
                "category": "A-Category Items (High Value)",
                "items_count": len(a_items),
                "strategy": "Just-in-Time (JIT) Procurement",
                "recommendations": [
                    "Implement frequent, smaller orders to minimize carrying costs",
                    "Establish supplier partnerships for reliable delivery",
                    "Monitor daily to avoid stockouts",
                    "Negotiate volume discounts despite smaller order sizes"
                ],
                "review_frequency": "Weekly",
                "target_inventory_days": "7-14 days"
            })
            
            # B-category items
            b_items = item_metrics_sorted[item_metrics_sorted['abc_category'] == 'B']
            recommendations.append({
                "category": "B-Category Items (Medium Value)",
                "items_count": len(b_items),
                "strategy": "Economic Order Quantity (EOQ) Model",
                "recommendations": [
                    "Calculate optimal order quantities to balance ordering and carrying costs",
                    "Implement periodic review system",
                    "Maintain safety stock for demand variations",
                    "Group orders for efficiency"
                ],
                "review_frequency": "Bi-weekly",
                "target_inventory_days": "30-45 days"
            })
            
            # C-category items
            c_items = item_metrics_sorted[item_metrics_sorted['abc_category'] == 'C']
            recommendations.append({
                "category": "C-Category Items (Low Value)",
                "items_count": len(c_items),
                "strategy": "Bulk Purchasing",
                "recommendations": [
                    "Purchase in larger quantities to reduce ordering frequency",
                    "Focus on reducing administrative costs",
                    "Consider blanket orders or consignment arrangements",
                    "Review quarterly to eliminate obsolete items"
                ],
                "review_frequency": "Quarterly",
                "target_inventory_days": "60-90 days"
            })
            
            # Seasonal planning
            monthly_purchases = inventory_data.groupby(inventory_data['date'].dt.month)['quantity'].sum()
            peak_month = monthly_purchases.idxmax()
            low_month = monthly_purchases.idxmin()
            
            seasonal_recommendations = {
                "peak_season_planning": f"Increase inventory levels by 20-30% before month {peak_month}",
                "low_season_strategy": f"Reduce inventory and focus on clearing slow-moving items in month {low_month}",
                "supplier_communication": "Share seasonal forecasts with key suppliers for better planning"
            }
            
            return {
                "inventory_planning_recommendations": recommendations,
                "abc_analysis": {
                    "a_items": len(a_items),
                    "b_items": len(b_items),
                    "c_items": len(c_items)
                },
                "seasonal_planning": seasonal_recommendations,
                "optimization_opportunities": {
                    "total_inventory_value": round(total_value, 2),
                    "high_frequency_items": len(item_metrics_sorted[item_metrics_sorted['purchase_frequency'] > 5]),
                    "potential_cost_savings": round(total_value * 0.10, 2)  # 10% savings through optimization
                }
            }
            
        except Exception as e:
            return {"error": f"Inventory planning recommendations failed: {str(e)}"}
    
    def generate_contract_negotiation_strategies(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate contract negotiation strategies"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Contract negotiation analysis
            supplier_data = data[data['amount'] > 0].copy()
            supplier_data['date'] = pd.to_datetime(supplier_data['date'])
            
            # Supplier leverage analysis
            supplier_metrics = supplier_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean'],
                'date': ['min', 'max'],
                'voucher_number': 'nunique'
            })
            
            # Flatten column names
            supplier_metrics.columns = ['_'.join(col).strip() for col in supplier_metrics.columns]
            
            # Calculate negotiation leverage
            total_spending = supplier_metrics['amount_sum'].sum()
            supplier_metrics['spending_share'] = supplier_metrics['amount_sum'] / total_spending * 100
            supplier_metrics['relationship_duration'] = (
                pd.to_datetime(supplier_metrics['date_max']) - 
                pd.to_datetime(supplier_metrics['date_min'])
            ).dt.days
            
            # Categorize suppliers for negotiation
            high_leverage = supplier_metrics[supplier_metrics['spending_share'] > 5]  # >5% of total spend
            medium_leverage = supplier_metrics[
                (supplier_metrics['spending_share'] >= 1) & 
                (supplier_metrics['spending_share'] <= 5)
            ]
            low_leverage = supplier_metrics[supplier_metrics['spending_share'] < 1]
            
            # Generate negotiation strategies
            negotiation_strategies = []
            
            # High leverage suppliers
            for supplier in high_leverage.index[:5]:  # Top 5 high leverage
                supplier_info = high_leverage.loc[supplier]
                negotiation_strategies.append({
                    "supplier": supplier,
                    "leverage_level": "High",
                    "spending_share": round(supplier_info['spending_share'], 2),
                    "strategy": "Strategic Partnership Negotiation",
                    "negotiation_points": [
                        "Volume-based pricing tiers",
                        "Extended payment terms (45-60 days)",
                        "Performance-based rebates",
                        "Innovation collaboration agreements",
                        "Exclusive supplier arrangements for key categories"
                    ],
                    "expected_savings": round(supplier_info['amount_sum'] * 0.08, 2),  # 8% savings
                    "contract_duration": "2-3 years"
                })
            
            # Medium leverage suppliers
            medium_count = min(3, len(medium_leverage))
            for supplier in medium_leverage.index[:medium_count]:
                supplier_info = medium_leverage.loc[supplier]
                negotiation_strategies.append({
                    "supplier": supplier,
                    "leverage_level": "Medium",
                    "spending_share": round(supplier_info['spending_share'], 2),
                    "strategy": "Competitive Negotiation",
                    "negotiation_points": [
                        "Price benchmarking against competitors",
                        "Service level agreements",
                        "Bulk order discounts",
                        "Standardized payment terms (30 days)",
                        "Annual price review mechanisms"
                    ],
                    "expected_savings": round(supplier_info['amount_sum'] * 0.05, 2),  # 5% savings
                    "contract_duration": "1-2 years"
                })
            
            # Contract optimization recommendations
            contract_optimization = {
                "payment_terms": {
                    "current_average": "Immediate to 30 days",
                    "target": "45-60 days for strategic suppliers, 30 days for others",
                    "cash_flow_benefit": round(total_spending * 0.02, 2)  # 2% cost of capital benefit
                },
                "price_protection": {
                    "recommendation": "Implement price escalation clauses tied to inflation indices",
                    "risk_mitigation": "Protect against sudden price increases"
                },
                "performance_metrics": {
                    "delivery_performance": "98% on-time delivery",
                    "quality_standards": "Zero defect tolerance with penalty clauses",
                    "service_levels": "24-hour response time for issues"
                }
            }
            
            # Negotiation timeline
            negotiation_timeline = [
                {
                    "phase": "Preparation (Month 1)",
                    "activities": [
                        "Gather spend analysis and market intelligence",
                        "Identify alternative suppliers",
                        "Develop negotiation strategies by supplier category"
                    ]
                },
                {
                    "phase": "Negotiation (Months 2-3)",
                    "activities": [
                        "Conduct negotiations with high-leverage suppliers first",
                        "Use competitive tension for medium-leverage suppliers",
                        "Implement pilot programs for new terms"
                    ]
                },
                {
                    "phase": "Implementation (Months 4-6)",
                    "activities": [
                        "Execute new contracts",
                        "Implement performance tracking systems",
                        "Monitor supplier compliance and benefits realization"
                    ]
                }
            ]
            
            return {
                "negotiation_strategies": negotiation_strategies,
                "contract_optimization": contract_optimization,
                "implementation_timeline": negotiation_timeline,
                "expected_benefits": {
                    "total_potential_savings": round(sum([s['expected_savings'] for s in negotiation_strategies]), 2),
                    "cash_flow_improvement": round(total_spending * 0.02, 2),
                    "risk_reduction": "Improved supplier performance and price stability"
                },
                "success_metrics": [
                    "Cost savings achievement vs. target",
                    "Supplier performance improvement",
                    "Contract compliance rates",
                    "Payment term optimization"
                ]
            }
            
        except Exception as e:
            return {"error": f"Contract negotiation strategies failed: {str(e)}"}
    
    def generate_risk_mitigation_strategies(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk mitigation strategies"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Risk analysis
            risk_data = data[data['amount'] > 0].copy()
            risk_data['date'] = pd.to_datetime(risk_data['date'])
            
            # Supplier concentration risk
            supplier_concentration = risk_data.groupby('party_name')['amount'].sum()
            total_spending = supplier_concentration.sum()
            supplier_concentration_pct = supplier_concentration / total_spending * 100
            
            # Identify risks
            risks_identified = []
            
            # Supplier concentration risk
            high_concentration_suppliers = supplier_concentration_pct[supplier_concentration_pct > 20]
            if len(high_concentration_suppliers) > 0:
                risks_identified.append({
                    "risk_type": "Supplier Concentration Risk",
                    "severity": "High",
                    "description": f"{len(high_concentration_suppliers)} suppliers account for >20% of spending each",
                    "potential_impact": "Supply disruption, price manipulation, dependency",
                    "mitigation_strategies": [
                        "Develop alternative suppliers for critical categories",
                        "Implement dual sourcing for high-risk items",
                        "Create supplier diversity programs",
                        "Establish emergency supplier protocols"
                    ],
                    "implementation_priority": "High",
                    "timeline": "3-6 months"
                })
            
            # Price volatility risk
            if 'rate' in risk_data.columns:
                item_price_volatility = risk_data.groupby('item')['rate'].std()
                high_volatility_items = item_price_volatility[item_price_volatility > item_price_volatility.quantile(0.8)]
                
                if len(high_volatility_items) > 0:
                    risks_identified.append({
                        "risk_type": "Price Volatility Risk",
                        "severity": "Medium",
                        "description": f"{len(high_volatility_items)} items show high price volatility",
                        "potential_impact": "Budget overruns, unpredictable costs",
                        "mitigation_strategies": [
                            "Implement fixed-price contracts for volatile items",
                            "Use price escalation clauses tied to market indices",
                            "Develop strategic inventory buffers",
                            "Implement cost-plus contracts with caps"
                        ],
                        "implementation_priority": "Medium",
                        "timeline": "2-4 months"
                    })
            
            # Supplier relationship risk
            current_date = risk_data['date'].max()
            supplier_last_purchase = risk_data.groupby('party_name')['date'].max()
            inactive_suppliers = supplier_last_purchase[
                (current_date - supplier_last_purchase).dt.days > 90
            ]
            
            if len(inactive_suppliers) > 0:
                risks_identified.append({
                    "risk_type": "Supplier Relationship Risk",
                    "severity": "Medium",
                    "description": f"{len(inactive_suppliers)} suppliers inactive for >90 days",
                    "potential_impact": "Loss of supplier relationships, reduced negotiation power",
                    "mitigation_strategies": [
                        "Implement regular supplier communication programs",
                        "Develop supplier performance scorecards",
                        "Create supplier development initiatives",
                        "Establish quarterly business reviews"
                    ],
                    "implementation_priority": "Medium",
                    "timeline": "1-3 months"
                })
            
            # Quality risk
            risks_identified.append({
                "risk_type": "Quality and Compliance Risk",
                "severity": "High",
                "description": "Potential quality issues and regulatory compliance gaps",
                "potential_impact": "Product recalls, regulatory penalties, reputation damage",
                "mitigation_strategies": [
                    "Implement supplier quality audits",
                    "Establish quality certification requirements",
                    "Create incoming inspection protocols",
                    "Develop supplier training programs"
                ],
                "implementation_priority": "High",
                "timeline": "2-4 months"
            })
            
            # Financial risk
            risks_identified.append({
                "risk_type": "Supplier Financial Risk",
                "severity": "Medium",
                "description": "Risk of supplier financial instability",
                "potential_impact": "Supply interruption, payment disputes, contract breaches",
                "mitigation_strategies": [
                    "Conduct regular financial health assessments",
                    "Implement supplier financial monitoring",
                    "Diversify supplier base to reduce dependency",
                    "Establish backup suppliers for critical items"
                ],
                "implementation_priority": "Medium",
                "timeline": "3-6 months"
            })
            
            # Risk mitigation action plan
            action_plan = {
                "immediate_actions": [
                    "Identify and qualify alternative suppliers for high-concentration categories",
                    "Implement supplier performance monitoring dashboard",
                    "Develop emergency supplier contact protocols"
                ],
                "short_term_actions": [
                    "Negotiate dual sourcing agreements",
                    "Implement price monitoring and alerting systems",
                    "Establish supplier financial health monitoring"
                ],
                "long_term_actions": [
                    "Develop strategic supplier partnerships",
                    "Implement comprehensive supplier development programs",
                    "Create integrated risk management framework"
                ]
            }
            
            return {
                "risk_assessment": risks_identified,
                "risk_mitigation_action_plan": action_plan,
                "risk_monitoring_kpis": [
                    "Supplier concentration index",
                    "Price volatility index",
                    "Supplier performance scores",
                    "Supply chain disruption incidents",
                    "Contract compliance rates"
                ],
                "contingency_planning": {
                    "emergency_suppliers": "Maintain list of pre-qualified emergency suppliers",
                    "inventory_buffers": "Strategic inventory for critical items",
                    "communication_protocols": "Clear escalation procedures for supply issues"
                }
            }
            
        except Exception as e:
            return {"error": f"Risk mitigation strategies failed: {str(e)}"}
    
    def generate_budget_allocation_optimization(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate budget allocation optimization strategies"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Budget analysis
            budget_data = data[data['amount'] > 0].copy()
            budget_data['date'] = pd.to_datetime(budget_data['date'])
            
            # Historical spending patterns
            monthly_spending = budget_data.groupby(budget_data['date'].dt.to_period('M'))['amount'].sum()
            
            # Category-wise analysis
            if 'item' in budget_data.columns:
                category_spending = budget_data.groupby('item')['amount'].sum()
                total_spending = category_spending.sum()
                category_allocation = category_spending / total_spending * 100
            else:
                category_spending = budget_data.groupby('party_name')['amount'].sum()
                total_spending = category_spending.sum()
                category_allocation = category_spending / total_spending * 100
            
            # Budget optimization recommendations
            optimization_strategies = []
            
            # Top categories analysis
            top_categories = category_allocation.nlargest(10)
            
            for category in top_categories.index[:5]:
                category_pct = category_allocation[category]
                optimization_strategies.append({
                    "category": category,
                    "current_allocation": round(category_pct, 2),
                    "spending_amount": round(category_spending[category], 2),
                    "optimization_strategy": self._get_category_strategy(category_pct),
                    "recommended_actions": self._get_category_actions(category_pct),
                    "target_savings": round(category_spending[category] * 0.05, 2)  # 5% target
                })
            
            # Seasonal budget planning
            seasonal_analysis = self._analyze_seasonal_patterns(monthly_spending)
            
            # Budget allocation framework
            allocation_framework = {
                "strategic_categories": {
                    "allocation": "60-70%",
                    "criteria": "High value, critical to operations",
                    "management_approach": "Strategic partnerships, long-term contracts"
                },
                "tactical_categories": {
                    "allocation": "20-30%",
                    "criteria": "Medium value, regular consumption",
                    "management_approach": "Competitive sourcing, annual contracts"
                },
                "operational_categories": {
                    "allocation": "10-20%",
                    "criteria": "Low value, high volume",
                    "management_approach": "Simplified procurement, bulk purchases"
                }
            }
            
            # Budget monitoring and control
            budget_controls = {
                "monthly_review": "Track spending vs. budget by category",
                "variance_analysis": "Investigate variances >5% of budget",
                "approval_limits": "Define approval thresholds by spending category",
                "forecasting": "Update quarterly forecasts based on actuals"
            }
            
            return {
                "budget_optimization_strategies": optimization_strategies,
                "seasonal_budget_planning": seasonal_analysis,
                "allocation_framework": allocation_framework,
                "budget_controls": budget_controls,
                "recommended_budget_allocation": {
                    "total_annual_budget": round(monthly_spending.mean() * 12, 2),
                    "monthly_baseline": round(monthly_spending.mean(), 2),
                    "contingency_reserve": round(monthly_spending.mean() * 12 * 0.05, 2)  # 5% contingency
                },
                "optimization_benefits": {
                    "cost_savings": round(total_spending * 0.08, 2),  # 8% savings target
                    "budget_accuracy": "Improved forecasting accuracy by 15-20%",
                    "spending_visibility": "Real-time spending tracking and control"
                }
            }
            
        except Exception as e:
            return {"error": f"Budget allocation optimization failed: {str(e)}"}
    
    def generate_strategic_sourcing_recommendations(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic sourcing recommendations"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Strategic sourcing analysis
            sourcing_data = data[data['amount'] > 0].copy()
            sourcing_data['date'] = pd.to_datetime(sourcing_data['date'])
            
            # Spend analysis
            total_spend = sourcing_data['amount'].sum()
            
            # Category analysis
            if 'item' in sourcing_data.columns:
                category_spend = sourcing_data.groupby('item').agg({
                    'amount': 'sum',
                    'party_name': 'nunique',
                    'voucher_number': 'nunique'
                })
            else:
                category_spend = sourcing_data.groupby('party_name').agg({
                    'amount': 'sum',
                    'voucher_number': 'nunique'
                })
                category_spend['party_name'] = 1  # Placeholder
            
            category_spend['spend_share'] = category_spend['amount'] / total_spend * 100
            
            # Strategic sourcing opportunities
            sourcing_opportunities = []
            
            # High-value categories
            high_value_categories = category_spend[category_spend['spend_share'] > 5]
            
            for category in high_value_categories.index[:5]:
                category_info = high_value_categories.loc[category]
                sourcing_opportunities.append({
                    "category": category,
                    "spend_amount": round(category_info['amount'], 2),
                    "spend_share": round(category_info['spend_share'], 2),
                    "sourcing_strategy": "Strategic Sourcing",
                    "approach": [
                        "Conduct detailed market analysis",
                        "Develop comprehensive RFP process",
                        "Negotiate long-term strategic partnerships",
                        "Implement performance-based contracts"
                    ],
                    "expected_savings": round(category_info['amount'] * 0.12, 2),  # 12% savings
                    "implementation_timeline": "6-9 months"
                })
            
            # Medium-value categories
            medium_value_categories = category_spend[
                (category_spend['spend_share'] >= 1) & 
                (category_spend['spend_share'] <= 5)
            ]
            
            for category in medium_value_categories.index[:3]:
                category_info = medium_value_categories.loc[category]
                sourcing_opportunities.append({
                    "category": category,
                    "spend_amount": round(category_info['amount'], 2),
                    "spend_share": round(category_info['spend_share'], 2),
                    "sourcing_strategy": "Competitive Sourcing",
                    "approach": [
                        "Multi-supplier competitive bidding",
                        "Standardize specifications",
                        "Negotiate annual contracts",
                        "Implement supplier scorecards"
                    ],
                    "expected_savings": round(category_info['amount'] * 0.08, 2),  # 8% savings
                    "implementation_timeline": "3-6 months"
                })
            
            # Sourcing strategy framework
            strategy_framework = {
                "spend_analysis": {
                    "description": "Comprehensive analysis of spending patterns",
                    "frequency": "Annual with quarterly updates",
                    "tools": "Spend analytics software, category management"
                },
                "market_intelligence": {
                    "description": "Continuous monitoring of supplier markets",
                    "frequency": "Ongoing",
                    "tools": "Market research, industry reports, supplier assessments"
                },
                "supplier_management": {
                    "description": "Strategic supplier relationship management",
                    "frequency": "Ongoing",
                    "tools": "Supplier scorecards, regular business reviews, development programs"
                },
                "contract_management": {
                    "description": "Optimize contract terms and performance",
                    "frequency": "Annual review",
                    "tools": "Contract management system, performance tracking"
                }
            }
            
            # Implementation roadmap
            implementation_roadmap = [
                {
                    "phase": "Assessment (Months 1-2)",
                    "activities": [
                        "Complete comprehensive spend analysis",
                        "Categorize spending into strategic buckets",
                        "Assess current supplier performance",
                        "Identify sourcing opportunities"
                    ]
                },
                {
                    "phase": "Strategy Development (Months 3-4)",
                    "activities": [
                        "Develop category-specific sourcing strategies",
                        "Design RFP processes and evaluation criteria",
                        "Create supplier selection frameworks",
                        "Establish performance metrics"
                    ]
                },
                {
                    "phase": "Execution (Months 5-12)",
                    "activities": [
                        "Execute sourcing events by category",
                        "Negotiate and finalize contracts",
                        "Implement new supplier relationships",
                        "Monitor performance and benefits realization"
                    ]
                }
            ]
            
            return {
                "sourcing_opportunities": sourcing_opportunities,
                "strategy_framework": strategy_framework,
                "implementation_roadmap": implementation_roadmap,
                "success_metrics": [
                    "Cost savings achievement",
                    "Supplier performance improvement",
                    "Contract compliance rates",
                    "Market competitiveness"
                ],
                "expected_benefits": {
                    "total_savings": round(sum([opp['expected_savings'] for opp in sourcing_opportunities]), 2),
                    "cost_avoidance": round(total_spend * 0.05, 2),  # 5% cost avoidance
                    "supplier_performance": "20-30% improvement in key metrics",
                    "process_efficiency": "40-50% reduction in sourcing cycle time"
                }
            }
            
        except Exception as e:
            return {"error": f"Strategic sourcing recommendations failed: {str(e)}"}
    
    def generate_general_purchase_recommendations(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate general purchase recommendations"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # General analysis
            purchase_data = data[data['amount'] > 0]
            
            total_spending = purchase_data['amount'].sum()
            unique_suppliers = purchase_data['party_name'].nunique()
            
            # General recommendations
            recommendations = [
                {
                    "area": "Supplier Management",
                    "recommendation": "Implement supplier performance management system",
                    "benefit": "Improved supplier relationships and performance"
                },
                {
                    "area": "Cost Management",
                    "recommendation": "Establish spend analytics and reporting",
                    "benefit": "Better visibility and control over procurement costs"
                },
                {
                    "area": "Process Optimization",
                    "recommendation": "Digitize procurement processes",
                    "benefit": "Increased efficiency and reduced processing time"
                },
                {
                    "area": "Strategic Sourcing",
                    "recommendation": "Develop category-based sourcing strategies",
                    "benefit": "Better negotiation outcomes and supplier relationships"
                }
            ]
            
            return {
                "general_recommendations": recommendations,
                "spending_overview": {
                    "total_spending": round(total_spending, 2),
                    "unique_suppliers": unique_suppliers,
                    "avg_transaction_value": round(purchase_data['amount'].mean(), 2)
                },
                "next_steps": [
                    "Conduct detailed spend analysis",
                    "Assess current supplier performance",
                    "Develop procurement strategy",
                    "Implement recommended improvements"
                ]
            }
            
        except Exception as e:
            return {"error": f"General purchase recommendations failed: {str(e)}"}
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the type of prescriptive analysis based on query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['cost', 'optimization', 'savings', 'reduce']):
            return 'cost_optimization_strategies'
        elif any(word in query_lower for word in ['supplier', 'consolidation', 'rationalization']):
            return 'supplier_consolidation_plan'
        elif any(word in query_lower for word in ['process', 'efficiency', 'workflow']):
            return 'procurement_process_optimization'
        elif any(word in query_lower for word in ['inventory', 'planning', 'stock']):
            return 'inventory_planning_recommendations'
        elif any(word in query_lower for word in ['contract', 'negotiation', 'terms']):
            return 'contract_negotiation_strategies'
        elif any(word in query_lower for word in ['risk', 'mitigation', 'contingency']):
            return 'risk_mitigation_strategies'
        elif any(word in query_lower for word in ['budget', 'allocation', 'planning']):
            return 'budget_allocation_optimization'
        elif any(word in query_lower for word in ['strategic', 'sourcing', 'category']):
            return 'strategic_sourcing_recommendations'
        else:
            return 'general_recommendations'
    
    def _create_implementation_roadmap(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create implementation roadmap for strategies"""
        roadmap = []
        
        # Sort strategies by priority
        high_priority = [s for s in strategies if s.get('implementation_priority') == 'High']
        medium_priority = [s for s in strategies if s.get('implementation_priority') == 'Medium']
        
        if high_priority:
            roadmap.append({
                "phase": "Phase 1 (Months 1-3)",
                "priority": "High",
                "strategies": [s['strategy'] for s in high_priority],
                "expected_savings": sum([s['potential_savings'] for s in high_priority])
            })
        
        if medium_priority:
            roadmap.append({
                "phase": "Phase 2 (Months 4-6)",
                "priority": "Medium", 
                "strategies": [s['strategy'] for s in medium_priority],
                "expected_savings": sum([s['potential_savings'] for s in medium_priority])
            })
        
        return roadmap
    
    def _get_category_strategy(self, allocation_pct: float) -> str:
        """Get optimization strategy based on allocation percentage"""
        if allocation_pct > 15:
            return "Strategic Focus - Develop partnerships and long-term agreements"
        elif allocation_pct > 5:
            return "Competitive Sourcing - Multi-supplier approach with regular reviews"
        else:
            return "Simplified Procurement - Streamlined processes and bulk purchasing"
    
    def _get_category_actions(self, allocation_pct: float) -> List[str]:
        """Get specific actions based on allocation percentage"""
        if allocation_pct > 15:
            return [
                "Conduct market analysis",
                "Develop supplier partnerships",
                "Negotiate long-term contracts",
                "Implement joint improvement programs"
            ]
        elif allocation_pct > 5:
            return [
                "Regular competitive bidding",
                "Supplier performance monitoring",
                "Annual contract reviews",
                "Cost benchmarking"
            ]
        else:
            return [
                "Consolidate suppliers",
                "Implement e-procurement",
                "Bulk purchasing agreements",
                "Simplified approval processes"
            ]
    
    def _analyze_seasonal_patterns(self, monthly_spending: pd.Series) -> Dict[str, Any]:
        """Analyze seasonal spending patterns"""
        if len(monthly_spending) < 12:
            return {"error": "Insufficient data for seasonal analysis"}
        
        # Calculate seasonal indices
        avg_spending = monthly_spending.mean()
        seasonal_indices = {}
        
        for month in range(1, 13):
            month_data = monthly_spending[monthly_spending.index.month == month]
            if len(month_data) > 0:
                seasonal_indices[month] = month_data.mean() / avg_spending
            else:
                seasonal_indices[month] = 1.0
        
        peak_month = max(seasonal_indices, key=seasonal_indices.get)
        low_month = min(seasonal_indices, key=seasonal_indices.get)
        
        return {
            "seasonal_indices": seasonal_indices,
            "peak_month": peak_month,
            "low_month": low_month,
            "seasonality_strength": max(seasonal_indices.values()) - min(seasonal_indices.values()),
            "budget_recommendations": {
                "peak_season": f"Increase budget by {(seasonal_indices[peak_month] - 1) * 100:.0f}% in month {peak_month}",
                "low_season": f"Reduce budget by {(1 - seasonal_indices[low_month]) * 100:.0f}% in month {low_month}"
            }
        }