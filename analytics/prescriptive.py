"""
Prescriptive Analytics - "What should we do?"
Actionable recommendations and optimization
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.optimize import minimize, linprog
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .base import AnalyticsBase, AnalyticsResponse


class PrescriptiveAnalytics(AnalyticsBase):
    """Prescriptive analytics for optimization and recommendations"""
    
    def __init__(self, agent_type: str = "financial"):
        super().__init__("PrescriptiveAnalytics", agent_type)
        self.supported_queries = [
            'cash_flow_optimization',
            'resource_allocation',
            'action_prioritization',
            'risk_mitigation',
            'cost_optimization',
            'performance_improvement'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform prescriptive analytics on the provided data"""
        start_time = datetime.now()
        
        analysis_type = self.classify_analysis_type(query)
        
        if analysis_type == 'cash_flow_optimization':
            results = self.optimize_cash_flow(data, params)
        elif analysis_type == 'resource_allocation':
            results = self.optimize_resource_allocation(data, params)
        elif analysis_type == 'action_prioritization':
            results = self.prioritize_actions(data, params)
        elif analysis_type == 'risk_mitigation':
            results = self.recommend_risk_mitigation(data, params)
        elif analysis_type == 'cost_optimization':
            results = self.optimize_costs(data, params)
        elif analysis_type == 'performance_improvement':
            results = self.recommend_performance_improvements(data, params)
        else:
            results = self.general_recommendations(data, params)
        
        return self.prepare_response(
            analytics_type='prescriptive',
            query=query,
            results=results,
            start_time=start_time,
            confidence_level=0.80
        )
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the specific type of prescriptive analysis needed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['cash flow', 'cashflow', 'liquidity']):
            return 'cash_flow_optimization'
        elif any(word in query_lower for word in ['resource', 'allocation', 'distribute']):
            return 'resource_allocation'
        elif any(word in query_lower for word in ['prioritize', 'priority', 'order']):
            return 'action_prioritization'
        elif any(word in query_lower for word in ['risk', 'mitigation', 'reduce']):
            return 'risk_mitigation'
        elif any(word in query_lower for word in ['cost', 'expense', 'reduce']):
            return 'cost_optimization'
        elif any(word in query_lower for word in ['performance', 'improve', 'optimize']):
            return 'performance_improvement'
        else:
            return 'general_recommendations'
    
    def optimize_cash_flow(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize cash flow through actionable recommendations"""
        required_columns = ['account_name', 'amount', 'date']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for cash flow optimization'}
        
        data['date'] = pd.to_datetime(data['date'])
        
        # Analyze current cash flow patterns
        cash_flow_analysis = self.analyze_cash_flow_patterns(data)
        
        # Identify optimization opportunities
        opportunities = self.identify_cash_flow_opportunities(data, cash_flow_analysis)
        
        # Generate optimization plan
        optimization_plan = self.create_cash_flow_optimization_plan(opportunities, params)
        
        # Calculate expected impact
        expected_impact = self.calculate_optimization_impact(optimization_plan, data)
        
        return {
            'current_state': cash_flow_analysis,
            'optimization_opportunities': opportunities,
            'optimization_plan': optimization_plan,
            'expected_impact': expected_impact,
            'implementation_roadmap': self.create_implementation_roadmap(optimization_plan),
            'success_metrics': self.define_success_metrics(expected_impact)
        }
    
    def optimize_resource_allocation(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation across different areas"""
        required_columns = ['resource_type', 'current_allocation', 'performance_metric']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for resource allocation optimization'}
        
        # Analyze current allocation efficiency
        efficiency_analysis = self.analyze_allocation_efficiency(data)
        
        # Perform optimization
        optimal_allocation = self.calculate_optimal_allocation(data, params)
        
        # Calculate reallocation recommendations
        reallocation_plan = self.create_reallocation_plan(data, optimal_allocation)
        
        return {
            'current_efficiency': efficiency_analysis,
            'optimal_allocation': optimal_allocation,
            'reallocation_plan': reallocation_plan,
            'expected_improvement': self.calculate_allocation_improvement(reallocation_plan),
            'implementation_steps': self.create_reallocation_steps(reallocation_plan)
        }
    
    def prioritize_actions(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize actions based on impact and feasibility"""
        required_columns = ['action_name', 'impact_score', 'effort_score']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for action prioritization'}
        
        # Calculate priority scores
        priority_scores = self.calculate_priority_scores(data, params)
        
        # Create priority matrix
        priority_matrix = self.create_priority_matrix(priority_scores)
        
        # Generate prioritized action plan
        action_plan = self.create_prioritized_action_plan(priority_matrix, params)
        
        return {
            'priority_scores': priority_scores,
            'priority_matrix': priority_matrix,
            'prioritized_actions': action_plan,
            'resource_requirements': self.calculate_resource_requirements(action_plan),
            'timeline': self.create_action_timeline(action_plan)
        }
    
    def recommend_risk_mitigation(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend risk mitigation strategies"""
        # Risk assessment
        risk_assessment = self.assess_current_risks(data)
        
        # Generate mitigation strategies
        mitigation_strategies = self.generate_mitigation_strategies(risk_assessment, params)
        
        # Prioritize strategies
        prioritized_strategies = self.prioritize_mitigation_strategies(mitigation_strategies)
        
        return {
            'risk_assessment': risk_assessment,
            'mitigation_strategies': mitigation_strategies,
            'prioritized_strategies': prioritized_strategies,
            'implementation_plan': self.create_risk_mitigation_plan(prioritized_strategies),
            'monitoring_framework': self.create_risk_monitoring_framework(prioritized_strategies)
        }
    
    def optimize_costs(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize costs through strategic recommendations"""
        required_columns = ['cost_category', 'amount', 'variability']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for cost optimization'}
        
        # Analyze cost structure
        cost_analysis = self.analyze_cost_structure(data)
        
        # Identify cost reduction opportunities
        reduction_opportunities = self.identify_cost_reduction_opportunities(data, cost_analysis)
        
        # Create optimization plan
        optimization_plan = self.create_cost_optimization_plan(reduction_opportunities, params)
        
        return {
            'cost_analysis': cost_analysis,
            'reduction_opportunities': reduction_opportunities,
            'optimization_plan': optimization_plan,
            'projected_savings': self.calculate_projected_savings(optimization_plan),
            'implementation_roadmap': self.create_cost_reduction_roadmap(optimization_plan)
        }
    
    def recommend_performance_improvements(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend performance improvements"""
        # Analyze current performance
        performance_analysis = self.analyze_current_performance(data)
        
        # Identify improvement opportunities
        improvement_opportunities = self.identify_improvement_opportunities(performance_analysis)
        
        # Create improvement plan
        improvement_plan = self.create_performance_improvement_plan(improvement_opportunities, params)
        
        return {
            'performance_analysis': performance_analysis,
            'improvement_opportunities': improvement_opportunities,
            'improvement_plan': improvement_plan,
            'expected_outcomes': self.project_improvement_outcomes(improvement_plan),
            'success_framework': self.create_performance_success_framework(improvement_plan)
        }
    
    def general_recommendations(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General recommendations based on data analysis"""
        # Basic data analysis
        data_insights = self.analyze_data_patterns(data)
        
        # Generate general recommendations
        recommendations = self.generate_general_recommendations(data_insights, params)
        
        return {
            'data_insights': data_insights,
            'recommendations': recommendations,
            'implementation_suggestions': self.create_general_implementation_suggestions(recommendations)
        }
    
    # Helper methods for specific optimization tasks
    
    def analyze_cash_flow_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current cash flow patterns"""
        # Calculate key metrics
        total_inflows = data[data['amount'] > 0]['amount'].sum()
        total_outflows = abs(data[data['amount'] < 0]['amount'].sum())
        net_flow = total_inflows - total_outflows
        
        # Analyze by account type
        account_patterns = data.groupby('account_name')['amount'].agg(['sum', 'count', 'mean']).round(2)
        
        # Identify cash flow drivers
        top_inflows = account_patterns[account_patterns['sum'] > 0].nlargest(5, 'sum')
        top_outflows = account_patterns[account_patterns['sum'] < 0].nsmallest(5, 'sum')
        
        return {
            'summary': {
                'total_inflows': total_inflows,
                'total_outflows': total_outflows,
                'net_flow': net_flow,
                'flow_ratio': total_inflows / total_outflows if total_outflows > 0 else float('inf')
            },
            'top_inflows': top_inflows.to_dict(),
            'top_outflows': top_outflows.to_dict(),
            'volatility': data['amount'].std()
        }
    
    def identify_cash_flow_opportunities(self, data: pd.DataFrame, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify cash flow optimization opportunities"""
        opportunities = []
        
        # Receivables acceleration opportunities
        if 'receivable' in ' '.join(data['account_name'].str.lower()):
            opportunities.append({
                'type': 'receivables_acceleration',
                'description': 'Accelerate receivables collection',
                'potential_impact': 'high',
                'implementation_effort': 'medium',
                'timeframe': '30-60 days'
            })
        
        # Payables optimization opportunities
        if 'payable' in ' '.join(data['account_name'].str.lower()):
            opportunities.append({
                'type': 'payables_optimization',
                'description': 'Optimize payment terms with suppliers',
                'potential_impact': 'medium',
                'implementation_effort': 'low',
                'timeframe': '15-30 days'
            })
        
        # Inventory optimization
        if 'inventory' in ' '.join(data['account_name'].str.lower()):
            opportunities.append({
                'type': 'inventory_optimization',
                'description': 'Optimize inventory levels',
                'potential_impact': 'medium',
                'implementation_effort': 'high',
                'timeframe': '60-90 days'
            })
        
        # Cash management
        opportunities.append({
            'type': 'cash_management',
            'description': 'Improve cash management processes',
            'potential_impact': 'medium',
            'implementation_effort': 'medium',
            'timeframe': '30-45 days'
        })
        
        return opportunities
    
    def create_cash_flow_optimization_plan(self, opportunities: List[Dict], params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive cash flow optimization plan"""
        # Prioritize opportunities
        prioritized_opportunities = sorted(
            opportunities, 
            key=lambda x: self.calculate_opportunity_priority(x), 
            reverse=True
        )
        
        # Create phases
        phases = {
            'immediate': [],  # 0-30 days
            'short_term': [],  # 30-90 days
            'medium_term': []  # 90+ days
        }
        
        for opp in prioritized_opportunities:
            timeframe = opp['timeframe']
            if '30' in timeframe or '15' in timeframe:
                phases['immediate'].append(opp)
            elif '60' in timeframe or '45' in timeframe:
                phases['short_term'].append(opp)
            else:
                phases['medium_term'].append(opp)
        
        return {
            'phases': phases,
            'total_opportunities': len(opportunities),
            'prioritization_method': 'impact_effort_matrix',
            'success_criteria': self.define_optimization_success_criteria()
        }
    
    def calculate_opportunity_priority(self, opportunity: Dict[str, Any]) -> float:
        """Calculate priority score for an opportunity"""
        impact_scores = {'high': 3, 'medium': 2, 'low': 1}
        effort_scores = {'low': 3, 'medium': 2, 'high': 1}
        
        impact_score = impact_scores.get(opportunity['potential_impact'], 2)
        effort_score = effort_scores.get(opportunity['implementation_effort'], 2)
        
        return impact_score * effort_score
    
    def calculate_optimization_impact(self, plan: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate expected impact of optimization plan"""
        # Estimate impact based on typical improvements
        base_flow = data['amount'].sum()
        
        # Conservative estimates
        immediate_impact = abs(base_flow) * 0.05  # 5% improvement
        short_term_impact = abs(base_flow) * 0.10  # 10% improvement
        medium_term_impact = abs(base_flow) * 0.15  # 15% improvement
        
        return {
            'immediate_impact': immediate_impact,
            'short_term_impact': short_term_impact,
            'medium_term_impact': medium_term_impact,
            'total_potential_impact': immediate_impact + short_term_impact + medium_term_impact,
            'confidence_level': 0.75
        }
    
    def create_implementation_roadmap(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create implementation roadmap"""
        roadmap = []
        
        # Process each phase
        for phase, opportunities in plan['phases'].items():
            for i, opp in enumerate(opportunities):
                roadmap.append({
                    'phase': phase,
                    'step': i + 1,
                    'action': opp['description'],
                    'type': opp['type'],
                    'timeline': opp['timeframe'],
                    'dependencies': [],
                    'success_metrics': self.define_action_success_metrics(opp)
                })
        
        return roadmap
    
    def define_success_metrics(self, impact: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define success metrics for optimization"""
        return [
            {
                'metric': 'Cash Flow Improvement',
                'target': f"{impact['total_potential_impact']:.0f}",
                'measurement': 'Monthly cash flow analysis',
                'timeline': '90 days'
            },
            {
                'metric': 'Collection Period',
                'target': '10% reduction',
                'measurement': 'Days sales outstanding',
                'timeline': '60 days'
            },
            {
                'metric': 'Payment Terms',
                'target': '5-day extension',
                'measurement': 'Average payment period',
                'timeline': '30 days'
            }
        ]
    
    def define_optimization_success_criteria(self) -> List[str]:
        """Define optimization success criteria"""
        return [
            "Positive cash flow achieved within 90 days",
            "10% reduction in collection period",
            "5% improvement in payment terms",
            "Reduced cash flow volatility"
        ]
    
    def define_action_success_metrics(self, opportunity: Dict[str, Any]) -> List[str]:
        """Define success metrics for specific actions"""
        metrics = {
            'receivables_acceleration': ['Collection period reduction', 'Bad debt ratio'],
            'payables_optimization': ['Payment terms extension', 'Supplier relationship score'],
            'inventory_optimization': ['Inventory turnover', 'Carrying cost reduction'],
            'cash_management': ['Cash conversion cycle', 'Liquidity ratio']
        }
        
        return metrics.get(opportunity['type'], ['Implementation completion', 'Process improvement'])
    
    def calculate_priority_scores(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate priority scores for actions"""
        data = data.copy()
        
        # Normalize scores
        scaler = StandardScaler()
        data['normalized_impact'] = scaler.fit_transform(data[['impact_score']])
        data['normalized_effort'] = scaler.fit_transform(data[['effort_score']])
        
        # Calculate priority score (high impact, low effort = high priority)
        data['priority_score'] = data['normalized_impact'] - data['normalized_effort']
        
        # Add priority category
        data['priority_category'] = pd.cut(
            data['priority_score'], 
            bins=3, 
            labels=['Low', 'Medium', 'High']
        )
        
        return data
    
    def create_priority_matrix(self, priority_data: pd.DataFrame) -> Dict[str, Any]:
        """Create priority matrix visualization data"""
        matrix = {
            'high_impact_low_effort': priority_data[
                (priority_data['impact_score'] > priority_data['impact_score'].median()) &
                (priority_data['effort_score'] < priority_data['effort_score'].median())
            ].to_dict('records'),
            'high_impact_high_effort': priority_data[
                (priority_data['impact_score'] > priority_data['impact_score'].median()) &
                (priority_data['effort_score'] > priority_data['effort_score'].median())
            ].to_dict('records'),
            'low_impact_low_effort': priority_data[
                (priority_data['impact_score'] < priority_data['impact_score'].median()) &
                (priority_data['effort_score'] < priority_data['effort_score'].median())
            ].to_dict('records'),
            'low_impact_high_effort': priority_data[
                (priority_data['impact_score'] < priority_data['impact_score'].median()) &
                (priority_data['effort_score'] > priority_data['effort_score'].median())
            ].to_dict('records')
        }
        
        return matrix
    
    def create_prioritized_action_plan(self, matrix: Dict[str, Any], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create prioritized action plan"""
        action_plan = []
        
        # Priority 1: High impact, low effort (quick wins)
        for action in matrix['high_impact_low_effort']:
            action_plan.append({
                'action': action['action_name'],
                'priority': 1,
                'category': 'Quick Win',
                'timeline': '0-30 days',
                'resources_needed': 'Low'
            })
        
        # Priority 2: High impact, high effort (major projects)
        for action in matrix['high_impact_high_effort']:
            action_plan.append({
                'action': action['action_name'],
                'priority': 2,
                'category': 'Major Project',
                'timeline': '30-90 days',
                'resources_needed': 'High'
            })
        
        # Priority 3: Low impact, low effort (fill-in tasks)
        for action in matrix['low_impact_low_effort']:
            action_plan.append({
                'action': action['action_name'],
                'priority': 3,
                'category': 'Fill-in Task',
                'timeline': '60-120 days',
                'resources_needed': 'Low'
            })
        
        return action_plan
    
    def assess_current_risks(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess current risks in the business"""
        risks = {
            'financial_risks': [],
            'operational_risks': [],
            'market_risks': []
        }
        
        # Simple risk assessment based on data patterns
        if 'amount' in data.columns:
            # Financial risk indicators
            volatility = data['amount'].std()
            if volatility > data['amount'].mean():
                risks['financial_risks'].append({
                    'risk': 'High cash flow volatility',
                    'probability': 'Medium',
                    'impact': 'High'
                })
        
        if 'customer_name' in data.columns:
            # Customer concentration risk
            customer_concentration = data.groupby('customer_name')['amount'].sum()
            top_customer_share = customer_concentration.max() / customer_concentration.sum()
            if top_customer_share > 0.3:
                risks['operational_risks'].append({
                    'risk': 'Customer concentration risk',
                    'probability': 'High',
                    'impact': 'High'
                })
        
        return risks
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on prescriptive analysis"""
        recommendations = []
        
        if 'optimization_plan' in analysis_results:
            recommendations.append("Implement the phased optimization plan starting with immediate actions")
        
        if 'prioritized_actions' in analysis_results:
            recommendations.append("Focus on high-priority, low-effort actions first for quick wins")
        
        if 'mitigation_strategies' in analysis_results:
            recommendations.append("Implement risk mitigation strategies to reduce exposure")
        
        return recommendations