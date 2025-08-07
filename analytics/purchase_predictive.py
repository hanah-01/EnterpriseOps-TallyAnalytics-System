"""
Purchase Predictive Analytics - "What will happen?"
Forecasting and prediction models for procurement performance
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .base import AnalyticsBase, AnalyticsResponse


class PurchasePredictiveAnalytics(AnalyticsBase):
    """Predictive analytics for purchase forecasting and trend analysis"""
    
    def __init__(self, agent_type: str = "purchase"):
        super().__init__("PurchasePredictiveAnalytics", agent_type)
        self.supported_queries = [
            'cost_forecasting',
            'supplier_performance_prediction',
            'procurement_trend_prediction',
            'supplier_risk_prediction',
            'seasonal_purchase_forecast',
            'item_demand_forecast',
            'budget_requirement_prediction',
            'supplier_behavior_prediction'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform predictive analytics on purchase data"""
        start_time = datetime.now()
        
        # Determine specific analysis type
        analysis_type = self.classify_analysis_type(query)
        
        # Perform analysis based on type
        if analysis_type == 'cost_forecasting':
            results = self.forecast_costs(data, params)
        elif analysis_type == 'supplier_performance_prediction':
            results = self.predict_supplier_performance(data, params)
        elif analysis_type == 'procurement_trend_prediction':
            results = self.predict_procurement_trends(data, params)
        elif analysis_type == 'supplier_risk_prediction':
            results = self.predict_supplier_risks(data, params)
        elif analysis_type == 'seasonal_purchase_forecast':
            results = self.forecast_seasonal_purchases(data, params)
        elif analysis_type == 'item_demand_forecast':
            results = self.forecast_item_demand(data, params)
        elif analysis_type == 'budget_requirement_prediction':
            results = self.predict_budget_requirements(data, params)
        elif analysis_type == 'supplier_behavior_prediction':
            results = self.predict_supplier_behavior(data, params)
        else:
            results = self.general_purchase_forecast(data, params)
        
        return self.prepare_response(
            analytics_type=analysis_type,
            query=query,
            results=results,
            start_time=start_time
        )
    
    def forecast_costs(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast procurement costs using ML models"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Prepare data for forecasting
            cost_data = data[data['amount'] > 0].copy()
            cost_data['date'] = pd.to_datetime(cost_data['date'])
            forecast_horizon = params.get('forecast_horizon', 30)  # days
            
            # Aggregate daily costs
            daily_costs = cost_data.groupby(cost_data['date'].dt.date)['amount'].sum().reset_index()
            daily_costs.columns = ['date', 'cost']
            daily_costs['date'] = pd.to_datetime(daily_costs['date'])
            daily_costs = daily_costs.sort_values('date')
            
            # Create features for ML model
            daily_costs['day_of_week'] = daily_costs['date'].dt.dayofweek
            daily_costs['month'] = daily_costs['date'].dt.month
            daily_costs['quarter'] = daily_costs['date'].dt.quarter
            daily_costs['day_of_month'] = daily_costs['date'].dt.day
            
            # Create lagged features
            daily_costs['cost_lag_1'] = daily_costs['cost'].shift(1)
            daily_costs['cost_lag_7'] = daily_costs['cost'].shift(7)
            daily_costs['cost_ma_7'] = daily_costs['cost'].rolling(window=7).mean()
            
            # Remove rows with NaN values
            daily_costs = daily_costs.dropna()
            
            if len(daily_costs) < 10:
                return {"error": "Insufficient data for cost forecasting"}
            
            # Prepare features and target
            feature_columns = ['day_of_week', 'month', 'quarter', 'day_of_month', 
                             'cost_lag_1', 'cost_lag_7', 'cost_ma_7']
            X = daily_costs[feature_columns]
            y = daily_costs['cost']
            
            # Split data for training and testing
            train_size = int(len(daily_costs) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Make predictions for test set
            y_pred = rf_model.predict(X_test)
            
            # Calculate accuracy metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Generate future forecasts
            last_date = daily_costs['date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='D')
            
            # Create future features
            future_features = []
            for date in future_dates:
                # Use last known values for lagged features
                last_cost = daily_costs['cost'].iloc[-1]
                last_7_avg = daily_costs['cost'].tail(7).mean()
                
                future_features.append({
                    'day_of_week': date.dayofweek,
                    'month': date.month,
                    'quarter': date.quarter,
                    'day_of_month': date.day,
                    'cost_lag_1': last_cost,
                    'cost_lag_7': last_cost,
                    'cost_ma_7': last_7_avg
                })
            
            future_df = pd.DataFrame(future_features)
            future_predictions = rf_model.predict(future_df)
            
            # Create forecast results
            forecasts = []
            for i, date in enumerate(future_dates):
                forecasts.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_cost': round(future_predictions[i], 2)
                })
            
            return {
                "forecasts": forecasts,
                "forecast_horizon": forecast_horizon,
                "model_accuracy": {
                    "mae": round(mae, 2),
                    "rmse": round(rmse, 2),
                    "mape": round(mape, 2)
                },
                "total_predicted_cost": round(sum(future_predictions), 2),
                "avg_daily_predicted_cost": round(np.mean(future_predictions), 2),
                "trend_direction": self._calculate_trend_direction(future_predictions)
            }
            
        except Exception as e:
            return {"error": f"Cost forecasting failed: {str(e)}"}
    
    def predict_supplier_performance(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict supplier performance and reliability"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Supplier performance analysis
            supplier_data = data[data['amount'] > 0].copy()
            supplier_data['date'] = pd.to_datetime(supplier_data['date'])
            
            # Calculate supplier metrics
            supplier_metrics = supplier_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean', 'std'],
                'date': ['min', 'max'],
                'voucher_number': 'nunique'
            })
            
            # Flatten column names
            supplier_metrics.columns = ['_'.join(col).strip() for col in supplier_metrics.columns]
            
            # Calculate supplier performance indicators
            current_date = supplier_data['date'].max()
            supplier_metrics['days_since_last'] = (current_date - pd.to_datetime(supplier_metrics['date_max'])).dt.days
            supplier_metrics['relationship_duration'] = (
                pd.to_datetime(supplier_metrics['date_max']) - 
                pd.to_datetime(supplier_metrics['date_min'])
            ).dt.days
            
            supplier_metrics['transaction_frequency'] = supplier_metrics['voucher_number_nunique'] / (supplier_metrics['relationship_duration'] + 1)
            supplier_metrics['spending_consistency'] = supplier_metrics['amount_std'] / supplier_metrics['amount_mean']
            
            # Simple performance prediction based on historical patterns
            def calculate_performance_score(row):
                score = 0
                # Recency score
                if row['days_since_last'] <= 30:
                    score += 40
                elif row['days_since_last'] <= 60:
                    score += 25
                elif row['days_since_last'] <= 90:
                    score += 10
                
                # Frequency score
                if row['transaction_frequency'] > 0.1:
                    score += 30
                elif row['transaction_frequency'] > 0.05:
                    score += 20
                elif row['transaction_frequency'] > 0.01:
                    score += 10
                
                # Consistency score
                if row['spending_consistency'] < 0.3:
                    score += 30
                elif row['spending_consistency'] < 0.6:
                    score += 20
                elif row['spending_consistency'] < 1.0:
                    score += 10
                
                return min(score, 100)
            
            supplier_metrics['performance_score'] = supplier_metrics.apply(calculate_performance_score, axis=1)
            
            # Segment suppliers by predicted performance
            def performance_category(score):
                if score >= 80:
                    return 'High Performance'
                elif score >= 60:
                    return 'Good Performance'
                elif score >= 40:
                    return 'Average Performance'
                else:
                    return 'Poor Performance'
            
            supplier_metrics['predicted_category'] = supplier_metrics['performance_score'].apply(performance_category)
            
            # Top performers
            top_performers = supplier_metrics.nlargest(10, 'performance_score')
            
            return {
                "supplier_performance_predictions": len(supplier_metrics),
                "performance_categories": supplier_metrics['predicted_category'].value_counts().to_dict(),
                "performance_statistics": {
                    "avg_performance_score": round(supplier_metrics['performance_score'].mean(), 2),
                    "top_performers": len(supplier_metrics[supplier_metrics['performance_score'] >= 80]),
                    "at_risk_suppliers": len(supplier_metrics[supplier_metrics['performance_score'] < 40])
                },
                "top_performers": top_performers[['performance_score', 'predicted_category', 'amount_sum']].to_dict('index'),
                "recommendations": self._generate_performance_predictions_recommendations(supplier_metrics)
            }
            
        except Exception as e:
            return {"error": f"Supplier performance prediction failed: {str(e)}"}
    
    def predict_procurement_trends(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict procurement trends and patterns"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Procurement trend analysis
            procurement_data = data[data['amount'] > 0].copy()
            procurement_data['date'] = pd.to_datetime(procurement_data['date'])
            
            # Monthly procurement trends
            monthly_procurement = procurement_data.groupby(procurement_data['date'].dt.to_period('M'))['amount'].sum()
            
            # Calculate trend using linear regression
            from sklearn.linear_model import LinearRegression
            
            # Prepare data for trend analysis
            months = np.arange(len(monthly_procurement)).reshape(-1, 1)
            procurement_values = monthly_procurement.values
            
            # Fit linear regression
            trend_model = LinearRegression()
            trend_model.fit(months, procurement_values)
            
            # Predict next 3 months
            future_months = np.arange(len(monthly_procurement), len(monthly_procurement) + 3).reshape(-1, 1)
            future_predictions = trend_model.predict(future_months)
            
            # Calculate trend metrics
            trend_slope = trend_model.coef_[0]
            trend_direction = "increasing" if trend_slope > 0 else "decreasing"
            
            # Category-wise trends if item data available
            category_trends = {}
            if 'item' in procurement_data.columns:
                top_items = procurement_data.groupby('item')['amount'].sum().nlargest(5)
                for item in top_items.index:
                    item_data = procurement_data[procurement_data['item'] == item]
                    item_monthly = item_data.groupby(item_data['date'].dt.to_period('M'))['amount'].sum()
                    if len(item_monthly) > 2:
                        item_trend = item_monthly.pct_change().mean() * 100
                        category_trends[item] = round(item_trend, 2)
            
            return {
                "trend_analysis": {
                    "trend_direction": trend_direction,
                    "trend_slope": round(trend_slope, 2),
                    "trend_strength": round(abs(trend_slope) / monthly_procurement.mean() * 100, 2)
                },
                "future_predictions": {
                    "next_month": round(future_predictions[0], 2),
                    "next_2_months": round(future_predictions[1], 2),
                    "next_3_months": round(future_predictions[2], 2)
                },
                "historical_trend": monthly_procurement.to_dict(),
                "category_trends": category_trends,
                "trend_confidence": round(trend_model.score(months, procurement_values), 2)
            }
            
        except Exception as e:
            return {"error": f"Procurement trend prediction failed: {str(e)}"}
    
    def predict_supplier_risks(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict supplier risks and potential issues"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Supplier risk prediction
            supplier_data = data[data['amount'] > 0].copy()
            supplier_data['date'] = pd.to_datetime(supplier_data['date'])
            
            # Calculate supplier risk features
            current_date = supplier_data['date'].max()
            supplier_features = supplier_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean', 'std'],
                'date': ['min', 'max'],
                'voucher_number': 'nunique'
            })
            
            # Flatten column names
            supplier_features.columns = ['_'.join(col).strip() for col in supplier_features.columns]
            
            # Calculate risk features
            supplier_features['days_since_last'] = (current_date - pd.to_datetime(supplier_features['date_max'])).dt.days
            supplier_features['relationship_duration'] = (
                pd.to_datetime(supplier_features['date_max']) - 
                pd.to_datetime(supplier_features['date_min'])
            ).dt.days
            
            supplier_features['transaction_frequency'] = supplier_features['voucher_number_nunique'] / (supplier_features['relationship_duration'] + 1)
            supplier_features['spending_volatility'] = supplier_features['amount_std'] / supplier_features['amount_mean']
            
            # Simple risk scoring
            def calculate_risk_score(row):
                risk_score = 0
                
                # Inactivity risk
                if row['days_since_last'] > 90:
                    risk_score += 40
                elif row['days_since_last'] > 60:
                    risk_score += 25
                elif row['days_since_last'] > 30:
                    risk_score += 10
                
                # Volatility risk
                if row['spending_volatility'] > 1.0:
                    risk_score += 30
                elif row['spending_volatility'] > 0.6:
                    risk_score += 20
                elif row['spending_volatility'] > 0.3:
                    risk_score += 10
                
                # Frequency risk
                if row['transaction_frequency'] < 0.01:
                    risk_score += 30
                elif row['transaction_frequency'] < 0.05:
                    risk_score += 15
                
                return min(risk_score, 100)
            
            supplier_features['risk_score'] = supplier_features.apply(calculate_risk_score, axis=1)
            
            # Risk categories
            def risk_category(score):
                if score >= 70:
                    return 'High Risk'
                elif score >= 40:
                    return 'Medium Risk'
                else:
                    return 'Low Risk'
            
            supplier_features['risk_category'] = supplier_features['risk_score'].apply(risk_category)
            
            # High-risk suppliers
            high_risk_suppliers = supplier_features[supplier_features['risk_category'] == 'High Risk']
            
            return {
                "total_suppliers": len(supplier_features),
                "risk_categories": supplier_features['risk_category'].value_counts().to_dict(),
                "high_risk_suppliers": len(high_risk_suppliers),
                "risk_predictions": {
                    "avg_risk_score": round(supplier_features['risk_score'].mean(), 2),
                    "suppliers_at_risk": len(supplier_features[supplier_features['risk_score'] > 50]),
                    "critical_risk_suppliers": len(supplier_features[supplier_features['risk_score'] > 80])
                },
                "high_risk_details": high_risk_suppliers[['risk_score', 'days_since_last', 'amount_sum']].to_dict('index'),
                "spending_at_risk": round(high_risk_suppliers['amount_sum'].sum(), 2)
            }
            
        except Exception as e:
            return {"error": f"Supplier risk prediction failed: {str(e)}"}
    
    def forecast_seasonal_purchases(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast seasonal purchase patterns"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Seasonal forecasting
            seasonal_data = data[data['amount'] > 0].copy()
            seasonal_data['date'] = pd.to_datetime(seasonal_data['date'])
            seasonal_data['month'] = seasonal_data['date'].dt.month
            seasonal_data['quarter'] = seasonal_data['date'].dt.quarter
            
            # Monthly seasonal patterns
            monthly_patterns = seasonal_data.groupby('month')['amount'].mean()
            
            # Quarterly seasonal patterns
            quarterly_patterns = seasonal_data.groupby('quarter')['amount'].mean()
            
            # Calculate seasonal indices
            overall_avg = seasonal_data['amount'].mean()
            monthly_indices = monthly_patterns / overall_avg
            quarterly_indices = quarterly_patterns / overall_avg
            
            # Forecast next 12 months
            forecast_months = []
            for i in range(1, 13):
                month = i
                seasonal_index = monthly_indices[month]
                forecasted_amount = overall_avg * seasonal_index
                
                forecast_months.append({
                    'month': month,
                    'forecasted_purchases': round(forecasted_amount, 2),
                    'seasonal_index': round(seasonal_index, 2)
                })
            
            return {
                "seasonal_forecasts": forecast_months,
                "seasonal_patterns": {
                    "monthly_indices": monthly_indices.to_dict(),
                    "quarterly_indices": quarterly_indices.to_dict()
                },
                "peak_month": monthly_patterns.idxmax(),
                "low_month": monthly_patterns.idxmin(),
                "seasonality_strength": round(monthly_patterns.std() / monthly_patterns.mean() * 100, 2)
            }
            
        except Exception as e:
            return {"error": f"Seasonal purchase forecasting failed: {str(e)}"}
    
    def forecast_item_demand(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast item demand and requirements"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Item demand forecasting
            item_data = data[(data['amount'] > 0) & (data['item'].notna())].copy()
            
            if len(item_data) == 0:
                return {"error": "No item data available"}
            
            # Analyze top items
            top_items = item_data.groupby('item')['amount'].sum().nlargest(10)
            
            # Forecast for each top item
            item_forecasts = {}
            for item in top_items.index:
                item_purchases = item_data[item_data['item'] == item]
                item_purchases['date'] = pd.to_datetime(item_purchases['date'])
                
                # Monthly purchases for this item
                monthly_purchases = item_purchases.groupby(item_purchases['date'].dt.to_period('M'))['amount'].sum()
                
                if len(monthly_purchases) > 3:
                    # Simple trend-based forecast
                    trend = monthly_purchases.pct_change().mean()
                    last_month_purchases = monthly_purchases.iloc[-1]
                    
                    # Forecast next 3 months
                    forecasts = []
                    for i in range(1, 4):
                        predicted_purchases = last_month_purchases * (1 + trend) ** i
                        forecasts.append(round(predicted_purchases, 2))
                    
                    item_forecasts[item] = {
                        'current_purchases': round(last_month_purchases, 2),
                        'forecasts': forecasts,
                        'trend': round(trend * 100, 2)
                    }
            
            return {
                "item_forecasts": item_forecasts,
                "top_items": top_items.to_dict(),
                "total_items_forecasted": len(item_forecasts)
            }
            
        except Exception as e:
            return {"error": f"Item demand forecasting failed: {str(e)}"}
    
    def predict_budget_requirements(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict budget requirements for future periods"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Budget prediction analysis
            budget_data = data[data['amount'] > 0].copy()
            budget_data['date'] = pd.to_datetime(budget_data['date'])
            
            # Historical spending patterns
            monthly_spending = budget_data.groupby(budget_data['date'].dt.to_period('M'))['amount'].sum()
            
            # Calculate growth trend
            growth_rate = monthly_spending.pct_change().mean()
            
            # Predict budget requirements for next 6 months
            last_month_spending = monthly_spending.iloc[-1]
            budget_predictions = []
            
            for i in range(1, 7):
                predicted_budget = last_month_spending * (1 + growth_rate) ** i
                budget_predictions.append({
                    'month': i,
                    'predicted_budget': round(predicted_budget, 2),
                    'confidence_level': max(0.5, 1 - (i * 0.1))  # Decreasing confidence over time
                })
            
            # Category-wise budget predictions
            category_budgets = {}
            if 'item' in budget_data.columns:
                top_categories = budget_data.groupby('item')['amount'].sum().nlargest(5)
                for category in top_categories.index:
                    category_spending = budget_data[budget_data['item'] == category]['amount'].sum()
                    category_percentage = category_spending / budget_data['amount'].sum()
                    category_budgets[category] = {
                        'current_percentage': round(category_percentage * 100, 2),
                        'predicted_next_month': round(predicted_budget * category_percentage, 2)
                    }
            
            return {
                "budget_predictions": budget_predictions,
                "total_predicted_6_months": round(sum([p['predicted_budget'] for p in budget_predictions]), 2),
                "growth_trend": {
                    "monthly_growth_rate": round(growth_rate * 100, 2),
                    "trend_direction": "increasing" if growth_rate > 0 else "decreasing"
                },
                "category_budgets": category_budgets,
                "budget_recommendations": self._generate_budget_recommendations(growth_rate, monthly_spending)
            }
            
        except Exception as e:
            return {"error": f"Budget requirement prediction failed: {str(e)}"}
    
    def predict_supplier_behavior(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict supplier behavior patterns"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Supplier behavior prediction
            supplier_data = data[data['amount'] > 0].copy()
            supplier_data['date'] = pd.to_datetime(supplier_data['date'])
            
            # Calculate supplier behavior features
            supplier_behavior = supplier_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean', 'std'],
                'date': ['min', 'max'],
                'voucher_type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
            })
            
            # Flatten column names
            supplier_behavior.columns = ['_'.join(col).strip() for col in supplier_behavior.columns]
            
            # Predict next purchase probability
            current_date = supplier_data['date'].max()
            supplier_behavior['days_since_last'] = (current_date - pd.to_datetime(supplier_behavior['date_max'])).dt.days
            supplier_behavior['avg_purchase_interval'] = (
                pd.to_datetime(supplier_behavior['date_max']) - 
                pd.to_datetime(supplier_behavior['date_min'])
            ).dt.days / supplier_behavior['amount_count']
            
            # Predict next purchase probability
            def predict_next_purchase(days_since_last, avg_interval):
                if avg_interval == 0:
                    return 0.5
                
                expected_next_purchase = days_since_last / avg_interval
                if expected_next_purchase <= 0.5:
                    return 0.9  # Very likely
                elif expected_next_purchase <= 1.0:
                    return 0.7  # Likely
                elif expected_next_purchase <= 2.0:
                    return 0.3  # Unlikely
                else:
                    return 0.1  # Very unlikely
            
            supplier_behavior['next_purchase_probability'] = supplier_behavior.apply(
                lambda row: predict_next_purchase(row['days_since_last'], row['avg_purchase_interval']), 
                axis=1
            )
            
            # Behavioral segments
            def behavioral_segment(prob):
                if prob >= 0.7:
                    return 'Active'
                elif prob >= 0.3:
                    return 'At Risk'
                else:
                    return 'Inactive'
            
            supplier_behavior['behavioral_segment'] = supplier_behavior['next_purchase_probability'].apply(behavioral_segment)
            
            return {
                "behavior_predictions": len(supplier_behavior),
                "behavioral_segments": supplier_behavior['behavioral_segment'].value_counts().to_dict(),
                "purchase_probabilities": {
                    "avg_probability": round(supplier_behavior['next_purchase_probability'].mean(), 2),
                    "high_probability_suppliers": len(supplier_behavior[supplier_behavior['next_purchase_probability'] > 0.7]),
                    "at_risk_suppliers": len(supplier_behavior[supplier_behavior['next_purchase_probability'] < 0.3])
                },
                "predicted_spending": round(
                    (supplier_behavior['amount_mean'] * supplier_behavior['next_purchase_probability']).sum(), 2
                )
            }
            
        except Exception as e:
            return {"error": f"Supplier behavior prediction failed: {str(e)}"}
    
    def general_purchase_forecast(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General purchase forecasting"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # General forecast
            purchase_data = data[data['amount'] > 0]
            
            # Basic forecasting metrics
            total_purchases = purchase_data['amount'].sum()
            avg_daily_purchases = purchase_data.groupby(pd.to_datetime(purchase_data['date']).dt.date)['amount'].sum().mean()
            
            # Simple 30-day forecast
            forecast_30_days = avg_daily_purchases * 30
            
            return {
                "overall_trends": {
                    "total_historical_purchases": round(total_purchases, 2),
                    "avg_daily_purchases": round(avg_daily_purchases, 2),
                    "forecast_30_days": round(forecast_30_days, 2)
                },
                "forecast_confidence": "Medium",
                "data_quality": {
                    "total_records": len(purchase_data),
                    "date_range": f"{purchase_data['date'].min()} to {purchase_data['date'].max()}"
                }
            }
            
        except Exception as e:
            return {"error": f"General purchase forecasting failed: {str(e)}"}
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the type of predictive analysis based on query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['cost', 'forecast', 'predict', 'expense']):
            return 'cost_forecasting'
        elif any(word in query_lower for word in ['supplier', 'performance', 'vendor']):
            return 'supplier_performance_prediction'
        elif any(word in query_lower for word in ['trend', 'pattern', 'direction']):
            return 'procurement_trend_prediction'
        elif any(word in query_lower for word in ['risk', 'reliability', 'threat']):
            return 'supplier_risk_prediction'
        elif any(word in query_lower for word in ['seasonal', 'quarterly', 'monthly']):
            return 'seasonal_purchase_forecast'
        elif any(word in query_lower for word in ['item', 'demand', 'product']):
            return 'item_demand_forecast'
        elif any(word in query_lower for word in ['budget', 'requirement', 'planning']):
            return 'budget_requirement_prediction'
        elif any(word in query_lower for word in ['behavior', 'next', 'purchase']):
            return 'supplier_behavior_prediction'
        else:
            return 'general_forecast'
    
    def _calculate_trend_direction(self, values: np.ndarray) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "stable"
        
        trend = np.polyfit(range(len(values)), values, 1)[0]
        if trend > 0:
            return "increasing"
        elif trend < 0:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_performance_predictions_recommendations(self, supplier_metrics: pd.DataFrame) -> List[str]:
        """Generate recommendations based on performance predictions"""
        recommendations = []
        
        high_performers = len(supplier_metrics[supplier_metrics['performance_score'] >= 80])
        poor_performers = len(supplier_metrics[supplier_metrics['performance_score'] < 40])
        
        if high_performers > 0:
            recommendations.append(f"Strengthen partnerships with {high_performers} high-performing suppliers")
        
        if poor_performers > 0:
            recommendations.append(f"Develop improvement plans for {poor_performers} underperforming suppliers")
        
        recommendations.append("Implement supplier performance monitoring dashboard")
        recommendations.append("Create supplier development programs for continuous improvement")
        
        return recommendations
    
    def _generate_budget_recommendations(self, growth_rate: float, monthly_spending: pd.Series) -> List[str]:
        """Generate budget recommendations"""
        recommendations = []
        
        if growth_rate > 0.1:  # 10% monthly growth
            recommendations.append("Budget for significant cost increases - implement cost control measures")
        elif growth_rate > 0.05:  # 5% monthly growth
            recommendations.append("Moderate budget increase expected - monitor spending trends")
        elif growth_rate < -0.05:  # Decreasing spending
            recommendations.append("Spending trend declining - review procurement efficiency")
        else:
            recommendations.append("Stable spending pattern - maintain current budget allocation")
        
        volatility = monthly_spending.std() / monthly_spending.mean()
        if volatility > 0.3:
            recommendations.append("High spending volatility detected - implement smoother procurement planning")
        
        return recommendations