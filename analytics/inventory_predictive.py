"""
Inventory Predictive Analytics - "What will happen?"
Forecasting and prediction models for inventory management
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


class InventoryPredictiveAnalytics(AnalyticsBase):
    """Predictive analytics for inventory forecasting and trend analysis"""
    
    def __init__(self, agent_type: str = "inventory"):
        super().__init__("InventoryPredictiveAnalytics", agent_type)
        self.supported_queries = [
            'demand_forecasting',
            'stock_level_prediction',
            'reorder_point_prediction',
            'seasonal_demand_forecast',
            'stockout_risk_prediction',
            'inventory_turnover_forecast',
            'price_trend_prediction',
            'supplier_performance_forecast'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform predictive analytics on inventory data"""
        start_time = datetime.now()
        
        # Determine specific analysis type
        analysis_type = self.classify_analysis_type(query)
        
        # Perform analysis based on type
        if analysis_type == 'demand_forecasting':
            results = self.forecast_demand(data, params)
        elif analysis_type == 'stock_level_prediction':
            results = self.predict_stock_levels(data, params)
        elif analysis_type == 'reorder_point_prediction':
            results = self.predict_reorder_points(data, params)
        elif analysis_type == 'seasonal_demand_forecast':
            results = self.forecast_seasonal_demand(data, params)
        elif analysis_type == 'stockout_risk_prediction':
            results = self.predict_stockout_risk(data, params)
        elif analysis_type == 'inventory_turnover_forecast':
            results = self.forecast_inventory_turnover(data, params)
        elif analysis_type == 'price_trend_prediction':
            results = self.predict_price_trends(data, params)
        elif analysis_type == 'supplier_performance_forecast':
            results = self.forecast_supplier_performance(data, params)
        else:
            results = self.general_forecast(data, params)
        
        return self.prepare_response(
            analytics_type=analysis_type,
            query=query,
            results=results,
            start_time=start_time
        )
    
    def forecast_demand(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast demand for inventory items"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            # Prepare data for forecasting
            data['date'] = pd.to_datetime(data['date'])
            forecast_horizon = params.get('forecast_horizon', 30)  # days
            
            # Aggregate daily demand
            daily_demand = data.groupby(['date', 'item']).agg({
                'quantity': 'sum',
                'amount': 'sum'
            }).reset_index()
            
            # Forecast for top items by activity
            item_activity = data.groupby('item').size().sort_values(ascending=False)
            top_items = item_activity.head(10).index.tolist()
            
            forecasts = {}
            for item in top_items:
                item_data = daily_demand[daily_demand['item'] == item].copy()
                
                if len(item_data) >= 7:  # Need at least a week of data
                    # Prepare features
                    item_data = item_data.sort_values('date')
                    item_data['day_of_week'] = item_data['date'].dt.dayofweek
                    item_data['month'] = item_data['date'].dt.month
                    item_data['day_of_month'] = item_data['date'].dt.day
                    
                    # Create lag features
                    item_data['quantity_lag1'] = item_data['quantity'].shift(1)
                    item_data['quantity_lag7'] = item_data['quantity'].shift(7)
                    item_data['quantity_ma3'] = item_data['quantity'].rolling(3).mean()
                    item_data['quantity_ma7'] = item_data['quantity'].rolling(7).mean()
                    
                    # Remove NaN values
                    item_data = item_data.dropna()
                    
                    if len(item_data) >= 5:
                        # Prepare features and target
                        feature_cols = ['day_of_week', 'month', 'day_of_month', 
                                       'quantity_lag1', 'quantity_lag7', 'quantity_ma3', 'quantity_ma7']
                        X = item_data[feature_cols]
                        y = item_data['quantity']
                        
                        # Split data (80% train, 20% test)
                        split_idx = int(len(item_data) * 0.8)
                        X_train, X_test = X[:split_idx], X[split_idx:]
                        y_train, y_test = y[:split_idx], y[split_idx:]
                        
                        # Train model
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                        model.fit(X_train, y_train)
                        
                        # Make predictions on test set
                        y_pred = model.predict(X_test)
                        
                        # Calculate accuracy metrics
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1))) * 100
                        
                        # Generate future predictions
                        last_date = item_data['date'].max()
                        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                                   periods=forecast_horizon, freq='D')
                        
                        future_predictions = []
                        current_data = item_data.tail(7).copy()  # Use last 7 days for prediction
                        
                        for future_date in future_dates:
                            # Prepare features for future prediction
                            future_features = {
                                'day_of_week': future_date.dayofweek,
                                'month': future_date.month,
                                'day_of_month': future_date.day,
                                'quantity_lag1': current_data['quantity'].iloc[-1],
                                'quantity_lag7': current_data['quantity'].iloc[-7] if len(current_data) >= 7 else current_data['quantity'].mean(),
                                'quantity_ma3': current_data['quantity'].tail(3).mean(),
                                'quantity_ma7': current_data['quantity'].tail(7).mean()
                            }
                            
                            # Make prediction
                            pred = model.predict([list(future_features.values())])[0]
                            future_predictions.append(max(0, pred))  # Ensure non-negative
                            
                            # Update current_data with prediction for next iteration
                            new_row = pd.DataFrame({
                                'date': [future_date],
                                'quantity': [pred],
                                'day_of_week': [future_date.dayofweek],
                                'month': [future_date.month],
                                'day_of_month': [future_date.day],
                                'quantity_lag1': [0],
                                'quantity_lag7': [0],
                                'quantity_ma3': [0],
                                'quantity_ma7': [0]
                            })
                            current_data = pd.concat([current_data, new_row], ignore_index=True)
                        
                        forecasts[item] = {
                            'forecast_horizon': forecast_horizon,
                            'predictions': [round(p, 2) for p in future_predictions],
                            'forecast_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                            'accuracy_metrics': {
                                'mae': round(mae, 2),
                                'rmse': round(rmse, 2),
                                'mape': round(mape, 2)
                            },
                            'historical_mean': round(item_data['quantity'].mean(), 2),
                            'predicted_mean': round(np.mean(future_predictions), 2),
                            'trend': 'increasing' if np.mean(future_predictions) > item_data['quantity'].mean() else 'decreasing'
                        }
            
            return {
                'forecasts': forecasts,
                'forecast_horizon_days': forecast_horizon,
                'items_forecasted': len(forecasts),
                'overall_trend': self._calculate_overall_trend(forecasts)
            }
            
        except Exception as e:
            return {"error": f"Demand forecasting failed: {str(e)}"}
    
    def predict_stock_levels(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future stock levels"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            # Calculate current stock levels
            current_stock = data.groupby('item')['quantity'].sum()
            
            # Calculate stock movement patterns
            data['date'] = pd.to_datetime(data['date'])
            stock_movements = data.groupby(['item', data['date'].dt.date]).agg({
                'quantity': 'sum'
            }).reset_index()
            
            predictions = {}
            forecast_days = params.get('forecast_days', 30)
            
            for item in current_stock.index:
                item_movements = stock_movements[stock_movements['item'] == item]
                
                if len(item_movements) >= 7:
                    # Calculate trend
                    recent_movements = item_movements.tail(14)['quantity'].mean()
                    
                    # Simple linear projection
                    daily_change = recent_movements
                    current_level = current_stock[item]
                    
                    future_levels = []
                    for day in range(1, forecast_days + 1):
                        predicted_level = current_level + (daily_change * day)
                        future_levels.append(max(0, predicted_level))
                    
                    predictions[item] = {
                        'current_stock': round(current_level, 2),
                        'predicted_levels': [round(l, 2) for l in future_levels],
                        'daily_change_rate': round(daily_change, 2),
                        'stockout_risk': 'high' if min(future_levels) < current_level * 0.2 else 'low'
                    }
            
            return {
                'stock_predictions': predictions,
                'forecast_horizon': forecast_days,
                'items_analyzed': len(predictions)
            }
            
        except Exception as e:
            return {"error": f"Stock level prediction failed: {str(e)}"}
    
    def predict_reorder_points(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal reorder points"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            # Calculate demand statistics
            demand_stats = data.groupby('item').agg({
                'quantity': ['mean', 'std', 'count']
            })
            
            # Calculate lead time (simplified as average days between transactions)
            data['date'] = pd.to_datetime(data['date'])
            lead_times = {}
            
            for item in demand_stats.index:
                item_data = data[data['item'] == item].sort_values('date')
                if len(item_data) > 1:
                    time_diffs = item_data['date'].diff().dt.days
                    lead_times[item] = time_diffs.mean()
                else:
                    lead_times[item] = 7  # Default to 7 days
            
            # Calculate reorder points
            reorder_points = {}
            service_level = params.get('service_level', 0.95)  # 95% service level
            z_score = 1.65  # For 95% service level
            
            for item in demand_stats.index:
                avg_demand = demand_stats.loc[item, ('quantity', 'mean')]
                demand_std = demand_stats.loc[item, ('quantity', 'std')]
                lead_time = lead_times.get(item, 7)
                
                # Calculate reorder point = Lead time demand + Safety stock
                lead_time_demand = abs(avg_demand) * lead_time
                safety_stock = z_score * demand_std * np.sqrt(lead_time)
                reorder_point = lead_time_demand + safety_stock
                
                reorder_points[item] = {
                    'reorder_point': round(max(0, reorder_point), 2),
                    'lead_time_demand': round(lead_time_demand, 2),
                    'safety_stock': round(safety_stock, 2),
                    'average_demand': round(abs(avg_demand), 2),
                    'demand_std': round(demand_std, 2),
                    'lead_time_days': round(lead_time, 1),
                    'service_level': service_level
                }
            
            return {
                'reorder_points': reorder_points,
                'service_level': service_level,
                'items_analyzed': len(reorder_points)
            }
            
        except Exception as e:
            return {"error": f"Reorder point prediction failed: {str(e)}"}
    
    def forecast_seasonal_demand(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast seasonal demand patterns"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            data['date'] = pd.to_datetime(data['date'])
            data['month'] = data['date'].dt.month
            data['quarter'] = data['date'].dt.quarter
            
            # Calculate monthly patterns
            monthly_demand = data.groupby(['item', 'month']).agg({
                'quantity': 'sum'
            }).reset_index()
            
            seasonal_forecasts = {}
            
            for item in monthly_demand['item'].unique():
                item_monthly = monthly_demand[monthly_demand['item'] == item]
                
                if len(item_monthly) >= 6:  # Need at least 6 months of data
                    # Calculate seasonal indices
                    overall_avg = item_monthly['quantity'].mean()
                    seasonal_indices = {}
                    
                    for month in range(1, 13):
                        month_data = item_monthly[item_monthly['month'] == month]
                        if len(month_data) > 0:
                            month_avg = month_data['quantity'].mean()
                            seasonal_indices[month] = month_avg / overall_avg if overall_avg > 0 else 1
                        else:
                            seasonal_indices[month] = 1
                    
                    # Forecast next 12 months
                    base_forecast = overall_avg
                    monthly_forecasts = {}
                    
                    for month in range(1, 13):
                        forecast = base_forecast * seasonal_indices[month]
                        monthly_forecasts[month] = round(max(0, forecast), 2)
                    
                    seasonal_forecasts[item] = {
                        'base_forecast': round(base_forecast, 2),
                        'seasonal_indices': {k: round(v, 3) for k, v in seasonal_indices.items()},
                        'monthly_forecasts': monthly_forecasts,
                        'peak_month': max(seasonal_indices, key=seasonal_indices.get),
                        'low_month': min(seasonal_indices, key=seasonal_indices.get)
                    }
            
            return {
                'seasonal_forecasts': seasonal_forecasts,
                'items_forecasted': len(seasonal_forecasts),
                'forecast_horizon': '12 months'
            }
            
        except Exception as e:
            return {"error": f"Seasonal demand forecasting failed: {str(e)}"}
    
    def predict_stockout_risk(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict stockout risk for items"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            # Calculate current stock and consumption rates
            current_stock = data.groupby('item')['quantity'].sum()
            
            # Calculate consumption patterns
            data['date'] = pd.to_datetime(data['date'])
            consumption_data = data[data['quantity'] < 0]  # Outbound movements
            
            consumption_rates = consumption_data.groupby('item').agg({
                'quantity': ['mean', 'std', 'count']
            })
            
            stockout_risks = {}
            risk_threshold_days = params.get('risk_threshold_days', 30)
            
            for item in current_stock.index:
                current_level = current_stock[item]
                
                if item in consumption_rates.index:
                    avg_consumption = abs(consumption_rates.loc[item, ('quantity', 'mean')])
                    consumption_std = consumption_rates.loc[item, ('quantity', 'std')]
                    
                    # Calculate days until stockout at current consumption rate
                    if avg_consumption > 0:
                        days_to_stockout = current_level / avg_consumption
                    else:
                        days_to_stockout = 999  # Very high number if no consumption
                    
                    # Calculate risk probability
                    if days_to_stockout <= risk_threshold_days:
                        if days_to_stockout <= 7:
                            risk_level = 'critical'
                            risk_probability = 0.9
                        elif days_to_stockout <= 14:
                            risk_level = 'high'
                            risk_probability = 0.7
                        else:
                            risk_level = 'medium'
                            risk_probability = 0.4
                    else:
                        risk_level = 'low'
                        risk_probability = 0.1
                    
                    stockout_risks[item] = {
                        'current_stock': round(current_level, 2),
                        'avg_consumption': round(avg_consumption, 2),
                        'days_to_stockout': round(days_to_stockout, 1),
                        'risk_level': risk_level,
                        'risk_probability': risk_probability,
                        'recommended_action': self._get_stockout_action(risk_level)
                    }
            
            # Summary statistics
            risk_summary = {
                'critical_items': len([k for k, v in stockout_risks.items() if v['risk_level'] == 'critical']),
                'high_risk_items': len([k for k, v in stockout_risks.items() if v['risk_level'] == 'high']),
                'medium_risk_items': len([k for k, v in stockout_risks.items() if v['risk_level'] == 'medium']),
                'low_risk_items': len([k for k, v in stockout_risks.items() if v['risk_level'] == 'low'])
            }
            
            return {
                'stockout_risks': stockout_risks,
                'risk_summary': risk_summary,
                'risk_threshold_days': risk_threshold_days,
                'items_analyzed': len(stockout_risks)
            }
            
        except Exception as e:
            return {"error": f"Stockout risk prediction failed: {str(e)}"}
    
    def forecast_inventory_turnover(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast inventory turnover rates"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            # Calculate historical turnover
            data['date'] = pd.to_datetime(data['date'])
            
            # Calculate turnover by item
            turnover_data = data.groupby('item').agg({
                'quantity': ['sum', 'count'],
                'amount': 'sum',
                'date': ['min', 'max']
            })
            
            # Calculate days active
            turnover_data['days_active'] = (
                pd.to_datetime(turnover_data[('date', 'max')]) - 
                pd.to_datetime(turnover_data[('date', 'min')])
            ).dt.days
            
            # Calculate turnover rate
            turnover_data['turnover_rate'] = turnover_data[('quantity', 'count')] / turnover_data['days_active'].replace(0, 1)
            
            # Forecast future turnover
            turnover_forecasts = {}
            forecast_period = params.get('forecast_period', 90)  # days
            
            for item in turnover_data.index:
                current_rate = turnover_data.loc[item, 'turnover_rate']
                
                # Simple trend analysis
                item_data = data[data['item'] == item].sort_values('date')
                if len(item_data) >= 4:
                    # Calculate trend over time
                    item_data['period'] = (item_data['date'] - item_data['date'].min()).dt.days // 30
                    period_counts = item_data.groupby('period').size()
                    
                    if len(period_counts) >= 2:
                        trend = period_counts.tail(2).mean() - period_counts.head(2).mean()
                        trend_rate = trend / 30 if trend != 0 else 0  # daily change
                    else:
                        trend_rate = 0
                    
                    # Forecast turnover
                    forecasted_rate = current_rate + (trend_rate * forecast_period)
                    forecasted_transactions = forecasted_rate * forecast_period
                    
                    turnover_forecasts[item] = {
                        'current_turnover_rate': round(current_rate, 4),
                        'forecasted_turnover_rate': round(max(0, forecasted_rate), 4),
                        'trend_rate': round(trend_rate, 6),
                        'forecasted_transactions': round(forecasted_transactions, 1),
                        'classification': self._classify_turnover_rate(forecasted_rate)
                    }
            
            return {
                'turnover_forecasts': turnover_forecasts,
                'forecast_period_days': forecast_period,
                'items_analyzed': len(turnover_forecasts)
            }
            
        except Exception as e:
            return {"error": f"Inventory turnover forecasting failed: {str(e)}"}
    
    def predict_price_trends(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict price trends for items"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            
            # Calculate price trends
            data['date'] = pd.to_datetime(data['date'])
            
            price_trends = {}
            forecast_horizon = params.get('forecast_horizon', 30)
            
            for item in data['item'].unique():
                item_data = data[data['item'] == item].sort_values('date')
                
                if len(item_data) >= 5:
                    # Calculate moving averages
                    item_data['price_ma3'] = item_data['rate'].rolling(3).mean()
                    item_data['price_ma7'] = item_data['rate'].rolling(7).mean()
                    
                    # Calculate price trend
                    recent_prices = item_data['rate'].tail(5)
                    price_change = recent_prices.iloc[-1] - recent_prices.iloc[0]
                    price_trend = price_change / len(recent_prices)
                    
                    # Forecast future prices
                    current_price = item_data['rate'].iloc[-1]
                    future_prices = []
                    
                    for day in range(1, forecast_horizon + 1):
                        predicted_price = current_price + (price_trend * day)
                        future_prices.append(max(0, predicted_price))
                    
                    # Calculate volatility
                    price_volatility = item_data['rate'].std() / item_data['rate'].mean() if item_data['rate'].mean() > 0 else 0
                    
                    price_trends[item] = {
                        'current_price': round(current_price, 2),
                        'predicted_prices': [round(p, 2) for p in future_prices],
                        'price_trend': round(price_trend, 4),
                        'trend_direction': 'increasing' if price_trend > 0 else 'decreasing' if price_trend < 0 else 'stable',
                        'volatility': round(price_volatility, 4),
                        'price_range': {
                            'min': round(min(future_prices), 2),
                            'max': round(max(future_prices), 2)
                        }
                    }
            
            return {
                'price_trends': price_trends,
                'forecast_horizon': forecast_horizon,
                'items_analyzed': len(price_trends)
            }
            
        except Exception as e:
            return {"error": f"Price trend prediction failed: {str(e)}"}
    
    def forecast_supplier_performance(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast supplier performance metrics"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            if 'party_name' not in data.columns:
                return {"error": "Supplier information not available"}
            
            # Calculate supplier metrics
            supplier_metrics = data.groupby('party_name').agg({
                'quantity': ['sum', 'count', 'std'],
                'amount': 'sum',
                'rate': ['mean', 'std']
            })
            
            supplier_forecasts = {}
            
            for supplier in supplier_metrics.index:
                supplier_data = data[data['party_name'] == supplier]
                
                # Calculate performance trends
                avg_order_value = supplier_data['amount'].mean()
                order_frequency = len(supplier_data) / ((supplier_data['date'].max() - supplier_data['date'].min()).days + 1)
                price_stability = 1 - (supplier_data['rate'].std() / supplier_data['rate'].mean()) if supplier_data['rate'].mean() > 0 else 0
                
                # Forecast future performance
                forecasted_orders = order_frequency * 30  # Next 30 days
                forecasted_value = forecasted_orders * avg_order_value
                
                supplier_forecasts[supplier] = {
                    'current_performance': {
                        'avg_order_value': round(avg_order_value, 2),
                        'order_frequency': round(order_frequency, 4),
                        'price_stability': round(price_stability, 4)
                    },
                    'forecasted_metrics': {
                        'orders_next_30_days': round(forecasted_orders, 1),
                        'value_next_30_days': round(forecasted_value, 2)
                    },
                    'performance_rating': self._rate_supplier_performance(price_stability, order_frequency)
                }
            
            return {
                'supplier_forecasts': supplier_forecasts,
                'suppliers_analyzed': len(supplier_forecasts),
                'forecast_period': '30 days'
            }
            
        except Exception as e:
            return {"error": f"Supplier performance forecasting failed: {str(e)}"}
    
    def general_forecast(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General inventory forecasting"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['quantity', 'amount', 'rate'])
            # Overall inventory trends
            data['date'] = pd.to_datetime(data['date'])
            
            # Daily totals
            daily_totals = data.groupby(data['date'].dt.date).agg({
                'quantity': 'sum',
                'amount': 'sum'
            })
            
            # Calculate trend
            if len(daily_totals) > 1:
                x = np.arange(len(daily_totals))
                quantity_trend = np.polyfit(x, daily_totals['quantity'], 1)[0]
                value_trend = np.polyfit(x, daily_totals['amount'], 1)[0]
                
                return {
                    'overall_trends': {
                        'quantity_trend': round(quantity_trend, 4),
                        'value_trend': round(value_trend, 2),
                        'trend_direction': 'increasing' if quantity_trend > 0 else 'decreasing'
                    },
                    'data_period': {
                        'start': daily_totals.index.min(),
                        'end': daily_totals.index.max(),
                        'days': len(daily_totals)
                    }
                }
            else:
                return {"error": "Insufficient data for trend analysis"}
                
        except Exception as e:
            return {"error": f"General forecasting failed: {str(e)}"}
    
    def _calculate_overall_trend(self, forecasts: Dict[str, Dict]) -> str:
        """Calculate overall demand trend"""
        if not forecasts:
            return "unknown"
        
        increasing = sum(1 for f in forecasts.values() if f['trend'] == 'increasing')
        decreasing = sum(1 for f in forecasts.values() if f['trend'] == 'decreasing')
        
        if increasing > decreasing:
            return "increasing"
        elif decreasing > increasing:
            return "decreasing"
        else:
            return "stable"
    
    def _get_stockout_action(self, risk_level: str) -> str:
        """Get recommended action for stockout risk"""
        actions = {
            'critical': 'Immediate reorder required',
            'high': 'Place order within 3 days',
            'medium': 'Plan reorder within 1 week',
            'low': 'Monitor stock levels'
        }
        return actions.get(risk_level, 'Monitor stock levels')
    
    def _classify_turnover_rate(self, rate: float) -> str:
        """Classify turnover rate"""
        if rate > 0.1:
            return 'fast_moving'
        elif rate > 0.05:
            return 'medium_moving'
        else:
            return 'slow_moving'
    
    def _rate_supplier_performance(self, price_stability: float, order_frequency: float) -> str:
        """Rate supplier performance"""
        score = (price_stability * 0.6) + (min(order_frequency * 100, 1) * 0.4)
        
        if score > 0.8:
            return 'excellent'
        elif score > 0.6:
            return 'good'
        elif score > 0.4:
            return 'average'
        else:
            return 'poor'
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the type of predictive analysis based on query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['demand', 'forecast', 'predict demand']):
            return 'demand_forecasting'
        elif any(word in query_lower for word in ['stock level', 'inventory level', 'predict stock']):
            return 'stock_level_prediction'
        elif any(word in query_lower for word in ['reorder', 'reorder point', 'order point']):
            return 'reorder_point_prediction'
        elif any(word in query_lower for word in ['seasonal', 'season', 'monthly']):
            return 'seasonal_demand_forecast'
        elif any(word in query_lower for word in ['stockout', 'stock out', 'shortage']):
            return 'stockout_risk_prediction'
        elif any(word in query_lower for word in ['turnover', 'rotation', 'movement']):
            return 'inventory_turnover_forecast'
        elif any(word in query_lower for word in ['price', 'cost', 'rate']):
            return 'price_trend_prediction'
        elif any(word in query_lower for word in ['supplier', 'vendor', 'party']):
            return 'supplier_performance_forecast'
        else:
            return 'general_forecast'