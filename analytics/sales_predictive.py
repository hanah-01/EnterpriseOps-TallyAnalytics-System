"""
Sales Predictive Analytics - "What will happen?"
Forecasting and prediction models for sales performance
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


class SalesPredictiveAnalytics(AnalyticsBase):
    """Predictive analytics for sales forecasting and trend analysis"""
    
    def __init__(self, agent_type: str = "sales"):
        super().__init__("SalesPredictiveAnalytics", agent_type)
        self.supported_queries = [
            'revenue_forecasting',
            'customer_lifetime_value_prediction',
            'sales_trend_prediction',
            'customer_churn_prediction',
            'seasonal_sales_forecast',
            'product_demand_forecast',
            'market_opportunity_prediction',
            'customer_behavior_prediction'
        ]
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform predictive analytics on sales data"""
        start_time = datetime.now()
        
        # Determine specific analysis type
        analysis_type = self.classify_analysis_type(query)
        
        # Perform analysis based on type
        if analysis_type == 'revenue_forecasting':
            results = self.forecast_revenue(data, params)
        elif analysis_type == 'customer_lifetime_value_prediction':
            results = self.predict_customer_lifetime_value(data, params)
        elif analysis_type == 'sales_trend_prediction':
            results = self.predict_sales_trends(data, params)
        elif analysis_type == 'customer_churn_prediction':
            results = self.predict_customer_churn(data, params)
        elif analysis_type == 'seasonal_sales_forecast':
            results = self.forecast_seasonal_sales(data, params)
        elif analysis_type == 'product_demand_forecast':
            results = self.forecast_product_demand(data, params)
        elif analysis_type == 'market_opportunity_prediction':
            results = self.predict_market_opportunities(data, params)
        elif analysis_type == 'customer_behavior_prediction':
            results = self.predict_customer_behavior(data, params)
        else:
            results = self.general_sales_forecast(data, params)
        
        return self.prepare_response(
            analytics_type=analysis_type,
            query=query,
            results=results,
            start_time=start_time
        )
    
    def forecast_revenue(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast revenue using ML models"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Prepare data for forecasting
            revenue_data = data[data['amount'] > 0].copy()
            revenue_data['date'] = pd.to_datetime(revenue_data['date'])
            forecast_horizon = params.get('forecast_horizon', 30)  # days
            
            # Aggregate daily revenue
            daily_revenue = revenue_data.groupby(revenue_data['date'].dt.date)['amount'].sum().reset_index()
            daily_revenue.columns = ['date', 'revenue']
            daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
            daily_revenue = daily_revenue.sort_values('date')
            
            # Create features for ML model
            daily_revenue['day_of_week'] = daily_revenue['date'].dt.dayofweek
            daily_revenue['month'] = daily_revenue['date'].dt.month
            daily_revenue['quarter'] = daily_revenue['date'].dt.quarter
            daily_revenue['day_of_month'] = daily_revenue['date'].dt.day
            
            # Create lagged features
            daily_revenue['revenue_lag_1'] = daily_revenue['revenue'].shift(1)
            daily_revenue['revenue_lag_7'] = daily_revenue['revenue'].shift(7)
            daily_revenue['revenue_ma_7'] = daily_revenue['revenue'].rolling(window=7).mean()
            
            # Remove rows with NaN values
            daily_revenue = daily_revenue.dropna()
            
            if len(daily_revenue) < 10:
                return {"error": "Insufficient data for revenue forecasting"}
            
            # Prepare features and target
            feature_columns = ['day_of_week', 'month', 'quarter', 'day_of_month', 
                             'revenue_lag_1', 'revenue_lag_7', 'revenue_ma_7']
            X = daily_revenue[feature_columns]
            y = daily_revenue['revenue']
            
            # Split data for training and testing
            train_size = int(len(daily_revenue) * 0.8)
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
            last_date = daily_revenue['date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='D')
            
            # Create future features
            future_features = []
            for date in future_dates:
                # Use last known values for lagged features
                last_revenue = daily_revenue['revenue'].iloc[-1]
                last_7_avg = daily_revenue['revenue'].tail(7).mean()
                
                future_features.append({
                    'day_of_week': date.dayofweek,
                    'month': date.month,
                    'quarter': date.quarter,
                    'day_of_month': date.day,
                    'revenue_lag_1': last_revenue,
                    'revenue_lag_7': last_revenue,
                    'revenue_ma_7': last_7_avg
                })
            
            future_df = pd.DataFrame(future_features)
            future_predictions = rf_model.predict(future_df)
            
            # Create forecast results
            forecasts = []
            for i, date in enumerate(future_dates):
                forecasts.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_revenue': round(future_predictions[i], 2)
                })
            
            return {
                "forecasts": forecasts,
                "forecast_horizon": forecast_horizon,
                "model_accuracy": {
                    "mae": round(mae, 2),
                    "rmse": round(rmse, 2),
                    "mape": round(mape, 2)
                },
                "total_predicted_revenue": round(sum(future_predictions), 2),
                "avg_daily_predicted_revenue": round(np.mean(future_predictions), 2),
                "trend_direction": self._calculate_trend_direction(future_predictions)
            }
            
        except Exception as e:
            return {"error": f"Revenue forecasting failed: {str(e)}"}
    
    def predict_customer_lifetime_value(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict customer lifetime value"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Customer LTV analysis
            customer_data = data[data['amount'] > 0].copy()
            customer_data['date'] = pd.to_datetime(customer_data['date'])
            
            # Calculate customer metrics
            customer_metrics = customer_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean'],
                'date': ['min', 'max'],
                'voucher_number': 'nunique'
            })
            
            # Flatten column names
            customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns]
            
            # Calculate customer lifetime and frequency
            customer_metrics['lifetime_days'] = (
                pd.to_datetime(customer_metrics['date_max']) - 
                pd.to_datetime(customer_metrics['date_min'])
            ).dt.days
            
            customer_metrics['frequency'] = customer_metrics['voucher_number_nunique'] / (customer_metrics['lifetime_days'] + 1)
            customer_metrics['monetary_value'] = customer_metrics['amount_mean']
            
            # Predict CLV using simplified model
            # CLV = (Average Order Value × Purchase Frequency × Gross Margin × Customer Lifetime)
            avg_margin = params.get('gross_margin', 0.3)  # 30% default margin
            
            customer_metrics['predicted_clv'] = (
                customer_metrics['monetary_value'] * 
                customer_metrics['frequency'] * 
                365 * avg_margin  # Annualized
            )
            
            # Segment customers by CLV
            clv_thresholds = customer_metrics['predicted_clv'].quantile([0.33, 0.66, 1.0])
            
            def clv_segment(clv):
                if clv <= clv_thresholds[0.33]:
                    return 'Low'
                elif clv <= clv_thresholds[0.66]:
                    return 'Medium'
                else:
                    return 'High'
            
            customer_metrics['clv_segment'] = customer_metrics['predicted_clv'].apply(clv_segment)
            
            # Top customers by CLV
            top_customers = customer_metrics.nlargest(10, 'predicted_clv')
            
            return {
                "customer_clv_predictions": len(customer_metrics),
                "clv_segments": customer_metrics['clv_segment'].value_counts().to_dict(),
                "clv_statistics": {
                    "avg_clv": round(customer_metrics['predicted_clv'].mean(), 2),
                    "median_clv": round(customer_metrics['predicted_clv'].median(), 2),
                    "max_clv": round(customer_metrics['predicted_clv'].max(), 2),
                    "total_clv": round(customer_metrics['predicted_clv'].sum(), 2)
                },
                "top_customers": top_customers[['predicted_clv', 'clv_segment', 'amount_sum']].to_dict('index'),
                "clv_thresholds": {
                    "low_threshold": round(clv_thresholds[0.33], 2),
                    "medium_threshold": round(clv_thresholds[0.66], 2)
                }
            }
            
        except Exception as e:
            return {"error": f"Customer LTV prediction failed: {str(e)}"}
    
    def predict_sales_trends(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict sales trends and patterns"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Sales trend analysis
            sales_data = data[data['amount'] > 0].copy()
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            
            # Monthly sales trends
            monthly_sales = sales_data.groupby(sales_data['date'].dt.to_period('M'))['amount'].sum()
            
            # Calculate trend using linear regression
            from sklearn.linear_model import LinearRegression
            
            # Prepare data for trend analysis
            months = np.arange(len(monthly_sales)).reshape(-1, 1)
            sales_values = monthly_sales.values
            
            # Fit linear regression
            trend_model = LinearRegression()
            trend_model.fit(months, sales_values)
            
            # Predict next 3 months
            future_months = np.arange(len(monthly_sales), len(monthly_sales) + 3).reshape(-1, 1)
            future_predictions = trend_model.predict(future_months)
            
            # Calculate trend metrics
            trend_slope = trend_model.coef_[0]
            trend_direction = "increasing" if trend_slope > 0 else "decreasing"
            
            return {
                "trend_analysis": {
                    "trend_direction": trend_direction,
                    "trend_slope": round(trend_slope, 2),
                    "trend_strength": round(abs(trend_slope) / monthly_sales.mean() * 100, 2)
                },
                "future_predictions": {
                    "next_month": round(future_predictions[0], 2),
                    "next_2_months": round(future_predictions[1], 2),
                    "next_3_months": round(future_predictions[2], 2)
                },
                "historical_trend": monthly_sales.to_dict(),
                "trend_confidence": round(trend_model.score(months, sales_values), 2)
            }
            
        except Exception as e:
            return {"error": f"Sales trend prediction failed: {str(e)}"}
    
    def predict_customer_churn(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict customer churn probability"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Customer churn prediction
            customer_data = data[data['amount'] > 0].copy()
            customer_data['date'] = pd.to_datetime(customer_data['date'])
            
            # Calculate customer features
            current_date = customer_data['date'].max()
            customer_features = customer_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean', 'std'],
                'date': ['min', 'max'],
                'voucher_number': 'nunique'
            })
            
            # Flatten column names
            customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns]
            
            # Calculate churn features
            customer_features['days_since_last'] = (current_date - pd.to_datetime(customer_features['date_max'])).dt.days
            customer_features['customer_lifetime'] = (
                pd.to_datetime(customer_features['date_max']) - 
                pd.to_datetime(customer_features['date_min'])
            ).dt.days
            
            customer_features['purchase_frequency'] = customer_features['voucher_number_nunique'] / (customer_features['customer_lifetime'] + 1)
            customer_features['spending_consistency'] = customer_features['amount_std'] / customer_features['amount_mean']
            
            # Simple churn prediction based on recency
            churn_threshold = params.get('churn_threshold', 90)  # days
            
            # Calculate churn probability
            def calculate_churn_probability(days_since_last, avg_frequency):
                if days_since_last <= 30:
                    return 0.1  # Low risk
                elif days_since_last <= 60:
                    return 0.3  # Medium risk
                elif days_since_last <= 90:
                    return 0.6  # High risk
                else:
                    return 0.9  # Very high risk
            
            customer_features['churn_probability'] = customer_features.apply(
                lambda row: calculate_churn_probability(row['days_since_last'], row['purchase_frequency']), 
                axis=1
            )
            
            # Segment customers by churn risk
            def churn_risk_segment(prob):
                if prob <= 0.3:
                    return 'Low Risk'
                elif prob <= 0.6:
                    return 'Medium Risk'
                else:
                    return 'High Risk'
            
            customer_features['churn_risk'] = customer_features['churn_probability'].apply(churn_risk_segment)
            
            # High-risk customers
            high_risk_customers = customer_features[customer_features['churn_risk'] == 'High Risk']
            
            return {
                "total_customers": len(customer_features),
                "churn_risk_segments": customer_features['churn_risk'].value_counts().to_dict(),
                "high_risk_customers": len(high_risk_customers),
                "churn_predictions": {
                    "avg_churn_probability": round(customer_features['churn_probability'].mean(), 2),
                    "customers_at_risk": len(customer_features[customer_features['churn_probability'] > 0.5]),
                    "immediate_attention_needed": len(customer_features[customer_features['churn_probability'] > 0.8])
                },
                "high_risk_details": high_risk_customers[['churn_probability', 'days_since_last', 'amount_sum']].to_dict('index'),
                "revenue_at_risk": round(high_risk_customers['amount_sum'].sum(), 2)
            }
            
        except Exception as e:
            return {"error": f"Customer churn prediction failed: {str(e)}"}
    
    def forecast_seasonal_sales(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast seasonal sales patterns"""
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
                    'forecasted_sales': round(forecasted_amount, 2),
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
            return {"error": f"Seasonal sales forecasting failed: {str(e)}"}
    
    def forecast_product_demand(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast product demand"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Product demand forecasting
            product_data = data[(data['amount'] > 0) & (data['item'].notna())].copy()
            
            if len(product_data) == 0:
                return {"error": "No product data available"}
            
            # Analyze top products
            top_products = product_data.groupby('item')['amount'].sum().nlargest(10)
            
            # Forecast for each top product
            product_forecasts = {}
            for product in top_products.index:
                product_sales = product_data[product_data['item'] == product]
                product_sales['date'] = pd.to_datetime(product_sales['date'])
                
                # Monthly sales for this product
                monthly_sales = product_sales.groupby(product_sales['date'].dt.to_period('M'))['amount'].sum()
                
                if len(monthly_sales) > 3:
                    # Simple trend-based forecast
                    trend = monthly_sales.pct_change().mean()
                    last_month_sales = monthly_sales.iloc[-1]
                    
                    # Forecast next 3 months
                    forecasts = []
                    for i in range(1, 4):
                        predicted_sales = last_month_sales * (1 + trend) ** i
                        forecasts.append(round(predicted_sales, 2))
                    
                    product_forecasts[product] = {
                        'current_sales': round(last_month_sales, 2),
                        'forecasts': forecasts,
                        'trend': round(trend * 100, 2)
                    }
            
            return {
                "product_forecasts": product_forecasts,
                "top_products": top_products.to_dict(),
                "total_products_forecasted": len(product_forecasts)
            }
            
        except Exception as e:
            return {"error": f"Product demand forecasting failed: {str(e)}"}
    
    def predict_market_opportunities(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict market opportunities"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Market opportunity analysis
            sales_data = data[data['amount'] > 0].copy()
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            
            # Customer expansion opportunities
            customer_metrics = sales_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean'],
                'voucher_number': 'nunique'
            })
            
            # Flatten column names
            customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns]
            
            # Identify growth opportunities
            high_frequency_low_value = customer_metrics[
                (customer_metrics['voucher_number_nunique'] > customer_metrics['voucher_number_nunique'].quantile(0.8)) &
                (customer_metrics['amount_mean'] < customer_metrics['amount_mean'].quantile(0.5))
            ]
            
            low_frequency_high_value = customer_metrics[
                (customer_metrics['voucher_number_nunique'] < customer_metrics['voucher_number_nunique'].quantile(0.5)) &
                (customer_metrics['amount_mean'] > customer_metrics['amount_mean'].quantile(0.8))
            ]
            
            # Market size estimation
            total_addressable_market = customer_metrics['amount_sum'].sum()
            avg_customer_value = customer_metrics['amount_sum'].mean()
            
            return {
                "market_opportunities": {
                    "upsell_opportunities": len(high_frequency_low_value),
                    "cross_sell_opportunities": len(low_frequency_high_value),
                    "customer_expansion_potential": round(avg_customer_value * 0.3, 2)  # 30% growth potential
                },
                "market_size": {
                    "total_addressable_market": round(total_addressable_market, 2),
                    "avg_customer_value": round(avg_customer_value, 2),
                    "growth_potential": round(total_addressable_market * 0.2, 2)  # 20% growth potential
                },
                "opportunity_details": {
                    "high_frequency_low_value": high_frequency_low_value.to_dict('index'),
                    "low_frequency_high_value": low_frequency_high_value.to_dict('index')
                }
            }
            
        except Exception as e:
            return {"error": f"Market opportunity prediction failed: {str(e)}"}
    
    def predict_customer_behavior(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict customer behavior patterns"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # Customer behavior prediction
            customer_data = data[data['amount'] > 0].copy()
            customer_data['date'] = pd.to_datetime(customer_data['date'])
            
            # Calculate customer behavior features
            customer_behavior = customer_data.groupby('party_name').agg({
                'amount': ['sum', 'count', 'mean', 'std'],
                'date': ['min', 'max'],
                'voucher_type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
            })
            
            # Flatten column names
            customer_behavior.columns = ['_'.join(col).strip() for col in customer_behavior.columns]
            
            # Predict next purchase probability
            current_date = customer_data['date'].max()
            customer_behavior['days_since_last'] = (current_date - pd.to_datetime(customer_behavior['date_max'])).dt.days
            customer_behavior['avg_purchase_interval'] = (
                pd.to_datetime(customer_behavior['date_max']) - 
                pd.to_datetime(customer_behavior['date_min'])
            ).dt.days / customer_behavior['amount_count']
            
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
            
            customer_behavior['next_purchase_probability'] = customer_behavior.apply(
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
            
            customer_behavior['behavioral_segment'] = customer_behavior['next_purchase_probability'].apply(behavioral_segment)
            
            return {
                "behavior_predictions": len(customer_behavior),
                "behavioral_segments": customer_behavior['behavioral_segment'].value_counts().to_dict(),
                "purchase_probabilities": {
                    "avg_probability": round(customer_behavior['next_purchase_probability'].mean(), 2),
                    "high_probability_customers": len(customer_behavior[customer_behavior['next_purchase_probability'] > 0.7]),
                    "at_risk_customers": len(customer_behavior[customer_behavior['next_purchase_probability'] < 0.3])
                },
                "predicted_revenue": round(
                    (customer_behavior['amount_mean'] * customer_behavior['next_purchase_probability']).sum(), 2
                )
            }
            
        except Exception as e:
            return {"error": f"Customer behavior prediction failed: {str(e)}"}
    
    def general_sales_forecast(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General sales forecasting"""
        try:
            # Convert decimal columns to numeric
            data = self.convert_decimal_columns(data, ['amount', 'quantity', 'rate'])
            
            # General forecast
            sales_data = data[data['amount'] > 0]
            
            # Basic forecasting metrics
            total_sales = sales_data['amount'].sum()
            avg_daily_sales = sales_data.groupby(pd.to_datetime(sales_data['date']).dt.date)['amount'].sum().mean()
            
            # Simple 30-day forecast
            forecast_30_days = avg_daily_sales * 30
            
            return {
                "overall_trends": {
                    "total_historical_sales": round(total_sales, 2),
                    "avg_daily_sales": round(avg_daily_sales, 2),
                    "forecast_30_days": round(forecast_30_days, 2)
                },
                "forecast_confidence": "Medium",
                "data_quality": {
                    "total_records": len(sales_data),
                    "date_range": f"{sales_data['date'].min()} to {sales_data['date'].max()}"
                }
            }
            
        except Exception as e:
            return {"error": f"General sales forecasting failed: {str(e)}"}
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the type of predictive analysis based on query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['revenue', 'forecast', 'predict', 'sales']):
            return 'revenue_forecasting'
        elif any(word in query_lower for word in ['lifetime', 'value', 'clv', 'ltv']):
            return 'customer_lifetime_value_prediction'
        elif any(word in query_lower for word in ['trend', 'pattern', 'direction']):
            return 'sales_trend_prediction'
        elif any(word in query_lower for word in ['churn', 'retention', 'leaving']):
            return 'customer_churn_prediction'
        elif any(word in query_lower for word in ['seasonal', 'quarterly', 'monthly']):
            return 'seasonal_sales_forecast'
        elif any(word in query_lower for word in ['product', 'demand', 'item']):
            return 'product_demand_forecast'
        elif any(word in query_lower for word in ['opportunity', 'market', 'growth']):
            return 'market_opportunity_prediction'
        elif any(word in query_lower for word in ['behavior', 'next', 'purchase']):
            return 'customer_behavior_prediction'
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