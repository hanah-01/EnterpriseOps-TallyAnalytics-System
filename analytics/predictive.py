"""
Predictive Analytics - "What will happen?"
Forecasting and trend projection using ML/DL models
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from .base import AnalyticsBase, AnalyticsResponse


class PredictiveAnalytics(AnalyticsBase):
    """Predictive analytics for forecasting and trend projection"""
    
    def __init__(self, agent_type: str = "financial"):
        super().__init__("PredictiveAnalytics", agent_type)
        self.supported_queries = [
            'cash_flow_forecast',
            'payment_prediction',
            'demand_forecast',
            'risk_assessment',
            'seasonal_forecast',
            'trend_projection'
        ]
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
    
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform predictive analytics on the provided data"""
        start_time = datetime.now()
        
        analysis_type = self.classify_analysis_type(query)
        
        if analysis_type == 'cash_flow_forecast':
            results = self.forecast_cash_flow(data, params)
        elif analysis_type == 'payment_prediction':
            results = self.predict_payments(data, params)
        elif analysis_type == 'demand_forecast':
            results = self.forecast_demand(data, params)
        elif analysis_type == 'risk_assessment':
            results = self.assess_risk(data, params)
        elif analysis_type == 'seasonal_forecast':
            results = self.forecast_seasonal(data, params)
        elif analysis_type == 'trend_projection':
            results = self.project_trends(data, params)
        else:
            results = self.general_forecast(data, params)
        
        confidence_level = results.get('confidence_level', 0.75)
        model_info = results.get('model_info', {})
        
        return self.prepare_response(
            analytics_type='predictive',
            query=query,
            results=results,
            start_time=start_time,
            confidence_level=confidence_level,
            model_info=model_info
        )
    
    def classify_analysis_type(self, query: str) -> str:
        """Classify the specific type of predictive analysis needed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['cash flow', 'cashflow', 'liquidity']):
            return 'cash_flow_forecast'
        elif any(word in query_lower for word in ['payment', 'collection', 'receivable']):
            return 'payment_prediction'
        elif any(word in query_lower for word in ['demand', 'sales', 'volume']):
            return 'demand_forecast'
        elif any(word in query_lower for word in ['risk', 'probability', 'chance']):
            return 'risk_assessment'
        elif any(word in query_lower for word in ['seasonal', 'monthly', 'quarterly']):
            return 'seasonal_forecast'
        elif any(word in query_lower for word in ['trend', 'projection', 'future']):
            return 'trend_projection'
        else:
            return 'general_forecast'
    
    def forecast_cash_flow(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast cash flow using time series and regression models"""
        required_columns = ['date', 'amount']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for cash flow forecasting'}
        
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
        # Prepare time series features
        features_df = self.prepare_time_series_features(data)
        
        # Split data for training and validation
        train_size = int(len(features_df) * 0.8)
        train_data = features_df[:train_size]
        val_data = features_df[train_size:]
        
        # Prepare features and target
        feature_cols = [col for col in features_df.columns if col not in ['date', 'amount']]
        X_train = train_data[feature_cols]
        y_train = train_data['amount']
        X_val = val_data[feature_cols]
        y_val = val_data['amount']
        
        # Train multiple models and select best
        model_results = {}
        best_model = None
        best_score = float('inf')
        
        for model_name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                
                mae = mean_absolute_error(y_val, predictions)
                rmse = np.sqrt(mean_squared_error(y_val, predictions))
                r2 = r2_score(y_val, predictions)
                
                model_results[model_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': predictions.tolist()
                }
                
                if mae < best_score:
                    best_score = mae
                    best_model = model_name
                    
            except Exception as e:
                model_results[model_name] = {'error': str(e)}
        
        # Generate forecasts
        forecast_periods = params.get('forecast_periods', 12)
        forecasts = self.generate_forecasts(
            features_df, 
            self.models[best_model], 
            forecast_periods,
            feature_cols
        )
        
        # Calculate confidence intervals
        confidence_intervals = self.calculate_confidence_intervals(
            forecasts, 
            model_results[best_model]['mae']
        )
        
        return {
            'forecast_summary': {
                'forecast_periods': forecast_periods,
                'best_model': best_model,
                'model_accuracy': {
                    'mae': model_results[best_model]['mae'],
                    'rmse': model_results[best_model]['rmse'],
                    'r2': model_results[best_model]['r2']
                }
            },
            'forecasts': forecasts,
            'confidence_intervals': confidence_intervals,
            'model_comparison': model_results,
            'assumptions': self.get_forecast_assumptions(data),
            'confidence_level': max(0.5, min(0.95, model_results[best_model]['r2'])),
            'model_info': {
                'best_model': best_model,
                'training_size': len(train_data),
                'validation_size': len(val_data),
                'feature_count': len(feature_cols)
            }
        }
    
    def predict_payments(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict payment probabilities and timing"""
        required_columns = ['customer_name', 'amount', 'due_date']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for payment prediction'}
        
        # Prepare features for payment prediction
        features_df = self.prepare_payment_features(data)
        
        # Train payment probability model
        if 'payment_status' in features_df.columns:
            # Binary classification for payment probability
            payment_predictions = self.train_payment_classifier(features_df)
        else:
            # Use heuristic approach if no historical payment data
            payment_predictions = self.heuristic_payment_prediction(features_df)
        
        # Predict payment timing
        timing_predictions = self.predict_payment_timing(features_df, params)
        
        return {
            'payment_predictions': payment_predictions,
            'timing_predictions': timing_predictions,
            'risk_assessment': self.assess_payment_risk(payment_predictions),
            'recommendations': self.generate_payment_recommendations(payment_predictions),
            'confidence_level': 0.72,
            'model_info': {
                'prediction_method': 'ensemble' if 'payment_status' in features_df.columns else 'heuristic',
                'customers_analyzed': len(features_df)
            }
        }
    
    def forecast_demand(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast demand for products or services"""
        required_columns = ['date', 'product', 'quantity']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for demand forecasting'}
        
        data['date'] = pd.to_datetime(data['date'])
        
        # Forecast demand for each product
        product_forecasts = {}
        
        for product in data['product'].unique():
            product_data = data[data['product'] == product].copy()
            
            if len(product_data) < 10:  # Minimum data requirement
                continue
                
            # Prepare features
            features_df = self.prepare_demand_features(product_data)
            
            # Train model
            forecast = self.train_demand_model(features_df, params)
            
            product_forecasts[product] = forecast
        
        # Aggregate forecasts
        total_forecast = self.aggregate_demand_forecasts(product_forecasts)
        
        return {
            'demand_forecasts': product_forecasts,
            'total_forecast': total_forecast,
            'market_assumptions': self.get_demand_assumptions(data),
            'confidence_level': 0.68,
            'model_info': {
                'products_analyzed': len(product_forecasts),
                'forecasting_method': 'time_series_regression'
            }
        }
    
    def assess_risk(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assess various types of business risks"""
        risk_assessments = {}
        
        # Financial risk assessment
        if 'amount' in data.columns:
            financial_risk = self.assess_financial_risk(data)
            risk_assessments['financial'] = financial_risk
        
        # Customer risk assessment
        if 'customer_name' in data.columns:
            customer_risk = self.assess_customer_risk(data)
            risk_assessments['customer'] = customer_risk
        
        # Market risk assessment
        if 'date' in data.columns:
            market_risk = self.assess_market_risk(data)
            risk_assessments['market'] = market_risk
        
        # Overall risk score
        overall_risk = self.calculate_overall_risk(risk_assessments)
        
        return {
            'risk_assessments': risk_assessments,
            'overall_risk': overall_risk,
            'risk_factors': self.identify_risk_factors(data),
            'mitigation_strategies': self.suggest_risk_mitigation(risk_assessments),
            'confidence_level': 0.75,
            'model_info': {
                'risk_types_analyzed': len(risk_assessments),
                'assessment_method': 'statistical_analysis'
            }
        }
    
    def forecast_seasonal(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast seasonal patterns and trends"""
        required_columns = ['date', 'amount']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for seasonal forecasting'}
        
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
        # Decompose time series
        seasonal_decomposition = self.decompose_time_series(data)
        
        # Forecast seasonal components
        seasonal_forecast = self.forecast_seasonal_components(seasonal_decomposition, params)
        
        return {
            'seasonal_decomposition': seasonal_decomposition,
            'seasonal_forecast': seasonal_forecast,
            'seasonal_patterns': self.identify_seasonal_patterns(data),
            'confidence_level': 0.71,
            'model_info': {
                'decomposition_method': 'additive',
                'forecast_horizon': params.get('forecast_periods', 12)
            }
        }
    
    def project_trends(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Project trends into the future"""
        required_columns = ['date', 'amount']
        if not self.validate_data(data, required_columns):
            return {'error': 'Invalid data format for trend projection'}
        
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
        # Fit trend models
        trend_models = self.fit_trend_models(data)
        
        # Project trends
        projection_periods = params.get('projection_periods', 12)
        trend_projections = self.project_trend_models(trend_models, projection_periods)
        
        return {
            'trend_analysis': trend_models,
            'trend_projections': trend_projections,
            'trend_confidence': self.assess_trend_confidence(trend_models),
            'confidence_level': 0.69,
            'model_info': {
                'trend_models_used': len(trend_models),
                'projection_periods': projection_periods
            }
        }
    
    def general_forecast(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """General forecasting for any time series data"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return {'error': 'No numeric columns found for forecasting'}
        
        forecasts = {}
        
        for column in numeric_columns:
            if data[column].notna().sum() < 5:  # Minimum data requirement
                continue
                
            # Simple moving average forecast
            forecast = self.simple_moving_average_forecast(data[column], params)
            forecasts[column] = forecast
        
        return {
            'column_forecasts': forecasts,
            'forecasting_method': 'moving_average',
            'confidence_level': 0.60,
            'model_info': {
                'columns_forecasted': len(forecasts),
                'method': 'simple_moving_average'
            }
        }
    
    # Helper methods for specific forecasting tasks
    
    def prepare_time_series_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series features for modeling"""
        features_df = data.copy()
        
        # Add time-based features
        features_df['year'] = features_df['date'].dt.year
        features_df['month'] = features_df['date'].dt.month
        features_df['day'] = features_df['date'].dt.day
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['day_of_year'] = features_df['date'].dt.dayofyear
        
        # Add lag features
        for lag in [1, 3, 7, 30]:
            features_df[f'lag_{lag}'] = features_df['amount'].shift(lag)
        
        # Add rolling statistics
        for window in [3, 7, 30]:
            features_df[f'rolling_mean_{window}'] = features_df['amount'].rolling(window=window).mean()
            features_df[f'rolling_std_{window}'] = features_df['amount'].rolling(window=window).std()
        
        # Drop rows with NaN values
        features_df = features_df.dropna()
        
        return features_df
    
    def generate_forecasts(self, features_df: pd.DataFrame, model, periods: int, feature_cols: List[str]) -> List[Dict[str, Any]]:
        """Generate forecasts for future periods"""
        forecasts = []
        
        # Get the last known values
        last_row = features_df.iloc[-1].copy()
        
        for i in range(periods):
            # Predict next value
            X_pred = last_row[feature_cols].values.reshape(1, -1)
            prediction = model.predict(X_pred)[0]
            
            # Create forecast entry
            forecast_date = pd.to_datetime(last_row['date']) + timedelta(days=i+1)
            
            forecasts.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'predicted_amount': prediction,
                'period': i + 1
            })
            
            # Update features for next prediction (simplified)
            last_row['amount'] = prediction
            last_row['date'] = forecast_date
        
        return forecasts
    
    def calculate_confidence_intervals(self, forecasts: List[Dict], mae: float) -> List[Dict[str, Any]]:
        """Calculate confidence intervals for forecasts"""
        confidence_intervals = []
        
        for i, forecast in enumerate(forecasts):
            # Simple confidence interval calculation
            # In practice, this would use more sophisticated methods
            prediction = forecast['predicted_amount']
            
            # Confidence interval widens with forecast horizon
            uncertainty = mae * (1 + 0.1 * i)
            
            confidence_intervals.append({
                'date': forecast['date'],
                'lower_bound': prediction - uncertainty,
                'upper_bound': prediction + uncertainty,
                'confidence_width': uncertainty * 2
            })
        
        return confidence_intervals
    
    def get_forecast_assumptions(self, data: pd.DataFrame) -> List[str]:
        """Get assumptions made in forecasting"""
        assumptions = [
            "Historical patterns will continue",
            "No major structural changes in business",
            "Economic conditions remain stable",
            "Seasonal patterns persist"
        ]
        
        return assumptions
    
    def prepare_payment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for payment prediction"""
        features_df = data.copy()
        
        # Convert dates
        features_df['due_date'] = pd.to_datetime(features_df['due_date'])
        
        # Add customer features
        customer_stats = features_df.groupby('customer_name').agg({
            'amount': ['count', 'sum', 'mean', 'std']
        }).round(2)
        
        # Flatten column names
        customer_stats.columns = ['_'.join(col).strip() for col in customer_stats.columns]
        
        # Merge back to main dataframe
        features_df = features_df.merge(customer_stats, left_on='customer_name', right_index=True)
        
        # Add amount-based features
        features_df['amount_bin'] = pd.cut(features_df['amount'], bins=5, labels=False)
        features_df['is_high_value'] = (features_df['amount'] > features_df['amount'].quantile(0.8)).astype(int)
        
        return features_df
    
    def heuristic_payment_prediction(self, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict payments using heuristic approach"""
        predictions = []
        
        for _, row in features_df.iterrows():
            # Simple heuristic based on customer history and amount
            base_probability = 0.75  # Base payment probability
            
            # Adjust based on customer transaction count
            if row['amount_count'] > 10:
                base_probability += 0.15
            elif row['amount_count'] < 3:
                base_probability -= 0.15
            
            # Adjust based on amount
            if row['is_high_value']:
                base_probability -= 0.10
            
            # Ensure probability is between 0 and 1
            probability = max(0.1, min(0.95, base_probability))
            
            predictions.append({
                'customer_name': row['customer_name'],
                'amount': row['amount'],
                'payment_probability': probability,
                'risk_level': 'high' if probability < 0.5 else 'medium' if probability < 0.8 else 'low'
            })
        
        return predictions
    
    def predict_payment_timing(self, features_df: pd.DataFrame, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict payment timing"""
        timing_predictions = []
        
        for _, row in features_df.iterrows():
            # Predict days delay based on customer history
            base_delay = 5  # Base delay in days
            
            # Adjust based on customer patterns
            if row['amount_count'] > 10:
                base_delay -= 2
            elif row['amount_count'] < 3:
                base_delay += 5
            
            # Adjust based on amount
            if row['is_high_value']:
                base_delay += 3
            
            predicted_payment_date = pd.to_datetime(row['due_date']) + timedelta(days=base_delay)
            
            timing_predictions.append({
                'customer_name': row['customer_name'],
                'due_date': row['due_date'].strftime('%Y-%m-%d'),
                'predicted_payment_date': predicted_payment_date.strftime('%Y-%m-%d'),
                'predicted_delay_days': base_delay
            })
        
        return timing_predictions
    
    def assess_payment_risk(self, payment_predictions: List[Dict]) -> Dict[str, Any]:
        """Assess overall payment risk"""
        total_amount = sum(p['amount'] for p in payment_predictions)
        weighted_probability = sum(p['amount'] * p['payment_probability'] for p in payment_predictions) / total_amount
        
        high_risk_count = sum(1 for p in payment_predictions if p['risk_level'] == 'high')
        
        return {
            'overall_payment_probability': weighted_probability,
            'high_risk_customers': high_risk_count,
            'expected_collection_rate': weighted_probability * 100,
            'potential_bad_debt': total_amount * (1 - weighted_probability)
        }
    
    def generate_payment_recommendations(self, payment_predictions: List[Dict]) -> List[str]:
        """Generate recommendations based on payment predictions"""
        recommendations = []
        
        high_risk_customers = [p for p in payment_predictions if p['risk_level'] == 'high']
        
        if high_risk_customers:
            recommendations.append(f"Focus collection efforts on {len(high_risk_customers)} high-risk customers")
        
        recommendations.append("Implement early payment incentives for medium-risk customers")
        recommendations.append("Consider credit terms adjustment for new customers")
        
        return recommendations
    
    def simple_moving_average_forecast(self, series: pd.Series, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simple moving average forecast"""
        window = params.get('window', 5)
        periods = params.get('forecast_periods', 3)
        
        # Calculate moving average
        moving_avg = series.rolling(window=window).mean().iloc[-1]
        
        # Generate forecasts
        forecasts = [moving_avg] * periods
        
        return {
            'method': 'moving_average',
            'window_size': window,
            'forecasts': forecasts,
            'forecast_periods': periods
        }
    
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        return self.supported_queries
    
    def generate_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate insights specific to predictive analytics"""
        insights = []
        
        # Forecast insights
        if 'forecast_summary' in analysis_results:
            forecast = analysis_results['forecast_summary']
            best_model = forecast.get('best_model', 'unknown')
            r2_score = forecast.get('model_accuracy', {}).get('r2', 0)
            
            insights.append(f"Best forecasting model: {best_model} with R² score of {r2_score:.2f}")
        
        # Risk insights
        if 'overall_risk' in analysis_results:
            risk_level = analysis_results['overall_risk'].get('risk_level', 'unknown')
            insights.append(f"Overall risk level assessed as: {risk_level}")
        
        # Payment insights
        if 'payment_predictions' in analysis_results:
            predictions = analysis_results['payment_predictions']
            high_risk = sum(1 for p in predictions if p.get('risk_level') == 'high')
            insights.append(f"Identified {high_risk} high-risk payment customers")
        
        return insights