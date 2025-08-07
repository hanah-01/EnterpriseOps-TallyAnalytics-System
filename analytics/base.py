"""
Base Analytics Framework for 4-Tier Business Intelligence
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(__name__)


class AnalyticsResponse(BaseModel):
    """Response model for analytics operations"""
    analytics_type: str  # 'descriptive', 'diagnostic', 'predictive', 'prescriptive'
    query: str
    results: Dict[str, Any]
    insights: List[str]
    recommendations: Optional[List[str]] = None
    confidence_level: Optional[float] = None
    execution_time_ms: int
    model_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}


class AnalyticsBase(ABC):
    """Base class for all analytics implementations"""
    
    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        self.logger = structlog.get_logger(f"{__name__}.{name}")
        
    @abstractmethod
    def analyze(self, query: str, data: pd.DataFrame, params: Dict[str, Any]) -> AnalyticsResponse:
        """Perform analytics on the provided data"""
        pass
    
    @abstractmethod
    def get_supported_queries(self) -> List[str]:
        """Return list of supported query types"""
        pass
    
    def validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that data contains required columns"""
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    def calculate_execution_time(self, start_time: datetime) -> int:
        """Calculate execution time in milliseconds"""
        return int((datetime.now() - start_time).total_seconds() * 1000)
    
    def format_currency(self, amount: float, currency: str = "INR") -> str:
        """Format currency amount"""
        if currency == "INR":
            return f"₹{amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    
    def calculate_percentage_change(self, current: float, previous: float) -> float:
        """Calculate percentage change"""
        if previous == 0:
            return 0.0
        return ((current - previous) / previous) * 100
    
    def identify_trends(self, data: pd.Series, periods: int = 3) -> str:
        """Identify trend direction in time series data"""
        if len(data) < periods:
            return "insufficient_data"
        
        recent_avg = data.tail(periods).mean()
        previous_avg = data.head(periods).mean()
        
        change = self.calculate_percentage_change(recent_avg, previous_avg)
        
        if change > 5:
            return "increasing"
        elif change < -5:
            return "decreasing"
        else:
            return "stable"
    
    def detect_anomalies(self, data: pd.Series, threshold: float = 2.0) -> List[int]:
        """Detect anomalies using z-score method"""
        z_scores = np.abs((data - data.mean()) / data.std())
        return data[z_scores > threshold].index.tolist()
    
    def calculate_correlation(self, data: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """Calculate correlation between target and other numeric columns"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        correlations = {}
        
        for col in numeric_columns:
            if col != target_column:
                correlation = data[target_column].corr(data[col])
                if not np.isnan(correlation):
                    correlations[col] = correlation
        
        return correlations
    
    def generate_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate textual insights from analysis results"""
        insights = []
        
        # This is a base implementation - subclasses should override
        if 'trends' in analysis_results:
            trend = analysis_results['trends']
            insights.append(f"Data shows a {trend} trend over the analyzed period")
        
        if 'anomalies' in analysis_results:
            anomaly_count = len(analysis_results['anomalies'])
            if anomaly_count > 0:
                insights.append(f"Detected {anomaly_count} anomalous data points")
        
        return insights
    
    def classify_query_intent(self, query: str) -> str:
        """Classify query intent for analytics type routing"""
        query_lower = query.lower()
        
        # Descriptive patterns
        if any(word in query_lower for word in ['what', 'show', 'current', 'total', 'balance', 'summary']):
            return 'descriptive'
        
        # Diagnostic patterns
        if any(word in query_lower for word in ['why', 'cause', 'reason', 'analyze', 'explain']):
            return 'diagnostic'
        
        # Predictive patterns
        if any(word in query_lower for word in ['predict', 'forecast', 'future', 'next', 'will']):
            return 'predictive'
        
        # Prescriptive patterns
        if any(word in query_lower for word in ['recommend', 'suggest', 'optimize', 'should', 'how to']):
            return 'prescriptive'
        
        return 'descriptive'  # Default to descriptive
    
    def prepare_response(
        self, 
        analytics_type: str, 
        query: str, 
        results: Dict[str, Any], 
        start_time: datetime,
        confidence_level: Optional[float] = None,
        model_info: Optional[Dict[str, Any]] = None
    ) -> AnalyticsResponse:
        """Prepare standardized analytics response"""
        
        insights = self.generate_insights(results)
        recommendations = self.generate_recommendations(results) if analytics_type in ['diagnostic', 'prescriptive'] else None
        
        return AnalyticsResponse(
            analytics_type=analytics_type,
            query=query,
            results=results,
            insights=insights,
            recommendations=recommendations,
            confidence_level=confidence_level,
            execution_time_ms=self.calculate_execution_time(start_time),
            model_info=model_info,
            metadata={
                'analyzer': self.name,
                'agent_type': self.agent_type,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results"""
        # Base implementation - subclasses should override
        return []
    
    def convert_decimal_columns(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Convert decimal columns to numeric types"""
        data_copy = data.copy()
        for col in columns:
            if col in data_copy.columns:
                try:
                    # Convert decimal.Decimal to float
                    data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
                except Exception as e:
                    self.logger.warning(f"Failed to convert column {col} to numeric: {e}")
        return data_copy