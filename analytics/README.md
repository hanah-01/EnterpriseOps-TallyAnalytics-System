# 4-Tier Analytics Framework for Multi-Agent Tally System

## 📊 Overview

This is a comprehensive 4-tier analytics framework designed specifically for your Tally ERP multi-agent system. It provides business intelligence capabilities across all domains: Financial, Sales, Inventory, Purchase, and Tax analytics.

## 🏗️ Architecture

### 4-Tier Analytics Structure

1. **📈 Descriptive Analytics** - "What happened?"
   - Historical summaries and KPIs
   - Current state reporting
   - Trend identification

2. **🔍 Diagnostic Analytics** - "Why did it happen?"
   - Root cause analysis
   - Correlation discovery
   - Pattern investigation

3. **🔮 Predictive Analytics** - "What will happen?"
   - Forecasting and projections
   - Trend predictions
   - Risk assessments

4. **💡 Prescriptive Analytics** - "What should we do?"
   - Strategic recommendations
   - Optimization strategies
   - Action plans

## 🚀 Quick Start

### 1. Installation

```bash
# Install required packages
pip install scikit-learn>=1.3.0 scipy>=1.10.0 statsmodels>=0.14.0

# Optional advanced packages
pip install prophet>=1.1.4 xgboost>=2.0.0 lightgbm>=4.0.0

# Or use the setup script
python analytics/setup_analytics.py
```

### 2. Basic Usage

```python
from analytics import SalesDescriptiveAnalytics, AnalyticsPromptTemplates

# Initialize analytics
sales_analytics = SalesDescriptiveAnalytics()

# Analyze data
result = sales_analytics.analyze(
    query="Show sales performance summary",
    data=sales_dataframe,
    params={'period': 'monthly'}
)

# Access results
print(result.insights)
print(result.recommendations)
print(f"Confidence: {result.confidence_level}")
```

### 3. Domain-Specific Analytics

#### Financial Analytics
```python
from analytics import DescriptiveAnalytics

financial = DescriptiveAnalytics("financial")
result = financial.analyze("cash flow analysis", data, {})
```

#### Sales Analytics  
```python
from analytics import SalesDescriptiveAnalytics

sales = SalesDescriptiveAnalytics()
result = sales.analyze("customer segmentation analysis", data, {})
```

#### Inventory Analytics
```python  
from analytics import InventoryDescriptiveAnalytics

inventory = InventoryDescriptiveAnalytics()
result = inventory.analyze("stock turnover analysis", data, {})
```

## 📋 Supported Query Types

### Financial Analytics
- `cash_flow_analysis`
- `account_balance_summary` 
- `financial_position`
- `payment_prediction`
- `liquidity_optimization`

### Sales Analytics
- `revenue_analysis`
- `customer_segmentation`
- `product_performance`
- `sales_forecasting`
- `customer_retention_strategy`

### Inventory Analytics
- `stock_level_analysis`
- `turnover_analysis`
- `reorder_optimization`
- `demand_forecasting`
- `dead_stock_identification`

### Purchase Analytics
- `spend_analysis`
- `supplier_performance`
- `cost_optimization`
- `purchase_forecasting`
- `vendor_risk_assessment`

## 🔧 Integration with Existing Agents

### Update Your Agent Classes

```python
from analytics import (
    DescriptiveAnalytics, DiagnosticAnalytics,
    PredictiveAnalytics, PrescriptiveAnalytics,
    AnalyticsPromptTemplates
)

class YourAgent(BaseAgent):
    def __init__(self):
        super().__init__(...)
        
        # Initialize 4-tier analytics
        self.descriptive = DescriptiveAnalytics(self.agent_type.value)
        self.diagnostic = DiagnosticAnalytics(self.agent_type.value)
        self.predictive = PredictiveAnalytics(self.agent_type.value)
        self.prescriptive = PrescriptiveAnalytics(self.agent_type.value)
    
    def _build_system_prompt(self, context=None):
        return AnalyticsPromptTemplates.get_prompt_for_agent_type(
            self.agent_type.value, context
        )
```

### System Prompts

The framework includes domain-specific system prompts:
- Financial Analytics Agent prompts
- Sales Analytics Agent prompts  
- Inventory Analytics Agent prompts
- Purchase Analytics Agent prompts
- Tax Analytics Agent prompts

## 📊 Response Format

All analytics return standardized `AnalyticsResponse` objects:

```python
class AnalyticsResponse:
    analytics_type: str        # 'descriptive', 'diagnostic', etc.
    query: str                 # Original query
    results: Dict[str, Any]    # Analysis results
    insights: List[str]        # Generated insights
    recommendations: List[str] # Action recommendations (optional)
    confidence_level: float    # Confidence in results (optional)
    execution_time_ms: int     # Processing time
    model_info: Dict          # Model details (optional)
    metadata: Dict            # Additional information
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python analytics/test_analytics.py
```

This tests:
- ✅ All 4-tier analytics classes
- ✅ Domain-specific analytics
- ✅ System prompt generation
- ✅ Error handling
- ✅ Data validation

## 📁 File Structure

```
analytics/
├── __init__.py                 # Main exports
├── base.py                     # Base classes
├── descriptive.py              # Descriptive analytics
├── diagnostic.py               # Diagnostic analytics  
├── predictive.py               # Predictive analytics
├── prescriptive.py             # Prescriptive analytics
├── models.py                   # ML/DL models
├── pipeline.py                 # Data pipeline
├── system_prompts.py           # Domain-specific prompts
├── 
├── # Domain-specific analytics
├── sales_descriptive.py        
├── sales_diagnostic.py
├── sales_predictive.py
├── sales_prescriptive.py
├── inventory_*.py
├── purchase_*.py
├──
├── # Utilities
├── setup_analytics.py          # Setup script
├── test_analytics.py           # Test suite
├── integration_helper.py       # Integration helper
└── README.md                   # This file
```

## 🎯 Business Value

### For Financial Management
- Cash flow optimization
- Accounts receivable management
- Financial risk assessment
- Working capital optimization

### For Sales Optimization
- Customer segmentation and retention
- Revenue forecasting
- Product portfolio optimization
- Sales resource allocation

### For Inventory Management
- Stock level optimization
- Demand forecasting
- Dead stock identification
- Reorder point calculation

### For Purchase Optimization
- Supplier performance analysis
- Cost reduction opportunities
- Procurement risk management
- Strategic sourcing

## 🔒 Data Security & Privacy

- No data is sent to external services
- All processing happens locally
- Follows your existing database security patterns
- Respects Tally ERP data access controls

## 🛠️ Customization

### Adding New Analytics Types

1. Extend base analytics classes
2. Implement domain-specific logic
3. Add to `__init__.py` exports
4. Create corresponding system prompts

### Custom Models

```python
from analytics.models import BaseAnalyticsModel

class CustomModel(BaseAnalyticsModel):
    def fit(self, X, y):
        # Your custom model logic
        pass
    
    def predict(self, X):
        # Your prediction logic
        pass
```

## 📚 Examples

### Complete Sales Analysis

```python
import pandas as pd
from analytics import SalesDescriptiveAnalytics

# Initialize
sales_analytics = SalesDescriptiveAnalytics()

# Sample data (replace with your actual data query)
sales_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'party_name': ['Customer A', 'Customer B'] * 50,
    'amount': np.random.exponential(1000, 100),
    'item': ['Product 1', 'Product 2'] * 50
})

# Run analysis
result = sales_analytics.analyze(
    query="Analyze customer performance and identify trends",
    data=sales_data,
    params={'period': 'monthly', 'segment_by': 'customer'}
)

# Display results
print("📊 SALES ANALYSIS RESULTS")
print(f"Analysis Type: {result.analytics_type}")
print(f"Execution Time: {result.execution_time_ms}ms")

print("\\n💡 KEY INSIGHTS:")
for insight in result.insights:
    print(f"• {insight}")

if result.recommendations:
    print("\\n🎯 RECOMMENDATIONS:")
    for rec in result.recommendations:
        print(f"• {rec}")
```

## 🤝 Support

For issues and questions:
1. Check the test suite: `python analytics/test_analytics.py`
2. Review error logs in the console
3. Validate your data format matches expected schema
4. Ensure all required packages are installed

## 🎉 Success Metrics

After integration, you should be able to:
- ✅ Generate 4-tier analytics insights from Tally data
- ✅ Get domain-specific recommendations
- ✅ Use advanced ML/DL models for predictions
- ✅ Maintain existing agent functionality while adding intelligence
- ✅ Scale analytics across different business domains

---

**Your 4-tier analytics framework is now ready to provide comprehensive business intelligence across your entire Tally ERP system! 🚀**
