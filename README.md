# 🕵️ Credit Card Fraud Detection Dashboard

A comprehensive, interactive dashboard built with Streamlit for analyzing credit card fraud patterns and detecting suspicious transactions using machine learning.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-Dashboard-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)

![Dashboard Preview](https://github.com/user-attachments/assets/51fe4886-2091-4b37-8c76-7aef7ceed6cf)

## 🌟 Project Overview

This repository contains a complete credit card fraud detection solution with two main components:

1. **📊 Interactive Streamlit Dashboard** - Real-time fraud analysis and visualization
2. **📓 Jupyter Notebook Pipeline** - End-to-end ML workflow from data to insights

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required packages (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/CodeByHashir/Credit_Card_Fraud_Detection_Dashboard.git
   cd Credit_Card_Fraud_Detection_Dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place your `creditcard.ftr` file in the project directory
   - Or use `creditcard.csv` as an alternative

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Enjoy your interactive fraud detection dashboard!

## 📁 Project Structure

```
Credit_Card_Fraud_Detection_Dashboard/
├── app.py                          # Main Streamlit dashboard application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── creditcard.ftr                  # Sample dataset (Feather format)
├── DataSet Link.csv                # Kaggle dataset (CSV format)
├── creditcard_fraud_end_to_end_minimal.ipynb  # End-to-end Jupyter notebook
└── .gitignore                      # Git ignore file
```

## 💳 Credit Card Fraud Detection — End-to-End Project

An end-to-end, self-contained, interactive credit card fraud detection solution.
This repository covers data loading, EDA, feature engineering, statistical testing, modeling, evaluation, explainability, and a fully interactive single-file HTML dashboard — ready for sharing or hosting on GitHub Pages.

Everything is reproducible from the included Jupyter Notebook and packaged into a polished, offline-capable dashboard.

### 📂 Contents

- **📊 Polished Dashboard** — Single HTML file (no server needed)
- **📓 Minimal End-to-End Notebook** — Full reproducible pipeline
- **📁 Dataset** — Kaggle: Credit Card Fraud Detection
- **⚙️ Modeling Pipeline** — Metrics, plots, explainability
- **📑 Interactive Drill-Down Table** — Search & filter transactions
- **🎨 Clean UI** — Responsive charts and compact layout

### 📁 Files

**creditcard.csv**
- Dataset file from Kaggle.
- Columns: Time, Amount, Class (0 = Non-Fraud, 1 = Fraud), V1–V28 (PCA-like anonymized features).

**creditcard_fraud_end_to_end_minimal.ipynb**
- Minimal pipeline:
  - Load & clean data
  - EDA & class imbalance check
  - Feature engineering (Hour/Day, scaling)
  - Stratified train/validation split
  - Model training (Logistic Regression, tree-based alternative)
  - Evaluation (ROC, PR, Confusion Matrix, AUC, AP)
  - Explainability (Feature importances, PCA)
  - Export interactive HTML dashboard

**creditcard_fraud_dashboard_v2_header_polished.html**
- Final polished dashboard — Overview, Trends, Feature Insights, Model Performance, Drill-Down.

**creditcard_fraud_dashboard_v2.html**
- Earlier dashboard version (kept for reference).

### 🛠 Step-by-Step Workflow

1. **Data Loading**
   - Load creditcard.csv
   - Validate shape, columns, and class imbalance

2. **EDA & Data Quality**
   - Distributions of Amount & Time
   - Fraud vs Non-Fraud counts & totals
   - Engineered Hour & Day features

3. **Feature Engineering**
   - Extract Hour and Day from Time
   - Standardize Amount
   - Keep PCA-like features V1–V28

4. **Train/Validation Split**
   - Stratified split to maintain class ratios

5. **Modeling**
   - Logistic Regression (class_weight for imbalance)
   - Optional: RandomForest/GradientBoosting
   - Predict probabilities & classes

6. **Evaluation**
   - ROC Curve & AUC
   - Precision-Recall Curve & Average Precision
   - Confusion Matrix
   - KPI extraction for dashboard

7. **Explainability & Insights**
   - Feature importances / coefficients
   - PCA 2D visualization
   - Class-based distribution overlays

## 📊 Streamlit Dashboard Features

### 📊 **Overview Dashboard**
- **Real-time KPIs**: Total transactions, fraud rate, fraud count, and total fraud amount
- **Interactive visualizations**: Pie charts and bar graphs for transaction distribution
- **Performance metrics**: Average fraud amounts and data coverage statistics

### 📈 **Trends Analysis**
- **Temporal patterns**: Fraud incidents over time with minute-level granularity
- **Transaction trends**: Amount patterns and hourly fraud distribution
- **Heatmap visualization**: Fraud patterns by hour and day for pattern recognition

### 🔍 **Feature Analysis**
- **Machine Learning Insights**: Random Forest-based feature importance analysis
- **Feature distributions**: Histograms showing fraud vs non-fraud patterns
- **Smart sampling**: Optimized data processing for large datasets

### 🎯 **Model Performance**
- **Classification metrics**: Precision, Recall, and F1-Score
- **ROC Curve**: Model performance visualization with AUC scores
- **Confusion Matrix**: Detailed classification results
- **Precision-Recall Curve**: Performance analysis for imbalanced datasets

### 🔎 **Drill-Down Analysis**
- **Transaction search**: Find specific transactions by ID or amount
- **Advanced filtering**: Sort and filter by multiple criteria
- **Data exploration**: Interactive tables with pagination

### 💡 **Advanced Insights**
- **Risk assessment**: Identify high-risk transaction patterns
- **Anomaly detection**: Statistical outlier analysis
- **Recommendations**: Actionable insights for fraud prevention

## 🛠️ Dependencies

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms
- **Feather**: Fast data storage format

## 📊 Dataset Information

The dashboard works with credit card transaction data containing:
- **Transaction features**: V1-V28 (anonymized features)
- **Amount**: Transaction amount in dollars
- **Time**: Transaction timestamp
- **Class**: Target variable (0 = legitimate, 1 = fraud)

## 🎨 Customization

### Styling
- Custom CSS for professional appearance
- Gradient backgrounds and modern design
- Responsive layout for different screen sizes

### Filters
- **Amount range**: Set minimum and maximum transaction amounts
- **Time filters**: Select specific hours for analysis
- **Advanced options**: Sample size control and fraud-only views

### Visualizations
- **Color schemes**: Consistent fraud/non-fraud color coding
- **Interactive charts**: Hover effects and zoom capabilities
- **Responsive design**: Adapts to different screen sizes

## 🔧 Configuration

### Environment Variables
- No external API keys required
- Local data processing for privacy

### Performance Settings
- **Smart sampling**: Automatic dataset size optimization
- **Caching**: Streamlit caching for improved performance
- **Memory management**: Efficient handling of large datasets

## 📈 Performance Features

- **Real-time analysis**: Instant results with interactive filters
- **Optimized ML models**: Fast training with Random Forest
- **Efficient visualizations**: Smooth rendering of complex charts
- **Responsive interface**: Quick navigation between tabs

## 🚨 Use Cases

### Financial Institutions
- **Fraud detection**: Identify suspicious transaction patterns
- **Risk assessment**: Evaluate transaction risk levels
- **Compliance**: Monitor for regulatory requirements

### Data Scientists
- **Model evaluation**: Test and compare ML algorithms
- **Feature analysis**: Understand important predictors
- **Performance metrics**: Comprehensive model assessment

### Business Analysts
- **Pattern recognition**: Identify fraud trends over time
- **Operational insights**: Optimize fraud prevention strategies
- **Reporting**: Generate comprehensive fraud analysis reports

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 coding standards
- Add comments for complex logic
- Test with different dataset sizes
- Ensure responsive design


## 🙏 Acknowledgments

- **Streamlit team** for the amazing web framework
- **Scikit-learn community** for ML algorithms
- **Plotly** for interactive visualizations
- **Open source contributors** for inspiration

## 📞 Support

If you have any questions or need help:

- **GitHub Issues**: [Create an issue](https://github.com/CodeByHashir/Credit_Card_Fraud_Detection_Dashboard/issues)
- **Email**: Hashirahmad330@Gmail.com
- **Documentation**: Check the code comments and this README

## 🔮 Future Enhancements

- [ ] **Real-time streaming**: Live transaction monitoring
- [ ] **Multiple ML models**: Support for different algorithms
- [ ] **API integration**: Connect to external fraud detection services
- [ ] **Mobile optimization**: Better mobile device support
- [ ] **Export functionality**: Generate PDF/Excel reports
- [ ] **User authentication**: Multi-user support with roles

---

**Built with ❤️ by [CodeByHashir](https://github.com/CodeByHashir)**

*For fraud detection and financial security analysis*
