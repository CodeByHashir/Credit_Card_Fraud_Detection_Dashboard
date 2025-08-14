# Streamlit Credit Card Fraud Detection Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title='Fraud Detection Dashboard',
    page_icon='🕵️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    .metric-card h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🕵️ Credit Card Fraud Detection Dashboard</h1></div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the credit card data"""
    try:
        df_local = pd.read_feather('creditcard.ftr')
        
        # Add derived features if they don't exist
        if 'Hour' not in df_local.columns and 'Time' in df_local.columns:
            df_local['Hour'] = (df_local['Time'] // 3600).astype(int) % 24
        if 'Day' not in df_local.columns and 'Time' in df_local.columns:
            df_local['Day'] = (df_local['Time'] // (3600 * 24)).astype(int)
        if 'TransactionID' not in df_local.columns:
            df_local['TransactionID'] = range(len(df_local))
            
        return df_local
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# Load data
with st.spinner('Loading data...'):
    df = load_data()

# Sidebar filters
st.sidebar.markdown("## 🎛️ Filters & Controls")

# Data overview in sidebar
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.info(f"**Total Records:** {len(df):,}\n\n**Fraud Rate:** {(df['Class'].mean()*100):.3f}%\n\n**Features:** {len(df.columns)}")

# Enhanced filters
st.sidebar.markdown("### 🎛️ Filter Options")

# Amount filter
amount_min, amount_max = float(df['Amount'].min()), float(df['Amount'].max())
amount_range = st.sidebar.slider(
    '💰 Amount Range ($)',
    min_value=amount_min,
    max_value=amount_max,
    value=(0.0, min(1000.0, amount_max)),
    step=1.0
)

# Time filters
col1, col2 = st.sidebar.columns(2)
with col1:
    start_hour = st.selectbox('🌅 Start Hour', range(24), index=0)
with col2:
    end_hour = st.sidebar.selectbox('🌇 End Hour', range(24), index=23)

# Additional filters
fraud_only = st.sidebar.checkbox('🚨 Show Fraud Only', value=False)
show_advanced = st.sidebar.checkbox('⚙️ Advanced Options', value=False)

if show_advanced:
    st.sidebar.markdown("### ⚙️ Advanced Filters")
    min_amount = st.sidebar.number_input('Minimum Amount', value=0.0, step=0.01)
    max_amount = st.sidebar.number_input('Maximum Amount', value=float(df['Amount'].max()), step=0.01)
    sample_size = st.sidebar.slider('Sample Size', 1000, len(df), min(10000, len(df)), step=1000)
else:
    min_amount, max_amount = amount_range
    sample_size = len(df)

# Apply filters
filtered = df[
    (df['Amount'].between(min_amount, max_amount)) & 
    (df['Hour'].between(start_hour, end_hour))
]

if len(filtered) == 0:
    st.error("❌ No data matches the current filters. Please adjust your selection.")
    st.stop()

# Sample data if too large
if len(filtered) > sample_size:
    filtered = filtered.sample(n=sample_size, random_state=42)
    st.info(f"📊 Showing {sample_size:,} random samples from {len(df):,} total records")

# Main tabs
overview, trends, features, performance, drill, insights = st.tabs([
    '📊 Overview', '📈 Trends', '🔍 Features', '🎯 Performance', '🔎 Drill-Down', '💡 Insights'
])

with overview:
    st.markdown("## 📊 Key Performance Indicators")
    
    # Enhanced KPIs
    total_tx = len(filtered)
    fraud_count = int(filtered['Class'].sum())
    fraud_pct = (fraud_count / max(total_tx, 1)) * 100.0
    total_fraud_amount = float(filtered.loc[filtered['Class'] == 1, 'Amount'].sum())
    avg_fraud_amount = float(filtered.loc[filtered['Class'] == 1, 'Amount'].mean()) if fraud_count > 0 else 0
    
    # KPI cards with better styling
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Transactions</h3>
            <h2>{total_tx:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🚨 Fraud Rate</h3>
            <h2>{fraud_pct:.3f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔍 Fraud Count</h3>
            <h2>{fraud_count:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Total Fraud Amount</h3>
            <h2>${total_fraud_amount:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Fraud Amount", f"${avg_fraud_amount:.2f}")
        st.metric("Non-Fraud Count", f"{total_tx - fraud_count:,}")
    
    with col2:
        st.metric("Fraud Amount Ratio", f"{(total_fraud_amount / filtered['Amount'].sum() * 100):.2f}%")
        st.metric("Data Coverage", f"{(len(filtered) / len(df) * 100):.1f}%")
    
    st.markdown("---")
    
    # Enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Improved pie chart
        pie_df = filtered['Class'].map({0:'Non-Fraud', 1:'Fraud'}).value_counts().reset_index()
        pie_df.columns = ['Type', 'Count']
        
        fig_pie = px.pie(
            pie_df, 
            values='Count', 
            names='Type', 
            color='Type',
            color_discrete_map={'Fraud':'#FF6B6B','Non-Fraud':'#4ECDC4'},
            title='Transaction Distribution'
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Enhanced bar chart
        amt_df = filtered.groupby('Class')['Amount'].agg(['sum', 'mean', 'count']).reset_index()
        amt_df['Type'] = amt_df['Class'].map({0:'Non-Fraud', 1:'Fraud'})
        
        fig_bar = px.bar(
            amt_df, 
            x='Type', 
            y='sum',
            color='Type',
            color_discrete_map={'Fraud':'#FF6B6B','Non-Fraud':'#4ECDC4'},
            title='Total Amount by Transaction Type',
            labels={'sum': 'Total Amount ($)'}
        )
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

with trends:
    st.markdown("## 📈 Temporal Trends & Patterns")
    
    # Time-based analysis
    time_agg = filtered.copy()
    time_agg['Minute'] = (time_agg['Time'] // 60).astype(int) if 'Time' in time_agg.columns else np.random.randint(0, 60, len(time_agg))
    time_agg['Hour_Group'] = pd.cut(time_agg['Hour'], bins=6, labels=['00-04', '04-08', '08-12', '12-16', '16-20', '20-24'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud over time with better aggregation
        fraud_over_time = time_agg.groupby('Minute')['Class'].sum().reset_index()
        
        fig_line = px.line(
            fraud_over_time, 
            x='Minute', 
            y='Class', 
            title='🚨 Fraud Incidents Over Time',
            labels={'Class': 'Fraud Count', 'Minute': 'Time (Minutes)'}
        )
        fig_line.update_traces(line_color='#FF6B6B', line_width=3)
        fig_line.update_layout(height=400)
        st.plotly_chart(fig_line, use_container_width=True)
    
    with col2:
        # Transaction amount trends
        amount_over_time = time_agg.groupby('Minute')['Amount'].sum().reset_index()
        
        fig_amt_line = px.line(
            amount_over_time, 
            x='Minute', 
            y='Amount', 
            title='💰 Transaction Amount Over Time',
            labels={'Amount': 'Total Amount ($)', 'Minute': 'Time (Minutes)'}
        )
        fig_amt_line.update_traces(line_color='#4ECDC4', line_width=3)
        fig_amt_line.update_layout(height=400)
        st.plotly_chart(fig_amt_line, use_container_width=True)
    
    # Enhanced heatmap
    st.markdown("### 🕐 Fraud Patterns by Hour & Day")
    time_agg['Day'] = (time_agg['Time'] // (60*60*24)).astype(int) if 'Time' in time_agg.columns else np.random.randint(0, 7, len(time_agg))
    heat = time_agg.groupby(['Day','Hour'])['Class'].sum().reset_index()
    heat_pivot = heat.pivot(index='Hour', columns='Day', values='Class').fillna(0)
    
    fig_heat = px.imshow(
        heat_pivot, 
        color_continuous_scale='Reds',
        origin='lower',
        aspect='auto',
        title='Fraud Heatmap: Hour vs Day',
        labels={'x': 'Day', 'y': 'Hour', 'color': 'Fraud Count'}
    )
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, use_container_width=True)

with features:
    st.markdown("## 🔍 Feature Analysis & Insights")
    
    # Find feature columns
    feat_cols = [c for c in df.columns if c.startswith('V') or c in ['Amount', 'Time']]
    
    if not feat_cols:
        st.error("❌ No feature columns found in the dataset!")
        st.stop()
    
    # Use a sample of data for feature analysis
    sample_size_features = min(10000, len(filtered))
    if len(filtered) > sample_size_features:
        sample_data = filtered.sample(n=sample_size_features, random_state=42)
    else:
        sample_data = filtered
    
    X = sample_data[feat_cols].fillna(0)
    y = sample_data['Class']
    
    if len(sample_data) > 0 and y.nunique() > 1:
        st.markdown("### 🎯 Feature Importance Analysis")
        
        try:
            with st.spinner('Training Random Forest model for feature importance...'):
                # Random Forest with optimized parameters
                rf_model = RandomForestClassifier(
                    n_estimators=50, 
                    random_state=42, 
                    n_jobs=-1,
                    max_depth=10,
                    min_samples_split=50
                )
                rf_model.fit(X, y)
                rf_importances = rf_model.feature_importances_
                
                # Create importance dataframe
                imp_df = pd.DataFrame({
                    'Feature': feat_cols,
                    'Importance': rf_importances
                })
                imp_df = imp_df.sort_values('Importance', ascending=False).head(20)
                
                # Feature importance plot
                fig_imp = px.bar(
                    imp_df, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Viridis',
                    title='Top 20 Most Important Features'
                )
                fig_imp.update_layout(height=600)
                st.plotly_chart(fig_imp, use_container_width=True)
                
                # Show top features table
                st.markdown("### 📊 Top Feature Details")
                st.dataframe(imp_df.round(6))
                
        except Exception as e:
            st.error(f"❌ Error in feature importance analysis: {str(e)}")
    
    else:
        st.error(f"⚠️ Not enough class variety after filtering to compute feature importances.")
    
    # Feature distributions
    st.markdown("### 📊 Feature Distributions")
    
    # Select key features for visualization
    key_feats = []
    for feat in ['V2', 'V3', 'V4', 'V14', 'Amount']:
        if feat in feat_cols:
            key_feats.append(feat)
    
    if not key_feats:
        key_feats = feat_cols[:5]
    
    for feat in key_feats:
        if feat in sample_data.columns:
            try:
                fig = px.histogram(
                    sample_data, 
                    x=feat, 
                    color=sample_data['Class'].map({0:'Non-Fraud',1:'Fraud'}),
                    nbins=30, 
                    barmode='overlay', 
                    opacity=0.7,
                    color_discrete_map={'Fraud':'#FF6B6B','Non-Fraud':'#4ECDC4'},
                    title=f'Distribution of {feat}'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not plot distribution for {feat}")

with performance:
    st.markdown("## 🎯 Model Performance Analysis")
    
    # Use a sample of the full dataset for performance analysis
    sample_size_perf = min(20000, len(df))
    if len(df) > sample_size_perf:
        perf_data = df.sample(n=sample_size_perf, random_state=42)
    else:
        perf_data = df
    
    feat_cols = [c for c in perf_data.columns if c.startswith('V') or c in ['Amount']]
    
    if not feat_cols:
        st.error("❌ No feature columns found for performance analysis!")
        st.stop()
    
    X = perf_data[feat_cols].fillna(0)
    y = perf_data['Class']
    
    if len(perf_data) > 0 and y.nunique() > 1:
        try:
            with st.spinner('Training models and computing metrics...'):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                
                # Train Random Forest
                rf_model = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42, 
                    n_jobs=-1,
                    max_depth=15,
                    min_samples_split=50
                )
                rf_model.fit(X_train, y_train)
                
                # Predictions
                y_proba = rf_model.predict_proba(X_test)[:,1]
                y_pred = (y_proba >= 0.5).astype(int)
                
                # Metrics
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precision", f"{precision:.3f}")
                with col2:
                    st.metric("Recall", f"{recall:.3f}")
                with col3:
                    st.metric("F1-Score", f"{f1:.3f}")
                
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, 
                    mode='lines', 
                    name=f'ROC (AUC: {roc_auc:.3f})', 
                    line=dict(color='#2563EB', width=3)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0,1], y=[0,1], 
                    mode='lines', 
                    name='Random', 
                    line=dict(dash='dash', color='#9CA3AF')
                ))
                fig_roc.update_layout(
                    title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=500
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                
                # Precision-Recall Curve
                prec, rec, _ = precision_recall_curve(y_test, y_proba)
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=rec, y=prec, 
                    mode='lines', 
                    name='Precision-Recall', 
                    line=dict(color='#D324EB', width=3)
                ))
                fig_pr.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    height=500
                )
                st.plotly_chart(fig_pr, use_container_width=True)
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(
                    cm, 
                    text_auto=True, 
                    color_continuous_scale='Blues', 
                    aspect='equal', 
                    origin='lower',
                    title='Confusion Matrix'
                )
                fig_cm.update_xaxes(title_text='Predicted')
                fig_cm.update_yaxes(title_text='Actual')
                fig_cm.update_layout(height=500)
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Classification report
                st.markdown("### 📋 Detailed Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4))
                
        except Exception as e:
            st.error(f"❌ Error in performance analysis: {str(e)}")
    else:
        st.error("❌ Not enough data for performance analysis")

with drill:
    st.markdown("## 🔎 Detailed Transaction Analysis")
    
    # Enhanced search and filtering
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_id = st.text_input('🔍 Search by Transaction ID', placeholder='Enter numeric ID...')
        search_amount = st.number_input('💰 Search by Amount', min_value=0.0, value=0.0, step=0.01)
    
    with col2:
        sort_by = st.selectbox('📊 Sort by', ['Time', 'Amount', 'Class'])
        sort_order = st.selectbox('🔄 Order', ['Ascending', 'Descending'])
    
    # Apply filters
    show_df = filtered.copy()
    
    if fraud_only:
        show_df = show_df[show_df['Class'] == 1]
        st.info(f"🚨 Showing {len(show_df):,} fraud transactions only")
    
    if search_id:
        try:
            sid = int(search_id)
            show_df = show_df[show_df['TransactionID'] == sid]
            if len(show_df) == 0:
                st.warning(f"⚠️ No transaction found with ID: {sid}")
        except ValueError:
            st.warning('⚠️ Please enter a numeric Transaction ID')
    
    if search_amount > 0:
        show_df = show_df[show_df['Amount'] == search_amount]
    
    # Sort data
    ascending = sort_order == 'Ascending'
    try:
        show_df = show_df.sort_values(sort_by, ascending=ascending)
    except:
        st.warning(f"⚠️ Could not sort by {sort_by}, using default sorting")
        show_df = show_df.sort_values('TransactionID', ascending=True)
    
    # Display results
    st.markdown(f"### 📊 Results ({len(show_df):,} transactions)")
    
    if len(show_df) > 0:
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Amount", f"${show_df['Amount'].sum():,.2f}")
        with col2:
            st.metric("Average Amount", f"${show_df['Amount'].mean():,.2f}")
        with col3:
            st.metric("Fraud Count", f"{show_df['Class'].sum():,}")
        
        # Data table with pagination
        st.dataframe(
            show_df.head(1000),
            use_container_width=True,
            height=400
        )
        
        if len(show_df) > 1000:
            st.info(f"📊 Showing first 1000 of {len(show_df):,} transactions. Use filters to narrow down results.")
    else:
        st.info("📭 No transactions match the current filters.")

with insights:
    st.markdown("## 💡 Advanced Insights & Recommendations")
    
    if len(filtered) > 0:
        try:
            # Risk scoring
            st.markdown("### 🚨 Risk Assessment")
            
            # Calculate risk factors
            risk_factors = {}
            
            # Amount-based risk
            high_amount_threshold = filtered['Amount'].quantile(0.95)
            high_amount_fraud_rate = filtered[filtered['Amount'] > high_amount_threshold]['Class'].mean()
            risk_factors['High Amount'] = high_amount_fraud_rate
            
            # Time-based risk
            night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
            night_fraud_rate = filtered[filtered['Hour'].isin(night_hours)]['Class'].mean()
            risk_factors['Night Hours'] = night_fraud_rate
            
            # Overall fraud rate
            overall_fraud_rate = filtered['Class'].mean()
            risk_factors['Overall'] = overall_fraud_rate
            
            # Display risk factors
            risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Risk Factor', 'Fraud Rate'])
            risk_df['Risk Level'] = pd.cut(
                risk_df['Fraud Rate'], 
                bins=[0, 0.01, 0.05, 0.1, 1], 
                labels=['Low', 'Medium', 'High', 'Critical']
            )
            
            st.write("**Risk factors calculated:**")
            st.dataframe(risk_df.round(4))
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            if high_amount_fraud_rate > overall_fraud_rate:
                st.warning("⚠️ **High-amount transactions** show elevated fraud risk. Consider additional verification for amounts above ${:.2f}.".format(high_amount_threshold))
            
            if night_fraud_rate > overall_fraud_rate:
                st.warning("⚠️ **Night-time transactions** show higher fraud rates. Consider enhanced monitoring during off-hours.")
            
            # Anomaly detection
            st.markdown("### 🔍 Anomaly Detection")
            
            # Statistical outliers
            amount_mean = filtered['Amount'].mean()
            amount_std = filtered['Amount'].std()
            outliers = filtered[abs(filtered['Amount'] - amount_mean) > 3 * amount_std]
            
            if len(outliers) > 0:
                st.info(f"📊 Found {len(outliers):,} statistical outliers (3+ standard deviations from mean)")
                
                # Outlier analysis
                outlier_fraud_rate = outliers['Class'].mean()
                
                if outlier_fraud_rate > overall_fraud_rate:
                    st.warning(f"🚨 Outliers show {outlier_fraud_rate:.1%} fraud rate vs {overall_fraud_rate:.1%} overall")
                else:
                    st.success(f"✅ Outliers show {outlier_fraud_rate:.1%} fraud rate vs {overall_fraud_rate:.1%} overall")
            
            # Performance trends
            st.markdown("### 📈 Performance Trends")
            
            # Hourly fraud distribution
            hourly_fraud = filtered.groupby('Hour')['Class'].sum().reset_index()
            hourly_total = filtered.groupby('Hour').size().reset_index(name='Total')
            hourly_analysis = hourly_fraud.merge(hourly_total, on='Hour')
            hourly_analysis['Fraud_Rate'] = hourly_analysis['Class'] / hourly_analysis['Total']
            
            fig_hourly = px.line(
                hourly_analysis,
                x='Hour',
                y='Fraud_Rate',
                title='Fraud Rate by Hour of Day',
                labels={'Fraud_Rate': 'Fraud Rate', 'Hour': 'Hour of Day'}
            )
            fig_hourly.update_layout(height=400)
            st.plotly_chart(fig_hourly, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error in insights analysis: {str(e)}")
    else:
        st.error("❌ No data available for insights")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🕵️ Credit Card Fraud Detection Dashboard | Built with Streamlit & Plotly</p>
    <p>For security analysis and fraud prevention</p>
</div>
""", unsafe_allow_html=True)
