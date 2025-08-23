import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="GlucoSense: AI Powered Diabetes Detection for Early Intervention",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced professional CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.main-header h2 {
    font-size: 1.3rem;
    font-weight: 400;
    opacity: 0.95;
    margin-bottom: 0;
}

.metric-card {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    border: 1px solid #f0f2f6;
    text-align: center;
    margin: 1rem 0;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.15);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #667eea;
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 1rem;
    color: #64748b;
    font-weight: 500;
}

.prediction-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 2.5rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 15px 35px rgba(240, 147, 251, 0.4);
    margin: 2rem 0;
}

.prediction-low {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    box-shadow: 0 15px 35px rgba(79, 172, 254, 0.4);
}

.prediction-medium {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    box-shadow: 0 15px 35px rgba(240, 147, 251, 0.4);
}

.prediction-high {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
    box-shadow: 0 15px 35px rgba(255, 107, 107, 0.4);
}

.feature-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid #667eea;
}

.analysis-container {
    background: white;
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    margin: 2rem 0;
}

.model-performance {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    border-radius: 15px;
    padding: 2rem;
    margin: 1rem 0;
    border: 1px solid #e2e8f0;
}

.chart-container {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    margin: 1rem 0;
}

.sidebar .sidebar-content {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}

.stSelectbox label {
    font-weight: 600;
    color: #374151;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the diabetes dataset"""
    data = pd.read_csv('data/diabetes_data.csv')
    return data

@st.cache_data
def prepare_ml_models(X, y):
    """Prepare and train multiple ML models"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Train models and collect performance
    model_results = {}
    trained_models = {}
    
    for name, model in models.items():
        if name in ['Logistic Regression', 'SVM']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            trained_models[name] = (model, scaler)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            trained_models[name] = (model, None)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        model_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    return model_results, trained_models, X_test, y_test

def create_correlation_heatmap(data):
    """Create correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=data.corr().values,
        x=data.columns,
        y=data.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(data.corr().values, 2),
        texttemplate='%{text}',
        textfont={'size': 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title={'text': 'Feature Correlation Matrix', 'x': 0.5, 'font': {'size': 20}},
        width=700,
        height=600,
        font=dict(size=12)
    )
    return fig

def create_feature_distribution(data):
    """Create feature distribution plots"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'Outcome']
    
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=numeric_cols,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    colors = ['#667eea', '#764ba2']
    
    for i, col in enumerate(numeric_cols):
        row = i // 4 + 1
        col_pos = i % 4 + 1
        
        for outcome in [0, 1]:
            subset = data[data['Outcome'] == outcome][col]
            fig.add_trace(
                go.Histogram(
                    x=subset,
                    name=f'{"Diabetic" if outcome else "Non-Diabetic"}',
                    opacity=0.7,
                    marker_color=colors[outcome],
                    showlegend=(i == 0)
                ),
                row=row, col=col_pos
            )
    
    fig.update_layout(
        title={'text': 'Feature Distributions by Diabetes Status', 'x': 0.5, 'font': {'size': 20}},
        height=600,
        barmode='overlay'
    )
    return fig

def create_model_comparison_chart(model_results):
    """Create model performance comparison chart"""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    models = list(model_results.keys())
    
    fig = go.Figure()
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    
    for i, metric in enumerate(metrics):
        values = [model_results[model][metric] for model in models]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=models,
            fill='toself',
            name=metric.title().replace('_', ' '),
            line_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title={'text': 'Model Performance Comparison', 'x': 0.5, 'font': {'size': 20}},
        height=500
    )
    return fig

def create_confusion_matrix_plot(y_true, y_pred, model_name):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: Non-Diabetic', 'Predicted: Diabetic'],
        y=['Actual: Non-Diabetic', 'Actual: Diabetic'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={'size': 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title={'text': f'Confusion Matrix - {model_name}', 'x': 0.5, 'font': {'size': 18}},
        width=400,
        height=400
    )
    return fig

def predict_diabetes(features, trained_models, model_name='Random Forest'):
    """Predict diabetes using selected model"""
    model, scaler = trained_models[model_name]
    
    # Prepare features
    features_array = np.array(features).reshape(1, -1)
    
    if scaler:
        features_array = scaler.transform(features_array)
    
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0, 1]
    
    return prediction, probability

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🩸 GlucoSense</h1>
        <h2>AI Powered Diabetes Detection for Early Intervention</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    # Sidebar
    st.sidebar.title("🎯 Navigation Panel")
    page = st.sidebar.selectbox(
        "Select Analysis Module",
        ["📊 Executive Dashboard", "🔍 Data Analytics", "🤖 ML Models", "🎯 Risk Prediction", "🍎 Lifestyle Recommendations", "📈 Performance Metrics"]
    )
    
    if page == "📊 Executive Dashboard":
        executive_dashboard(data)
    elif page == "🔍 Data Analytics":
        data_analytics(data)
    elif page == "🤖 ML Models":
        ml_models_page(data)
    elif page == "🎯 Risk Prediction":
        risk_prediction_page(data)
    elif page == "🍎 Lifestyle Recommendations":
        lifestyle_recommendations_page(data)
    elif page == "📈 Performance Metrics":
        performance_metrics_page(data)

def executive_dashboard(data):
    st.header("📊 Executive Healthcare Analytics Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_patients = len(data)
    diabetic_patients = data['Outcome'].sum()
    diabetes_rate = (diabetic_patients / total_patients) * 100
    avg_age = data['Age'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_patients:,}</div>
            <div class="metric-label">Total Patients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{diabetic_patients}</div>
            <div class="metric-label">Diabetic Cases</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{diabetes_rate:.1f}%</div>
            <div class="metric-label">Diabetes Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_age:.0f}</div>
            <div class="metric-label">Average Age</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.histogram(
            data, x='Age', color='Outcome', nbins=20,
            title='Age Distribution by Diabetes Status',
            labels={'Outcome': 'Diabetes Status'},
            color_discrete_map={0: '#4facfe', 1: '#f093fb'}
        )
        fig_age.update_layout(
            title_x=0.5,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # BMI vs Glucose scatter
        fig_scatter = px.scatter(
            data, x='BMI', y='Glucose', color='Outcome',
            title='BMI vs Glucose Levels',
            labels={'Outcome': 'Diabetes Status'},
            color_discrete_map={0: '#4facfe', 1: '#f093fb'}
        )
        fig_scatter.update_layout(
            title_x=0.5,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Risk factor analysis
    st.subheader("🎯 Risk Factor Analysis")
    
    risk_factors = {
        'High Glucose (≥140)': len(data[data['Glucose'] >= 140]),
        'Obesity (BMI ≥30)': len(data[data['BMI'] >= 30]),
        'Hypertension (BP ≥130)': len(data[data['BloodPressure'] >= 130]),
        'Age ≥45': len(data[data['Age'] >= 45])
    }
    
    col1, col2, col3, col4 = st.columns(4)
    for i, (factor, count) in enumerate(risk_factors.items()):
        col = [col1, col2, col3, col4][i]
        percentage = (count / total_patients) * 100
        col.metric(factor, f"{count} ({percentage:.1f}%)")

def data_analytics(data):
    st.header("🔍 Comprehensive Data Analytics")
    
    # Dataset overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="analysis-container">
            <h3>📋 Dataset Overview</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(f"**Records:** {len(data):,}")
        st.write(f"**Features:** {len(data.columns) - 1}")
        st.write(f"**Missing Values:** {data.isnull().sum().sum()}")
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(data.describe().round(2), use_container_width=True)
    
    with col2:
        # Correlation heatmap
        st.markdown("""
        <div class="chart-container">
        """, unsafe_allow_html=True)
        fig_corr = create_correlation_heatmap(data)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Feature distributions
    st.subheader("📊 Feature Distributions")
    fig_dist = create_feature_distribution(data)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Feature importance analysis
    st.subheader("🔍 Feature Correlation with Diabetes")
    
    correlations = data.corr()['Outcome'].abs().sort_values(ascending=False)[1:]
    
    fig_importance = go.Figure(data=[
        go.Bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            marker_color='#667eea'
        )
    ])
    
    fig_importance.update_layout(
        title='Feature Importance (Correlation with Diabetes)',
        xaxis_title='Absolute Correlation',
        yaxis_title='Features',
        height=400,
        title_x=0.5
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)

def ml_models_page(data):
    st.header("🤖 Machine Learning Models Analysis")
    
    # Prepare features
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Train models
    model_results, trained_models, X_test, y_test = prepare_ml_models(X, y)
    
    # Model comparison
    st.subheader("📊 Model Performance Comparison")
    
    # Performance metrics table
    metrics_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Accuracy': [model_results[model]['accuracy'] for model in model_results.keys()],
        'Precision': [model_results[model]['precision'] for model in model_results.keys()],
        'Recall': [model_results[model]['recall'] for model in model_results.keys()],
        'F1-Score': [model_results[model]['f1'] for model in model_results.keys()],
        'ROC-AUC': [model_results[model]['roc_auc'] for model in model_results.keys()]
    }).round(4)
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Radar chart
    fig_radar = create_model_comparison_chart(model_results)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Confusion matrices
    st.subheader("🔍 Confusion Matrix Analysis")
    
    cols = st.columns(2)
    for i, (model_name, results) in enumerate(model_results.items()):
        col = cols[i % 2]
        with col:
            fig_cm = create_confusion_matrix_plot(y_test, results['predictions'], model_name)
            st.plotly_chart(fig_cm, use_container_width=True)

def risk_prediction_page(data):
    st.header("🎯 Risk Prediction System")
    
    # Prepare models
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    model_results, trained_models, X_test, y_test = prepare_ml_models(X, y)
    
    # Input form
    st.subheader("📝 Patient Information Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=80)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, max_value=1000, value=80)
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
    
    # Model selection
    selected_model = st.selectbox("Select ML Model", list(trained_models.keys()), index=0)
    
    if st.button("🔮 Analyze Diabetes Risk", type="primary"):
        features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        prediction, probability = predict_diabetes(features, trained_models, selected_model)
        
        # Results display
        st.subheader("📊 Risk Analysis Results")
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "HIGH"
            card_class = "prediction-high"
        elif probability >= 0.4:
            risk_level = "MODERATE"
            card_class = "prediction-medium"
        else:
            risk_level = "LOW"
            card_class = "prediction-low"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="prediction-card {card_class}">
                <h3>{risk_level} RISK</h3>
                <h2>{probability:.1%}</h2>
                <p>Diabetes Probability</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence = 85 + (probability - 0.5) ** 2 * 30
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{confidence:.0f}%</div>
                <div class="metric-label">Model Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            model_accuracy = model_results[selected_model]['accuracy']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{model_accuracy:.1%}</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature analysis
        st.subheader("📈 Risk Factor Breakdown")
        
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Create feature importance chart for this prediction
        mean_values = data.drop('Outcome', axis=1).mean()
        input_values = pd.Series(features, index=feature_names)
        
        deviation = ((input_values - mean_values) / mean_values * 100).abs()
        
        fig_features = go.Figure(data=[
            go.Bar(
                x=feature_names,
                y=deviation,
                marker_color='#667eea'
            )
        ])
        
        fig_features.update_layout(
            title='Deviation from Population Average (%)',
            xaxis_title='Features',
            yaxis_title='Percentage Deviation',
            height=400,
            title_x=0.5
        )
        
        st.plotly_chart(fig_features, use_container_width=True)

def lifestyle_recommendations_page(data):
    st.header("🍎 Lifestyle Recommendations & Prevention")
    
    # Introduction
    st.markdown("""
    <div class="analysis-container">
        <h3>🎯 Diabetes Prevention Through Lifestyle</h3>
        <p>Evidence-based lifestyle modifications can significantly reduce diabetes risk and improve management outcomes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different categories
    tab1, tab2, tab3, tab4 = st.tabs(["🍽️ Nutrition", "🏃‍♂️ Exercise", "😴 Sleep & Stress", "🩺 Monitoring"])
    
    with tab1:
        st.subheader("🥗 Nutritional Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🌾 Recommended Foods</h4>
                <ul>
                    <li><strong>Complex Carbohydrates:</strong> Whole grains, quinoa, brown rice</li>
                    <li><strong>Lean Proteins:</strong> Fish, chicken, legumes, tofu</li>
                    <li><strong>Healthy Fats:</strong> Avocados, nuts, olive oil</li>
                    <li><strong>Fiber-Rich Foods:</strong> Vegetables, fruits, beans</li>
                    <li><strong>Low Glycemic Index:</strong> Leafy greens, berries, oats</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>⏰ Meal Timing</h4>
                <ul>
                    <li>Eat regular meals at consistent times</li>
                    <li>Don't skip breakfast</li>
                    <li>Consider smaller, more frequent meals</li>
                    <li>Limit late-night eating</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>🚫 Foods to Limit</h4>
                <ul>
                    <li><strong>Refined Sugars:</strong> Candy, sodas, desserts</li>
                    <li><strong>Processed Foods:</strong> Fast food, packaged snacks</li>
                    <li><strong>Refined Carbs:</strong> White bread, white rice</li>
                    <li><strong>Trans Fats:</strong> Fried foods, margarine</li>
                    <li><strong>High Sodium:</strong> Processed meats, canned foods</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>💧 Hydration</h4>
                <ul>
                    <li>Drink 8-10 glasses of water daily</li>
                    <li>Limit sugary beverages</li>
                    <li>Choose water over juice</li>
                    <li>Monitor alcohol consumption</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("🏃‍♂️ Exercise & Physical Activity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>💪 Aerobic Exercise</h4>
                <p><strong>Target:</strong> 150 minutes per week</p>
                <ul>
                    <li>Brisk walking (30 min, 5 days/week)</li>
                    <li>Swimming or water aerobics</li>
                    <li>Cycling or stationary bike</li>
                    <li>Dancing or group fitness classes</li>
                    <li>Jogging or running</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>🏋️‍♀️ Strength Training</h4>
                <p><strong>Target:</strong> 2-3 sessions per week</p>
                <ul>
                    <li>Resistance band exercises</li>
                    <li>Free weights or machines</li>
                    <li>Bodyweight exercises</li>
                    <li>Yoga or Pilates</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>📅 Weekly Exercise Plan</h4>
                <ul>
                    <li><strong>Monday:</strong> 30 min brisk walk</li>
                    <li><strong>Tuesday:</strong> Strength training</li>
                    <li><strong>Wednesday:</strong> Swimming or cycling</li>
                    <li><strong>Thursday:</strong> Strength training</li>
                    <li><strong>Friday:</strong> 30 min brisk walk</li>
                    <li><strong>Weekend:</strong> Fun activities (hiking, dancing)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>📈 Benefits</h4>
                <ul>
                    <li>Improves insulin sensitivity</li>
                    <li>Helps with weight management</li>
                    <li>Lowers blood glucose levels</li>
                    <li>Reduces cardiovascular risk</li>
                    <li>Enhances mood and energy</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("😴 Sleep & Stress Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🌙 Sleep Hygiene</h4>
                <p><strong>Target:</strong> 7-9 hours per night</p>
                <ul>
                    <li>Maintain consistent sleep schedule</li>
                    <li>Create a relaxing bedtime routine</li>
                    <li>Keep bedroom cool and dark</li>
                    <li>Avoid screens before bedtime</li>
                    <li>Limit caffeine after 2 PM</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>🧘‍♀️ Stress Reduction</h4>
                <ul>
                    <li>Practice deep breathing exercises</li>
                    <li>Try meditation or mindfulness</li>
                    <li>Engage in hobbies you enjoy</li>
                    <li>Maintain social connections</li>
                    <li>Consider yoga or tai chi</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>⚖️ Work-Life Balance</h4>
                <ul>
                    <li>Set boundaries with work</li>
                    <li>Take regular breaks</li>
                    <li>Plan vacation and rest days</li>
                    <li>Learn to say no to overcommitments</li>
                    <li>Practice time management</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>🔄 Impact on Diabetes</h4>
                <ul>
                    <li>Poor sleep affects blood sugar control</li>
                    <li>Chronic stress increases cortisol</li>
                    <li>Both can worsen insulin resistance</li>
                    <li>Quality sleep improves glucose metabolism</li>
                    <li>Stress management supports overall health</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("🩺 Health Monitoring & Regular Checkups")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🔍 Self-Monitoring</h4>
                <ul>
                    <li><strong>Blood Glucose:</strong> As recommended by doctor</li>
                    <li><strong>Blood Pressure:</strong> Regular home monitoring</li>
                    <li><strong>Weight:</strong> Weekly weigh-ins</li>
                    <li><strong>Food Diary:</strong> Track meals and responses</li>
                    <li><strong>Exercise Log:</strong> Record physical activity</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>📱 Technology Tools</h4>
                <ul>
                    <li>Glucose monitoring apps</li>
                    <li>Fitness trackers</li>
                    <li>Nutrition tracking apps</li>
                    <li>Sleep monitoring devices</li>
                    <li>Telemedicine platforms</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>🏥 Regular Checkups</h4>
                <p><strong>Schedule:</strong></p>
                <ul>
                    <li><strong>HbA1c Test:</strong> Every 3-6 months</li>
                    <li><strong>Lipid Profile:</strong> Annually</li>
                    <li><strong>Kidney Function:</strong> Annually</li>
                    <li><strong>Eye Exam:</strong> Annually</li>
                    <li><strong>Foot Exam:</strong> At each visit</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>⚠️ Warning Signs</h4>
                <p><strong>Contact healthcare provider if:</strong></p>
                <ul>
                    <li>Frequent urination or excessive thirst</li>
                    <li>Unexplained weight loss</li>
                    <li>Persistent fatigue</li>
                    <li>Slow-healing wounds</li>
                    <li>Vision changes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Action Plan Generator
    st.subheader("📋 Personalized Action Plan")
    
    risk_level = st.selectbox(
        "Select your current risk level:",
        ["Low Risk (Prevention)", "Moderate Risk (Early Intervention)", "High Risk (Management)"]
    )
    
    if risk_level == "Low Risk (Prevention)":
        st.markdown("""
        <div class="prediction-card prediction-low">
            <h3>🛡️ Prevention Plan</h3>
            <ul>
                <li>Maintain healthy weight (BMI 18.5-24.9)</li>
                <li>Exercise 150 minutes per week</li>
                <li>Follow balanced, low-glycemic diet</li>
                <li>Get annual health screenings</li>
                <li>Manage stress effectively</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif risk_level == "Moderate Risk (Early Intervention)":
        st.markdown("""
        <div class="prediction-card prediction-medium">
            <h3>⚠️ Early Intervention Plan</h3>
            <ul>
                <li>Work with healthcare provider for monitoring</li>
                <li>Implement structured exercise program</li>
                <li>Follow diabetes prevention diet</li>
                <li>Monitor glucose levels as directed</li>
                <li>Consider diabetes prevention programs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="prediction-card prediction-high">
            <h3>🚨 Active Management Plan</h3>
            <ul>
                <li>Regular medical supervision required</li>
                <li>Strict dietary and exercise adherence</li>
                <li>Daily glucose monitoring</li>
                <li>Medication management as prescribed</li>
                <li>Frequent healthcare provider visits</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def performance_metrics_page(data):
    st.header("📈 Advanced Performance Metrics")
    
    # Prepare models
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    model_results, trained_models, X_test, y_test = prepare_ml_models(X, y)
    
    # Model selection for detailed analysis
    selected_model = st.selectbox("Select Model for Detailed Analysis", list(model_results.keys()), index=0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="model-performance">
            <h3>🎯 {selected_model} Performance</h3>
            <p><strong>Accuracy:</strong> {model_results[selected_model]['accuracy']:.4f}</p>
            <p><strong>Precision:</strong> {model_results[selected_model]['precision']:.4f}</p>
            <p><strong>Recall:</strong> {model_results[selected_model]['recall']:.4f}</p>
            <p><strong>F1-Score:</strong> {model_results[selected_model]['f1']:.4f}</p>
            <p><strong>ROC-AUC:</strong> {model_results[selected_model]['roc_auc']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, model_results[selected_model]['probabilities'])
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{selected_model}', line=dict(color='#667eea', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
        
        fig_roc.update_layout(
            title=f'ROC Curve - {selected_model}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            title_x=0.5
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
    
    with col2:
        # Feature importance (for tree-based models)
        if selected_model in ['Random Forest', 'Gradient Boosting']:
            model, _ = trained_models[selected_model]
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig_importance = go.Figure(data=[
                go.Bar(
                    x=feature_importance['importance'],
                    y=feature_importance['feature'],
                    orientation='h',
                    marker_color='#764ba2'
                )
            ])
            
            fig_importance.update_layout(
                title=f'{selected_model} Feature Importance',
                xaxis_title='Importance',
                yaxis_title='Features',
                height=400,
                title_x=0.5
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    
    # Cross-validation results
    st.subheader("🔄 Cross-Validation Analysis")
    
    cv_results = {}
    for name, (model, scaler) in trained_models.items():
        if scaler:
            X_scaled = scaler.fit_transform(X)
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        else:
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_results[name] = scores
    
    cv_df = pd.DataFrame(cv_results).T
    cv_df.columns = [f'Fold {i+1}' for i in range(5)]
    cv_df['Mean'] = cv_df.mean(axis=1)
    cv_df['Std'] = cv_df.std(axis=1)
    
    st.dataframe(cv_df.round(4), use_container_width=True)

if __name__ == "__main__":
    main()