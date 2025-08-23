# GlucoSense: AI Powered Diabetes Detection for Early Intervention

## MCA Final Year Project

An enterprise-grade AI system for diabetes detection using advanced machine learning algorithms with professional healthcare UI/UX built on Streamlit.

## Features

- **Complete ML Pipeline**: 4 advanced algorithms (Random Forest, Gradient Boosting, Logistic Regression, SVM)
- **Professional UI**: Enterprise-grade interface with Inter font and gradient styling
- **Executive Dashboard**: Comprehensive healthcare analytics and metrics
- **Data Analytics**: Correlation heatmaps, feature distributions, statistical analysis
- **ML Models Comparison**: Performance metrics, confusion matrices, ROC curves
- **Risk Prediction**: Real-time diabetes risk assessment with model selection
- **Lifestyle Recommendations**: Nutrition, exercise, sleep, and monitoring guidance
- **Performance Metrics**: Cross-validation analysis and feature importance

## Installation & Setup

### Prerequisites
- Python 3.11+
- pip package manager

### Install Dependencies
```bash
pip install streamlit pandas numpy plotly scikit-learn seaborn matplotlib
```

### Run the Application
```bash
streamlit run app.py
```

The application will start on `http://localhost:8501`

## Project Structure
```
glucosense/
├── app.py                 # Main Streamlit application
├── data/
│   └── diabetes_data.csv  # Dataset (768 patient records)
├── README.md              # This file
├── replit.md              # Project documentation
├── pyproject.toml         # Python dependencies
└── uv.lock               # Lock file
```

## Dataset Information
- **Source**: Diabetes health records dataset
- **Records**: 768 patients
- **Features**: 8 medical parameters (Glucose, BMI, Age, Blood Pressure, etc.)
- **Target**: Binary diabetes outcome (0: Non-diabetic, 1: Diabetic)

## Usage Guide

### Navigation
1. **Executive Dashboard**: Overview metrics and visualizations
2. **Data Analytics**: Detailed dataset analysis and correlations
3. **ML Models**: Model comparison and performance analysis
4. **Risk Prediction**: Real-time diabetes risk assessment
5. **Lifestyle Recommendations**: Evidence-based health guidance
6. **Performance Metrics**: Advanced model evaluation

### Making Predictions
1. Go to "Risk Prediction" section
2. Enter patient information (age, glucose, BMI, etc.)
3. Select ML model (Random Forest recommended)
4. Click "Analyze Diabetes Risk"
5. View risk assessment and recommendations

## Technical Architecture

- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Python with scikit-learn ML pipeline
- **Models**: Ensemble approach with cross-validation
- **Visualizations**: Plotly interactive charts
- **Data Processing**: Pandas and NumPy

## Model Performance
- **Random Forest**: ~85% accuracy
- **Gradient Boosting**: ~83% accuracy
- **Logistic Regression**: ~82% accuracy
- **SVM**: ~80% accuracy

## Academic Information
- **Project Type**: MCA Final Year Project
- **Domain**: Healthcare Machine Learning
- **Complexity**: Enterprise-level implementation
- **Suitable For**: Academic presentation and demonstration

## Development Notes
- All models train in real-time using the provided dataset
- No external APIs or internet connection required
- Fully self-contained application
- Professional healthcare-grade UI design
- Comprehensive error handling and validation

## Author
MCA Student - Advanced AI System for Healthcare Applications

## License
Educational Use Only - MCA Final Year Project
