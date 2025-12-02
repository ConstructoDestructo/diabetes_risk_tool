"""
Type 2 Diabetes Risk Assessment Tool
Built with NHANES 1999-2023 Data (50,000+ patients)
Logistic Regression Models:
- Public Screening: 81.72% AUC (no labs)
- Clinical Assessment: 97.38% AUC (with HbA1c & labs)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724 !important;
    }
    .risk-low h1, .risk-low h2, .risk-low h3, .risk-low p {
        color: #155724 !important;
    }
    .risk-moderate {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404 !important;
    }
    .risk-moderate h1, .risk-moderate h2, .risk-moderate h3, .risk-moderate p {
        color: #856404 !important;
    }
    .risk-high {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24 !important;
    }
    .risk-high h1, .risk-high h2, .risk-high h3, .risk-high p {
        color: #721c24 !important;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        color: #333 !important;
    }
    .disclaimer {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin-top: 2rem;
        color: #856404 !important;
    }
    .disclaimer h3, .disclaimer p, .disclaimer ul, .disclaimer li {
        color: #856404 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_bmi(weight_lbs, height_inches):
    """Calculate BMI from weight (lbs) and height (inches)"""
    return (weight_lbs / (height_inches ** 2)) * 703

def get_bmi_category(bmi):
    """Categorize BMI"""
    if bmi < 18.5:
        return "Underweight", "🔵"
    elif bmi < 25:
        return "Normal Weight", "🟢"
    elif bmi < 30:
        return "Overweight", "🟡"
    else:
        return "Obese", "🔴"

def predict_diabetes_risk_basic(age, sex, bmi, waist, race, family_history, hypertension, physical_activity):
    """
    PUBLIC SCREENING VERSION - Logistic Regression Weights
    Trained on 44,841 NHANES patients
    Test AUC: 0.8172
    NO LAB TESTS REQUIRED
    """
    import numpy as np
    
    log_odds = -2.967044
    factors = []
    
    # Age contribution
    if age >= 45:
        age_contrib = 1.694507 + (age - 45) * 0.025160
        log_odds += age_contrib
        if abs(age_contrib) > 0.01:
            factors.append(("Age", age_contrib, f"{age} years"))
    elif age >= 40:
        age_contrib = 1.116910
        log_odds += age_contrib
        if abs(age_contrib) > 0.01:
            factors.append(("Age", age_contrib, f"{age} years"))
    
    # BMI contribution - Obese
    if bmi >= 30:
        bmi_contrib = 0.920373 + (bmi - 30) * 0.067961
        log_odds += bmi_contrib
        factors.append(("BMI", bmi_contrib, f"{bmi:.1f} kg/m² (Obese)"))
    elif bmi >= 25:
        bmi_contrib = 0.223897 + (bmi - 25) * 0.086140
        log_odds += bmi_contrib
        if abs(bmi_contrib) > 0.01:
            factors.append(("BMI", bmi_contrib, f"{bmi:.1f} kg/m² (Overweight)"))
    
    # Waist circumference
    if waist > 0:
        if (sex == "Male" and waist > 40) or (sex == "Female" and waist > 35):
            waist_contrib = 0.236657
            log_odds += waist_contrib
            if abs(waist_contrib) > 0.01:
                factors.append(("Waist", waist_contrib, f"{waist:.1f} inches (High)"))
        elif (sex == "Male" and waist > 37) or (sex == "Female" and waist > 32):
            waist_contrib = 0.035291
            log_odds += waist_contrib
            if abs(waist_contrib) > 0.01:
                factors.append(("Waist", waist_contrib, f"{waist:.1f} inches (Elevated)"))
    
    # Family history
    if family_history == "Yes":
        fh_contrib = -0.996535
        log_odds += fh_contrib
        factors.append(("Family History", fh_contrib, "Positive"))
    
    # Hypertension
    if hypertension == "Yes":
        htn_contrib = 0.768665
        log_odds += htn_contrib
        factors.append(("Hypertension", htn_contrib, "Positive"))
    
    # Physical activity
    if physical_activity == "Sedentary":
        activity_contrib = -0.146560
        log_odds += activity_contrib
        factors.append(("Physical Activity", activity_contrib, "Sedentary"))
    elif physical_activity == "Light":
        activity_contrib = 0.057300
        log_odds += activity_contrib
        if abs(activity_contrib) > 0.01:
            factors.append(("Physical Activity", activity_contrib, "Light"))
    
    # Race/ethnicity
    if race == "Hispanic":
        race_contrib = 0.574476
        log_odds += race_contrib
        if abs(race_contrib) > 0.01:
            factors.append(("Race/Ethnicity", race_contrib, race))
    elif race == "Non-Hispanic Black":
        race_contrib = 0.367321
        log_odds += race_contrib
        if abs(race_contrib) > 0.01:
            factors.append(("Race/Ethnicity", race_contrib, race))
    elif race == "Non-Hispanic Asian":
        race_contrib = 0.931855
        log_odds += race_contrib
        if abs(race_contrib) > 0.01:
            factors.append(("Race/Ethnicity", race_contrib, race))
    elif race == "Other/Mixed":
        race_contrib = 0.429936
        log_odds += race_contrib
        if abs(race_contrib) > 0.01:
            factors.append(("Race/Ethnicity", race_contrib, race))
    
    # Convert to probability
    risk_probability = 100 / (1 + np.exp(-log_odds))
    
    # Sort factors
    factors_sorted = sorted(factors, key=lambda x: abs(x[1]), reverse=True)
    
    return risk_probability, factors_sorted

def predict_diabetes_risk_clinical(age, sex, bmi, waist, race, family_history, hypertension, 
                                   physical_activity, glucose, hba1c, triglycerides, hdl, systolic_bp):
    """
    CLINICAL ASSESSMENT VERSION - Logistic Regression Weights
    Trained on 42,605 NHANES patients with lab data
    Test AUC: 0.9738
    INCLUDES LAB TESTS: HbA1c, Glucose, Lipids
    """
    import numpy as np
    
    log_odds = -6.701756
    factors = []
    
    # ========================================================================
    # HbA1c - MOST IMPORTANT PREDICTOR
    # ========================================================================
    if hba1c > 0:
        # Continuous contribution
        hba1c_contrib = hba1c * 1.075071
        
        # Categorical thresholds
        if hba1c >= 6.5:
            hba1c_contrib += 11.101285
            log_odds += hba1c_contrib
            factors.append(("HbA1c", hba1c_contrib, f"{hba1c:.1f}% (Diabetes)"))
        elif hba1c >= 5.7:
            hba1c_contrib += 0.785740
            log_odds += hba1c_contrib
            factors.append(("HbA1c", hba1c_contrib, f"{hba1c:.1f}% (Prediabetes)"))
        else:
            log_odds += hba1c_contrib
            if abs(hba1c_contrib) > 0.1:
                factors.append(("HbA1c", hba1c_contrib, f"{hba1c:.1f}%"))
    
    # ========================================================================
    # Glucose
    # ========================================================================
    if glucose > 0:
        glucose_contrib = glucose * -0.014690
        
        if glucose >= 126:
            glucose_contrib += 9.805472
            log_odds += glucose_contrib
            factors.append(("Glucose", glucose_contrib, f"{glucose:.0f} mg/dL (Diabetes)"))
        elif glucose >= 100:
            glucose_contrib += 0.130499
            log_odds += glucose_contrib
            factors.append(("Glucose", glucose_contrib, f"{glucose:.0f} mg/dL (Elevated)"))
        else:
            log_odds += glucose_contrib
            if abs(glucose_contrib) > 0.1:
                factors.append(("Glucose", glucose_contrib, f"{glucose:.0f} mg/dL"))
    
    # ========================================================================
    # Lipids
    # ========================================================================
    if hdl > 0:
        hdl_contrib = hdl * -0.009881
        if (sex == "Male" and hdl < 40) or (sex == "Female" and hdl < 50):
            hdl_contrib += 0.060562
            log_odds += hdl_contrib
            factors.append(("HDL", hdl_contrib, f"{hdl:.0f} mg/dL (Low)"))
        else:
            log_odds += hdl_contrib
            if abs(hdl_contrib) > 0.1:
                factors.append(("HDL", hdl_contrib, f"{hdl:.0f} mg/dL"))
    
    if triglycerides > 0:
        trig_contrib = triglycerides * 0.000520
        if triglycerides >= 150:
            trig_contrib += 0.020562
            log_odds += trig_contrib
            factors.append(("Triglycerides", trig_contrib, f"{triglycerides:.0f} mg/dL (High)"))
        else:
            log_odds += trig_contrib
            if abs(trig_contrib) > 0.1:
                factors.append(("Triglycerides", trig_contrib, f"{triglycerides:.0f} mg/dL"))
    
    # ========================================================================
    # BASIC FEATURES (same as public screening)
    # ========================================================================
    
    # Age
    if age >= 45:
        age_contrib = 0.827839 + (age - 45) * 0.011897
        log_odds += age_contrib
        if abs(age_contrib) > 0.01:
            factors.append(("Age", age_contrib, f"{age} years"))
    elif age >= 40:
        age_contrib = 0.378700
        log_odds += age_contrib
        if abs(age_contrib) > 0.01:
            factors.append(("Age", age_contrib, f"{age} years"))
    
    # BMI - Obese
    if bmi >= 30:
        bmi_contrib = 0.693186 + (bmi - 30) * 0.047991
        log_odds += bmi_contrib
        factors.append(("BMI", bmi_contrib, f"{bmi:.1f} kg/m² (Obese)"))
    elif bmi >= 25:
        bmi_contrib = 0.063613 + (bmi - 25) * 0.119280
        log_odds += bmi_contrib
        if abs(bmi_contrib) > 0.01:
            factors.append(("BMI", bmi_contrib, f"{bmi:.1f} kg/m² (Overweight)"))
    
    # Waist
    if waist > 0:
        if (sex == "Male" and waist > 40) or (sex == "Female" and waist > 35):
            waist_contrib = 0.092842
            log_odds += waist_contrib
            if abs(waist_contrib) > 0.01:
                factors.append(("Waist", waist_contrib, f"{waist:.1f} inches (High)"))
        elif (sex == "Male" and waist > 37) or (sex == "Female" and waist > 32):
            waist_contrib = -0.030688
            log_odds += waist_contrib
            if abs(waist_contrib) > 0.01:
                factors.append(("Waist", waist_contrib, f"{waist:.1f} inches (Elevated)"))
    
    # Family history
    if family_history == "Yes":
        fh_contrib = -4.307152
        log_odds += fh_contrib
        factors.append(("Family History", fh_contrib, "Positive"))
    
    # Hypertension
    if hypertension == "Yes":
        htn_contrib = 0.950919
        log_odds += htn_contrib
        factors.append(("Hypertension", htn_contrib, "Positive"))
    
    # Physical activity
    if physical_activity == "Sedentary":
        activity_contrib = -0.125318
        log_odds += activity_contrib
        factors.append(("Physical Activity", activity_contrib, "Sedentary"))
    elif physical_activity == "Light":
        activity_contrib = 0.248586
        log_odds += activity_contrib
        if abs(activity_contrib) > 0.01:
            factors.append(("Physical Activity", activity_contrib, "Light"))
    
    # Race/ethnicity
    if race == "Hispanic":
        race_contrib = 0.306734
        log_odds += race_contrib
        if abs(race_contrib) > 0.01:
            factors.append(("Race/Ethnicity", race_contrib, race))
    elif race == "Non-Hispanic Black":
        race_contrib = 0.214168
        log_odds += race_contrib
        if abs(race_contrib) > 0.01:
            factors.append(("Race/Ethnicity", race_contrib, race))
    elif race == "Non-Hispanic Asian":
        race_contrib = 0.493974
        log_odds += race_contrib
        if abs(race_contrib) > 0.01:
            factors.append(("Race/Ethnicity", race_contrib, race))
    elif race == "Other/Mixed":
        race_contrib = 0.443297
        log_odds += race_contrib
        if abs(race_contrib) > 0.01:
            factors.append(("Race/Ethnicity", race_contrib, race))
    
    # Convert to probability
    risk_probability = 100 / (1 + np.exp(-log_odds))
    
    # Sort factors
    factors_sorted = sorted(factors, key=lambda x: abs(x[1]), reverse=True)
    
    return risk_probability, factors_sorted

def get_risk_category(risk_probability, has_labs=False):
    """Categorize risk level"""
    if has_labs:
        # Stricter thresholds for clinical tool
        if risk_probability < 10:
            return "LOW", "🟢", "risk-low", "#28a745"
        elif risk_probability < 40:
            return "MODERATE", "🟡", "risk-moderate", "#ffc107"
        else:
            return "HIGH", "🔴", "risk-high", "#dc3545"
    else:
        # More conservative for screening tool
        if risk_probability < 20:
            return "LOW", "🟢", "risk-low", "#28a745"
        elif risk_probability < 50:
            return "MODERATE", "🟡", "risk-moderate", "#ffc107"
        else:
            return "HIGH", "🔴", "risk-high", "#dc3545"

def create_risk_gauge(risk_probability, risk_category):
    """Create a visual gauge for risk"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Level: {risk_category}", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 48}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#d4edda'},
                {'range': [20, 50], 'color': '#fff3cd'},
                {'range': [50, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_probability
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'family': "Arial"}
    )
    
    return fig

def create_factors_chart(factors):
    """Create horizontal bar chart of risk factors"""
    if not factors:
        return None
    
    df = pd.DataFrame(factors, columns=['Factor', 'Impact', 'Value'])
    df = df.head(8)  # Top 8 factors
    
    fig = px.bar(
        df,
        y='Factor',
        x='Impact',
        orientation='h',
        text='Value',
        title="Your Key Risk Factors",
        labels={'Impact': 'Risk Contribution', 'Factor': ''},
        color='Impact',
        color_continuous_scale=['#28a745', '#ffc107', '#dc3545']
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Risk Contribution Score",
        font={'family': "Arial"}
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🩺 Type 2 Diabetes Risk Assessment Tool</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Individual Risk Prediction | '
        'Based on 130,000 NHANES Patients (1999-2023)</div>',
        unsafe_allow_html=True
    )
    
    # Mode selection
    st.sidebar.title("Select Assessment Type")
    assessment_mode = st.sidebar.radio(
        "",
        ["🏠 Public Screening (No Lab Tests)", "🏥 Clinical Assessment (With Lab Results)"],
        help="Choose based on available information"
    )
    
    is_clinical = "Clinical" in assessment_mode
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About This Tool")
    st.sidebar.info(
        "**Model Performance:**\n"
        "- AUC: 96.56%\n"
        "- Precision: 99.86%\n"
        "- Algorithm: Random Forest\n"
        "- Training Data: 130,000 patients\n\n"
        "**Purpose:** Individual-level risk prediction for screening and clinical decision support."
    )
    
    # ========================================================================
    # INPUT SECTION
    # ========================================================================
    
    st.header("📋 Your Information")
    
    # Basic Demographics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=45, step=1)
    
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
    
    with col3:
        race = st.selectbox(
            "Race/Ethnicity",
            ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Non-Hispanic Asian", "Other/Mixed"]
        )
    
    # Body Measurements
    st.subheader("📏 Body Measurements")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        height_ft = st.number_input("Height (feet)", min_value=4, max_value=7, value=5, step=1)
        height_in = st.number_input("Height (inches)", min_value=0, max_value=11, value=8, step=1)
        total_height_inches = height_ft * 12 + height_in
    
    with col2:
        weight_lbs = st.number_input("Weight (pounds)", min_value=80, max_value=500, value=180, step=1)
        bmi = calculate_bmi(weight_lbs, total_height_inches)
        bmi_category, bmi_emoji = get_bmi_category(bmi)
        st.metric("Your BMI", f"{bmi:.1f} {bmi_emoji}", bmi_category)
    
    with col3:
        waist = st.number_input(
            "Waist Circumference (inches)",
            min_value=0.0,
            max_value=80.0,
            value=0.0,
            step=0.5,
            help="Optional but recommended. Measure at belly button level."
        )
    
    # Health History
    st.subheader("🏥 Health History")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        family_history = st.selectbox(
            "Family History of Diabetes?",
            ["No", "Yes", "Unknown"],
            help="Parent, sibling, or child with diabetes"
        )
    
    with col2:
        hypertension = st.selectbox(
            "High Blood Pressure?",
            ["No", "Yes", "Unknown"]
        )
    
    with col3:
        physical_activity = st.selectbox(
            "Physical Activity Level",
            ["Sedentary", "Light", "Moderate", "Active"],
            help="Sedentary: <30 min/week\nLight: 30-150 min/week\nModerate: 150-300 min/week\nActive: >300 min/week"
        )
    
    # Clinical Lab Values (only for clinical mode)
    glucose = hba1c = triglycerides = hdl = systolic_bp = None
    
    if is_clinical:
        st.subheader("🔬 Laboratory Values")
        st.markdown("*Enter recent lab results (within past 3 months)*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            glucose = st.number_input(
                "Fasting Glucose (mg/dL)",
                min_value=50,
                max_value=400,
                value=95,
                step=1,
                help="Normal: <100 | Prediabetes: 100-125 | Diabetes: ≥126"
            )
            
            hba1c = st.number_input(
                "HbA1c (%)",
                min_value=3.0,
                max_value=15.0,
                value=5.4,
                step=0.1,
                help="Normal: <5.7 | Prediabetes: 5.7-6.4 | Diabetes: ≥6.5"
            )
            
            triglycerides = st.number_input(
                "Triglycerides (mg/dL)",
                min_value=20,
                max_value=1000,
                value=150,
                step=5,
                help="Normal: <150 | Borderline High: 150-199 | High: ≥200"
            )
        
        with col2:
            hdl = st.number_input(
                "HDL Cholesterol (mg/dL)",
                min_value=10,
                max_value=100,
                value=50,
                step=1,
                help="Low (risk): <40 (men), <50 (women) | Optimal: ≥60"
            )
            
            systolic_bp = st.number_input(
                "Systolic Blood Pressure (mmHg)",
                min_value=80,
                max_value=200,
                value=120,
                step=1,
                help="Normal: <120 | Elevated: 120-129 | High: ≥130"
            )
    
    # ========================================================================
    # CALCULATE RISK BUTTON
    # ========================================================================
    
    st.markdown("---")
    
    if st.button("🔍 Calculate My Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing your risk profile..."):
            # Predict risk
            if is_clinical:
                risk_prob, factors = predict_diabetes_risk_clinical(
                    age, sex, bmi, waist, race, family_history, hypertension,
                    physical_activity, glucose, hba1c, triglycerides, hdl, systolic_bp
                )
                confidence = "Very High (97.4% AUC)"
            else:
                risk_prob, factors = predict_diabetes_risk_basic(
                    age, sex, bmi, waist, race, family_history, hypertension, physical_activity
                )
                confidence = "High (81.7% AUC)"
            
            risk_level, risk_emoji, risk_class, risk_color = get_risk_category(risk_prob, is_clinical)
            
            # ================================================================
            # RESULTS SECTION
            # ================================================================
            
            st.markdown("---")
            st.header("📊 Your Risk Assessment Results")
            
            # Risk gauge and summary
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig_gauge = create_risk_gauge(risk_prob, risk_level)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                st.markdown(f"""
                <div class="{risk_class}" style="height: 280px; display: flex; flex-direction: column; justify-content: center;">
                    <h1 style="font-size: 3rem; margin: 0;">{risk_emoji} {risk_level} RISK</h1>
                    <h2 style="font-size: 2rem; margin: 10px 0;">{risk_prob:.1f}%</h2>
                    <p style="font-size: 1.1rem; margin: 5px 0;">Confidence: {confidence}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk interpretation
            st.markdown("### 📈 What This Means")
            
            if risk_level == "LOW":
                st.success(
                    f"✅ Your estimated diabetes risk is **{risk_prob:.1f}%**, which is considered **LOW**. "
                    f"The average for your age group is approximately 12%. Continue maintaining a healthy lifestyle!"
                )
            elif risk_level == "MODERATE":
                st.warning(
                    f"⚠️ Your estimated diabetes risk is **{risk_prob:.1f}%**, which is considered **MODERATE**. "
                    f"This is {risk_prob/12:.1f}x higher than average. Consider lifestyle modifications and medical screening."
                )
            else:
                st.error(
                    f"🚨 Your estimated diabetes risk is **{risk_prob:.1f}%**, which is considered **HIGH**. "
                    f"This is {risk_prob/12:.1f}x higher than average. Medical evaluation is strongly recommended."
                )
            
            # Risk factors visualization
            if factors:
                st.markdown("### 🎯 Your Key Risk Factors")
                fig_factors = create_factors_chart(factors)
                st.plotly_chart(fig_factors, use_container_width=True)
                
                # Detailed factors list
                with st.expander("📋 View Detailed Risk Factors"):
                    for i, (factor, impact, value) in enumerate(factors, 1):
                        st.write(f"**{i}. {factor}:** {value} (Impact: {impact:.1f} points)")
            
            # Clinical interpretation (for clinical mode)
            if is_clinical:
                st.markdown("### 🏥 Clinical Interpretation")
                
                clinical_notes = []
                
                if hba1c >= 6.5:
                    clinical_notes.append("⚠️ **CRITICAL:** HbA1c ≥6.5% meets diagnostic criteria for diabetes")
                elif hba1c >= 5.7:
                    clinical_notes.append("⚠️ HbA1c in prediabetes range (5.7-6.4%)")
                
                if glucose >= 126:
                    clinical_notes.append("⚠️ **CRITICAL:** Fasting glucose ≥126 mg/dL meets diagnostic criteria for diabetes")
                elif glucose >= 100:
                    clinical_notes.append("⚠️ Fasting glucose in prediabetes range (100-125 mg/dL)")
                
                if bmi >= 30:
                    clinical_notes.append("⚠️ BMI indicates obesity (Class I or higher)")
                
                if triglycerides >= 200:
                    clinical_notes.append("⚠️ Triglycerides high (≥200 mg/dL)")
                
                if clinical_notes:
                    for note in clinical_notes:
                        st.markdown(note)
                else:
                    st.info("✅ No critical laboratory values detected.")
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            recommendations = []
            
            if risk_level == "HIGH":
                recommendations.append("🏥 **Schedule medical appointment immediately** for comprehensive diabetes screening")
                recommendations.append("🔬 **Get tested:** Fasting glucose and HbA1c tests if not done recently")
                recommendations.append("💊 **Discuss with your doctor:** Potential need for medications or interventions")
            elif risk_level == "MODERATE":
                recommendations.append("🏥 **Schedule medical appointment** within the next month for diabetes screening")
                recommendations.append("🔬 **Get tested:** Fasting glucose and HbA1c tests")
            else:
                recommendations.append("✅ **Continue healthy habits** to maintain low risk")
                recommendations.append("🔬 **Screen regularly:** Get tested every 3 years as recommended for adults")
            
            # Universal recommendations
            if bmi >= 25:
                recommendations.append("⚖️ **Weight management:** Aim for 5-10% weight loss to reduce risk significantly")
            
            if physical_activity == "Sedentary":
                recommendations.append("🏃 **Increase physical activity:** Aim for 150 minutes moderate exercise per week")
            
            recommendations.append("🥗 **Healthy diet:** Focus on whole grains, vegetables, lean proteins, and limited processed foods")
            recommendations.append("🚭 **Avoid tobacco:** If you smoke, consider cessation programs")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Download report button
            st.markdown("---")
            report_data = {
                "Assessment Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Assessment Type": "Clinical" if is_clinical else "Screening",
                "Risk Percentage": f"{risk_prob:.1f}%",
                "Risk Level": risk_level,
                "Confidence": confidence,
                "Age": age,
                "Sex": sex,
                "BMI": f"{bmi:.1f}",
                "Top Risk Factors": ", ".join([f[0] for f in factors[:5]])
            }
            
            st.download_button(
                label="📥 Download Risk Assessment Report",
                data=pd.DataFrame([report_data]).to_csv(index=False),
                file_name=f"diabetes_risk_assessment_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # ========================================================================
    # DISCLAIMER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        <h3>⚠️ Important Disclaimers</h3>
        <p><strong>This tool is for risk assessment only and does NOT diagnose diabetes.</strong></p>
        <ul>
            <li>Only a licensed healthcare provider can diagnose diabetes through proper medical evaluation</li>
            <li>This tool should not replace professional medical advice, diagnosis, or treatment</li>
            <li>If you have concerning symptoms or high-risk results, consult a healthcare provider immediately</li>
            <li>This tool is based on NHANES data (US population) and may not be appropriate for all populations</li>
        </ul>
        <p><strong>For Medical Professionals:</strong> This tool supports clinical decision-making but should not replace clinical judgment and standard diagnostic protocols.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p><strong>Model Details:</strong> Logistic Regression trained on NHANES 1999-2023 data (n=50,000+)</p>
        <p>Public Screening: 81.7% AUC (no labs) | Clinical Assessment: 97.4% AUC (with HbA1c)</p>
        <p><strong>For academic purposes only</strong> | Developed as part of biomedical engineering research project</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
