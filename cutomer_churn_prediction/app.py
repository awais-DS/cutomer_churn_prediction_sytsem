import streamlit as st
import pandas as pd
from src.predict import run_production_inference

# Set up clean web page configurations
st.set_page_config(
    page_title="Customer Churn Risk Analytics Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application Header Title
st.title("📊 Customer Churn Risk Analytics Portal")
st.markdown("""
This system evaluates raw customer parameters, detects attrition patterns via machine learning models, 
and delivers diagnostic risk factor breakdowns along with actionable retention suggestions.
""")
st.markdown("---")

# Setup clean visual input containers using column splits
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographic & Account Details")
    
    # 1. Gender Selection (Nominal Binary)
    gender = st.selectbox("Gender:", ["Male", "Female"])
    
    # 2. Dependents Status (Nominal Binary)
    dependents = st.selectbox("Has Dependents:", ["Yes", "No"])
    
    # 3. Contract Type Selection (Ordinal Category)
    contract = st.selectbox(
        "Active Contract Type:", 
        ["month-to-month", "One year", "Two year"]
    )

with col2:
    st.subheader("Financial & Usage Details")
    
    # 4. Tenure Metric Selection (Numeric)
    tenure = st.slider(
        "Account Tenure Length (Months):", 
        min_value=0, 
        max_value=72, 
        value=12,
        help="Number of months the customer has stayed with the company."
    )
    
    # 5. Monthly Bill Charges Selection (Numeric)
    monthly_charges = st.number_input(
        "Current Monthly Charges ($):", 
        min_value=0.0, 
        max_value=200.0, 
        value=50.0,
        step=1.00,
        help="The amount charged to the customer monthly."
    )

st.markdown("---")

# Execution Triggers
if st.button("Analyze Customer Risk Profile", type="primary"):
    
    # Map visual states back to the exact dictionary format expected by src/
    user_payload = {
        'gender': gender,
        'Dependents': dependents,
        'Contract': contract,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges
    }
    
    with st.spinner("Processing customer profile through neural architecture layers..."):
        # Single-line execution of your isolated inference scripts
        result = run_production_inference(user_payload)
        
    # Check if files or dependencies threw structural framework exceptions
    if result["status"] == "error":
        st.error(f"Inference System Fault: {result['message']}")
        
    else:
        # Layout Results beautifully using distinct visual feedback boxes
        st.subheader("🔍 Retention Diagnostic Breakdown")
        
        # Split display between status classifications
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if result["churn_risk"] == "Yes":
                st.metric(label="Churn Risk Status", value="HIGH RISK", delta="Action Required", delta_color="inverse")
            else:
                st.metric(label="Churn Risk Status", value="LOW RISK", delta="Stable Profile", delta_color="normal")
                
        with res_col2:
            st.metric(label="Calculated Probability Rate", value=result["churn_probability"])
            
        # 1. Display Risk Factors
        st.markdown("#### **Detected Attrition Risk Factors:**")
        for factor in result["risk_factors"]:
            st.write(f"⚠️ {factor}")
            
        # 2. Display Strategic Recommendations
        st.markdown("#### **Tailored Prevention Action Plan:**")
        for action in result["action_plan"]:
            st.info(f"💡 {action}")

# Professional Footer Note
st.markdown("---")
st.caption("Engineered using Scikit-Learn Pipelines and Streamlit Core Engine Frameworks.")
