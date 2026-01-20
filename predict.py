import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
h1, h2, h3, label {
    color: white !important;
}
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ================= RESET FUNCTION =================
def reset_form(): #Resets all input fields to default values
    defaults = {
        "customerID": "",
        "gender": " ",
        "SeniorCitizen": 0,
        "Partner": " ",
        "Dependents": " ",
        "tenure": 0,
        "PhoneService": " ",
        "InternetService": " ",
        "OnlineSecurity": " ",
        "TechSupport": " ",
        "Contract": " ",
        "PaperlessBilling": " ",
        "PaymentMethod": " ",
        "MonthlyCharges": 0.0,
        "TotalCharges": 0.0
    }
    for k, v in defaults.items():
        st.session_state[k] = v

# ================= INIT SESSION STATE =================
if "customerID" not in st.session_state:
    reset_form()

# ================= TITLE =================
st.title("📊 Telecom  Customer Churn Prediction APP")
img = Image.open("telecommunication.jpg")
st.image(img, width=700)

# ================= LOAD MODEL =================
scaler = pickle.load(open("minmax_scaler.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("label_encoders.pkl", "rb"))

# ================= FORM =================
with st.form("churn_form", clear_on_submit=False):

    # -------- CUSTOMER DEMOGRAPHICS --------
    with st.expander("🧍 Customer Demographics", expanded=True):
        c1, c2 = st.columns(2)

        with c1:
            customerID = st.text_input("Customer ID", key="customerID")
            gender = st.selectbox("👤 Gender", [" ","Female", "Male"], key="gender")
            SeniorCitizen = st.slider("🧓 Senior Citizen", 0, 1, key="SeniorCitizen")

        with c2:
            Partner = st.selectbox("💍 Partner", [" ","Yes", "No"], key="Partner")
            Dependents = st.selectbox("👶 Dependents", [" ","Yes", "No"], key="Dependents")
            tenure = st.slider("📆 Tenure (Months)", 0, 72, key="tenure")

    # -------- SERVICES USED --------
    with st.expander("📞 Services Used", expanded=True):
        c3, c4 = st.columns(2)

        with c3:
            PhoneService = st.selectbox("📞 Phone Service", [" ","Yes", "No"], key="PhoneService")
            InternetService = st.selectbox(
                "🌐 Internet Service", [" ","DSL", "Fiber optic", "No"], key="InternetService"
            )

        with c4:
            OnlineSecurity = st.selectbox(
                "🔐 Online Security", [" ","Yes", "No", "No internet service"],
                key="OnlineSecurity"
            )
            TechSupport = st.selectbox(
                "🛠 Tech Support", [" ","Yes", "No", "No internet service"],
                key="TechSupport"
            )

    # -------- CONTRACT DETAILS --------
    with st.expander("📄 Contract Details", expanded=True):
        Contract = st.selectbox(
            "📄 Contract Type",
            [" ","Month-to-month", "One year", "Two year"],
            key="Contract"
        )
        PaperlessBilling = st.selectbox(
            "🧾 Paperless Billing", [" ","Yes", "No"], key="PaperlessBilling"
        )

    # -------- BILLING INFORMATION --------
    with st.expander("💳 Billing Information", expanded=True):
        c5, c6 = st.columns(2)

        with c5:
            PaymentMethod = st.selectbox(
                "💳 Payment Method",
                [" ","Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"],
                key="PaymentMethod"
            )

        with c6:
            MonthlyCharges = st.slider("💰 Monthly Charges", 0.0, 200.0, key="MonthlyCharges")
            TotalCharges = st.slider("💵 Total Charges", 0.0, 9000.0, key="TotalCharges")

    # -------- BUTTONS --------
    col_submit, col_clear = st.columns(2)

    with col_submit:
        submit = st.form_submit_button("🚀 RUN CHURN ANALYSIS")

    with col_clear:
        st.form_submit_button("🧹 CLEAR FORM", on_click=reset_form)

# ================= VALIDATION & PREDICTION =================
if submit:
    cate_col = [customerID, gender, Partner, Dependents, PhoneService,
                InternetService, OnlineSecurity, TechSupport, Contract,
                PaperlessBilling, PaymentMethod]
    if " " in cate_col or tenure == 0 or MonthlyCharges == 0 or TotalCharges == 0:
        st.error("⚠️ Please fill all required fields")
    else:
        input_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [SeniorCitizen],
            "Partner": [Partner],
            "Dependents": [Dependents],
            "tenure": [tenure],
            "PhoneService": [PhoneService],
            "InternetService": [InternetService],
            "OnlineSecurity": [OnlineSecurity],
            "TechSupport": [TechSupport],
            "Contract": [Contract],
            "PaperlessBilling": [PaperlessBilling],
            "PaymentMethod": [PaymentMethod],
            "MonthlyCharges": [MonthlyCharges],
            "TotalCharges": [TotalCharges]
        })

        # Encode categorical columns
        for col in encoder:
            if col in input_data.columns:
                input_data[col] = encoder[col].transform(input_data[col])

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]


        st.markdown("### 📊 Prediction Result")

        if prediction == 1:
            st.error(
                f"⚠️ Customer is likely to CHURN"
            )
        else:
            st.success(
                f"✅ Customer is NOT likely to churn"
            )
