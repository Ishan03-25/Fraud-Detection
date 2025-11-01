import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🛡️",
    layout="wide"
)

# --- CUSTOM CSS FOR A POLISHED & MODERN DARK THEME ---
# The CSS is defined once and applied via a markdown call.
MODERN_DARK_THEME_CSS = """
<style>
    /* Main Dark Theme */
    body {
        color: #fafafa;
        background-color: #0E1117;
    }
    .stApp {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #fafafa;
    }
    
    /* Card-like elements with shadows */
    [data-testid="stForm"], [data-testid="stMetric"] {
        background-color: #161B22;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363D;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        color: #fafafa;
        border: 1px solid #30363D;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00A99D; /* New Accent Color */
        color: white;
        font-weight: bold;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #00A99D; /* New Accent Color */
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #007A70; /* Darker shade for hover */
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22;
    }
</style>
"""
st.markdown(MODERN_DARK_THEME_CSS, unsafe_allow_html=True)


# --- SESSION STATE INITIALIZATION ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Load the saved model pipeline, cached for performance."""
    try:
        model = joblib.load('fraud_detection_pipeline.pickle')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'fraud_detection_pipeline.pickle' is in the same directory.")
        return None

# --- HELPER FUNCTIONS ---
def validate_transaction(ttype, amount, old_b_orig, new_b_orig, old_b_dest, new_b_dest):
    """
    Performs a sanity check on both sender and receiver transaction values.
    Returns (isValid, deviation, message).
    """
    epsilon = 0.01  # Tolerance for float comparisons
    
    # --- Sender Validation ---
    if ttype in ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT']:
        expected_new_b_orig = old_b_orig - amount
        deviation = abs(expected_new_b_orig - new_b_orig)
        if deviation > epsilon:
            return False, deviation, f"Sender's balance is inconsistent. Expected {expected_new_b_orig:.2f}, but new balance is {new_b_orig:.2f}."
    elif ttype == 'CASH_IN':
        expected_new_b_orig = old_b_orig + amount
        deviation = abs(expected_new_b_orig - new_b_orig)
        if deviation > epsilon:
            return False, deviation, f"Sender's balance is inconsistent. Expected {expected_new_b_orig:.2f}, but new balance is {new_b_orig:.2f}."

    # --- Receiver Validation ---
    if ttype in ['TRANSFER', 'PAYMENT']:
        expected_new_b_dest = old_b_dest + amount
        deviation = abs(expected_new_b_dest - new_b_dest)
        if deviation > epsilon:
            return False, deviation, f"Receiver's balance is inconsistent. Expected {expected_new_b_dest:.2f}, but new balance is {new_b_dest:.2f}."

    return True, 0.0, "Valid"

def calculate_deviation_fraud_prob(deviation, balance):
    """Calculates fraud probability based on the deviation from the expected balance."""
    if balance <= 0:
        return 100.0 if deviation > 0 else 0.0
    deviation_percentage = (deviation / balance) * 100
    fraud_prob = deviation_percentage * 2
    return min(100.0, fraud_prob)


# --- UI RENDERING FUNCTIONS ---
def render_single_prediction_tab(model):
    """Renders the UI and logic for the single prediction tab."""
    with st.form("transaction_form"):
        st.header("Single Transaction Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            transaction_type = st.selectbox('Transaction Type', ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT'])
        with col2:
            amount = st.number_input('Amount', min_value=0.0, value=1000.0)

        st.divider()
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Sender's Account")
            old_balance_original = st.number_input('Old Balance', min_value=0.0, value=10000.0, key='old_sender')
            new_balance_original = st.number_input('New Balance', min_value=0.0, value=9000.0, key='new_sender')
        with col4:
            st.subheader("Receiver's Account")
            old_balance_destination = st.number_input('Old Balance', min_value=0.0, value=0.0, key='old_receiver')
            new_balance_destination = st.number_input('New Balance', min_value=0.0, value=0.0, key='new_receiver')

        submitted = st.form_submit_button("Analyze Transaction")

    if submitted:
        is_valid, deviation, validation_message = validate_transaction(
            transaction_type, amount, old_balance_original, new_balance_original, old_balance_destination, new_balance_destination
        )
        
        input_data = pd.DataFrame([{'type': transaction_type, 'amount': amount, 'oldbalanceOrg': old_balance_original, 'newbalanceOrig': new_balance_original, 'oldbalanceDest': old_balance_destination, 'newbalanceDest': new_balance_destination}])
        
        if not is_valid:
            prediction = [1]
            base_balance = max(old_balance_original, old_balance_destination)
            fraud_prob = calculate_deviation_fraud_prob(deviation, base_balance)
        else:
            with st.spinner('Analyzing...'):
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)
                fraud_prob = prediction_proba[0][1] * 100

        st.divider()
        st.header("Prediction Result")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=fraud_prob, title={'text': "Fraud Risk"},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#00A99D"},
                       'steps': [{'range': [0, 40], 'color': 'lightgreen'}, {'range': [40, 70], 'color': 'yellow'}, {'range': [70, 100], 'color': 'red'}]}))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10), font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if prediction[0] == 1:
                st.error(f'🚨 High Risk: Transaction is likely fraudulent! (Confidence: {fraud_prob:.2f}%)', icon="⚠️")
            else:
                st.success(f'✅ Low Risk: Transaction appears legitimate. (Confidence: {100-fraud_prob:.2f}%)', icon="👍")
            
            with st.expander("Show Risk Factor Analysis"):
                has_risk_factors = False
                if not is_valid:
                    st.markdown(f"**- ⚠️ Impossible Balance Change:** {validation_message}")
                    has_risk_factors = True
                if transaction_type == 'CASH_OUT':
                    st.markdown(f"**- 💸 High-Risk Type:** `{transaction_type}` transactions are often used for fraud.")
                    has_risk_factors = True
                if old_balance_original > 0 and new_balance_original == 0:
                    st.markdown("**-  emptying Account Emptied:** Sender's account was depleted, a common indicator.")
                    has_risk_factors = True
                if amount > 100000:
                    st.markdown("**- 💰 Large Amount:** The transaction amount is unusually high.")
                    has_risk_factors = True
                
                if not has_risk_factors and is_valid:
                    st.markdown("✅ No major risk factors detected based on simple rules.")

        history_entry = input_data.to_dict('records')[0]
        history_entry['prediction'] = 'Fraudulent' if prediction[0] == 1 else 'Legitimate'
        history_entry['confidence'] = f"{fraud_prob:.2f}%"
        st.session_state.history.insert(0, history_entry)

def render_batch_prediction_tab(model, batch_data):
    """Renders the UI and logic for the batch prediction tab."""
    st.header("Batch Transaction Prediction")
    if batch_data is not None:
        try:
            with st.spinner("Processing batch file..."):
                predictions = model.predict(batch_data)
                prediction_probas = model.predict_proba(batch_data)
                results_df = batch_data.copy()
                results_df['isFraud_Prediction'] = ['Fraudulent' if p == 1 else 'Legitimate' for p in predictions]
                results_df['Fraud_Probability'] = [f"{p[1]*100:.2f}%" for p in prediction_probas]
            st.dataframe(results_df)
            st.download_button(label="Download Results", data=results_df.to_csv(index=False).encode('utf-8'), file_name='prediction_results.csv')
        except Exception as e:
            st.error(f"Error during batch prediction: {e}")
    else:
        st.info("Upload a CSV file in the sidebar to perform batch prediction.")

def render_dashboard_tab():
    """Renders the UI for the dashboard and history tab."""
    st.header("Session Dashboard & History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        
        total_predictions = len(history_df)
        fraud_predictions = len(history_df[history_df['prediction'] == 'Fraudulent'])
        fraud_rate = (fraud_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        
        col1, col2 = st.columns(2)
        col1.metric("📊 Total Predictions", f"{total_predictions}")
        col2.metric("🚨 Fraudulent Transactions", f"{fraud_predictions} ({fraud_rate:.2f}%)")
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Prediction Breakdown")
            prediction_counts = history_df['prediction'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=prediction_counts.index, values=prediction_counts.values, hole=.4, marker_colors=['#E74C3C', '#2ECC71'])])
            fig.update_layout(showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("Transaction Types Analyzed")
            type_counts = history_df['type'].value_counts()
            st.bar_chart(type_counts, use_container_width=True)

        with st.expander("Show Raw Prediction History", expanded=False):
            st.dataframe(history_df)
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions made in this session yet. Use the 'Single Prediction' tab to start.")


# --- MAIN APP EXECUTION ---
model = load_model()
if model is None:
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/c/c3/Lock-icon.svg", width=80)
    st.title("Fraud Detection System")
    st.info("This application uses a Machine Learning model to detect fraudulent financial transactions.")
    
    st.subheader("Upload for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")
    
    batch_data = None
    if uploaded_file:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.success("File uploaded! Go to the 'Batch Prediction' tab to see results.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# --- MAIN APP LAYOUT ---
st.title("🛡️ Fraud Detection Dashboard")
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction", "📊 Dashboard & History"])

with tab1:
    render_single_prediction_tab(model)

with tab2:
    render_batch_prediction_tab(model, batch_data)

with tab3:
    render_dashboard_tab()


