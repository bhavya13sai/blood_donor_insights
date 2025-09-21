# blood_donation_dashboard_full.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ---------------------------
# 1. App layout
# ---------------------------
st.set_page_config(page_title="Blood Donation Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: red;'>🩸 Blood Donation ML Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# 2. Load dataset and train model
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('datasets/transfusion.data')
    df.rename(columns={'whether he/she donated blood in March 2007': 'target'}, inplace=True)
    return df

data = load_data()
data['avg_blood_per_donation'] = data['Monetary (c.c. blood)'] / data['Frequency (times)']
data['donation_consistency'] = data['Frequency (times)'] / data['Time (months)']

features = ['Recency (months)', 'Frequency (times)', 'Time (months)',
            'Monetary (c.c. blood)', 'avg_blood_per_donation', 'donation_consistency']

X_train, X_test, y_train, y_test = train_test_split(
    data[features], data['target'], test_size=0.25, random_state=42, stratify=data['target']
)

for col in ['Monetary (c.c. blood)', 'avg_blood_per_donation']:
    X_train[col + '_log'] = np.log(X_train[col] + 1)
    X_test[col + '_log'] = np.log(X_test[col] + 1)
    X_train.drop(columns=col, inplace=True)
    X_test.drop(columns=col, inplace=True)

model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# 3. Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["Single Donor", "Batch / Upload", "Interactive Analytics"])

# ---------------------------
# Tab 1: Single Donor Prediction
# ---------------------------
with tab1:
    st.subheader("Single Donor Prediction")
    st.sidebar.header("Single Donor Input")
    def single_input(df):
        input_data = {}
        for col in ['Recency (months)', 'Frequency (times)', 'Time (months)', 'Monetary (c.c. blood)']:
            min_val, max_val = int(df[col].min()), int(df[col].max())
            default = int(df[col].median())
            input_data[col] = st.sidebar.slider(col, min_val, max_val, default)
        return pd.DataFrame(input_data, index=[0])

    user_input = single_input(data)
    user_input['avg_blood_per_donation'] = user_input['Monetary (c.c. blood)'] / max(user_input['Frequency (times)'].iloc[0],1)
    user_input['donation_consistency'] = user_input['Frequency (times)'] / max(user_input['Time (months)'].iloc[0],1)
    user_input['Monetary (c.c. blood)_log'] = np.log(user_input['Monetary (c.c. blood)'] + 1)
    user_input['avg_blood_per_donation_log'] = np.log(user_input['avg_blood_per_donation'] + 1)
    user_input_model = user_input[X_train.columns]

    pred_prob = model.predict_proba(user_input_model)[0,1]
    st.metric("Predicted Probability", f"{pred_prob:.2f}")

    csv_single = user_input.copy()
    csv_single['Predicted_Probability'] = pred_prob
    st.download_button("Download Single Donor CSV", data=csv_single.to_csv(index=False).encode('utf-8'),
                       file_name="single_donor_prediction.csv", mime='text/csv')

# ---------------------------
# Tab 2: Batch / Upload
# ---------------------------
with tab2:
    st.subheader("Batch Prediction / File Upload")
    simulate_batch = st.checkbox("Generate Simulated Batch")
    uploaded_file = st.file_uploader("Or upload CSV file", type="csv")

    batch_data = None
    if simulate_batch:
        n_samples = st.slider("Number of simulated donors", 10, 200, 50)
        batch_data = pd.DataFrame()
        for col in ['Recency (months)', 'Frequency (times)', 'Time (months)', 'Monetary (c.c. blood)']:
            batch_data[col] = np.random.randint(int(data[col].min()), int(data[col].max())+1, size=n_samples)
    elif uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        required_cols = ['Recency (months)', 'Frequency (times)', 'Time (months)', 'Monetary (c.c. blood)']
        if not all(col in batch_data.columns for col in required_cols):
            st.error(f"CSV must include columns: {required_cols}")
            batch_data = None

    if batch_data is not None:
        batch_data['avg_blood_per_donation'] = batch_data['Monetary (c.c. blood)'] / batch_data['Frequency (times)'].replace(0,1)
        batch_data['donation_consistency'] = batch_data['Frequency (times)'] / batch_data['Time (months)'].replace(0,1)
        batch_data['Monetary (c.c. blood)_log'] = np.log(batch_data['Monetary (c.c. blood)'] + 1)
        batch_data['avg_blood_per_donation_log'] = np.log(batch_data['avg_blood_per_donation'] + 1)
        batch_input_model = batch_data[X_train.columns]
        batch_probs = model.predict_proba(batch_input_model)[:,1]
        batch_data['Predicted_Probability'] = batch_probs

        st.write("Predicted Probabilities:")
        st.dataframe(batch_data)
        st.write("Histogram of Predicted Probabilities:")
        fig, ax = plt.subplots()
        ax.hist(batch_probs, bins=15, color='red', alpha=0.7)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Number of Donors")
        st.pyplot(fig)

        st.download_button("Download Batch Predictions CSV",
                           data=batch_data.to_csv(index=False).encode('utf-8'),
                           file_name="batch_predictions.csv", mime='text/csv')

# ---------------------------
# Tab 3: Interactive Analytics
# ---------------------------
with tab3:
    st.subheader("Interactive Analytics")
    if batch_data is None:
        st.info("Upload or simulate batch data in Tab 2 to explore analytics.")
    else:
        # Probability filter
        st.write("Filter predicted probability range:")
        min_prob, max_prob = st.slider("Probability Range", 0.0, 1.0, (0.0,1.0))
        filtered_data = batch_data[(batch_data['Predicted_Probability'] >= min_prob) &
                                   (batch_data['Predicted_Probability'] <= max_prob)]
        st.metric("Number of Donors in Filtered Range", len(filtered_data))
        st.metric("Mean Predicted Probability", f"{filtered_data['Predicted_Probability'].mean():.2f}")

        # Scatter plot
        feature_to_plot = st.selectbox("Select Feature for X-axis", 
                                       ['Recency (months)','Frequency (times)','Time (months)','Monetary (c.c. blood)'])
        fig, ax = plt.subplots()
        ax.scatter(filtered_data[feature_to_plot], filtered_data['Predicted_Probability'], color='red', alpha=0.6)
        ax.set_xlabel(feature_to_plot)
        ax.set_ylabel("Predicted Probability")
        ax.set_title(f"{feature_to_plot} vs Predicted Probability")
        st.pyplot(fig)

        # Dynamic Feature Importance
        st.write("Dynamic Feature Importance")
        dynamic_importance = {}
        for feat, coef in zip(X_train.columns, model.coef_[0]):
            if feat in filtered_data.columns:
                dynamic_importance[feat] = np.mean(np.abs(filtered_data[feat] * coef))
            else:
                dynamic_importance[feat] = 0
        dynamic_importance_df = pd.DataFrame.from_dict(dynamic_importance, orient='index', columns=['Impact'])
        dynamic_importance_df = dynamic_importance_df.sort_values(by='Impact', ascending=False)
        st.bar_chart(dynamic_importance_df)
