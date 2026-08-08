import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢"
)


# =========================================================
# 2. LOAD MODEL AND PERFORMANCE
# =========================================================

model = joblib.load("titanic_model.pkl")
performance = joblib.load("model_performance.pkl")


# =========================================================
# 3. SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "live_records" not in st.session_state:
    st.session_state.live_records = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# =========================================================
# 4. TITLE
# =========================================================

st.title("🚢 Titanic Survival Prediction")

st.write(
    "Enter passenger details to predict whether the passenger "
    "is likely to survive."
)


# =========================================================
# 5. PASSENGER INPUT
# =========================================================

st.header("👤 Passenger Details")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex_input = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=25
)

sibsp = st.number_input(
    "Siblings/Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Parents/Children Aboard",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.0
)

embarked = st.selectbox(
    "Embarked",
    ["S", "Q"]
)


# =========================================================
# 6. CONVERT INPUTS
# =========================================================

sex = 0 if sex_input == "Male" else 1

embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0


# =========================================================
# 7. CREATE INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked_Q": [embarked_Q],
    "Embarked_S": [embarked_S]
})


# =========================================================
# 8. PREDICTION
# =========================================================
# =========================================================
# 8. PREDICTION
# =========================================================

if st.button("🚢 Predict Survival"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    survival_probability = probability[0][1] * 100
    death_probability = probability[0][0] * 100

    # -----------------------------------------------------
    # Prediction Result
    # -----------------------------------------------------

    st.subheader("🔮 Prediction Result")

    if prediction[0] == 1:
        st.success("🎉 Passenger is likely to Survive")
        prediction_text = "Survived"
    else:
        st.error("❌ Passenger is not likely to Survive")
        prediction_text = "Did Not Survive"

    # -----------------------------------------------------
    # Probabilities
    # -----------------------------------------------------

    st.write(
        f"### Survival Probability: "
        f"{survival_probability:.2f}%"
    )

    st.progress(int(survival_probability))

    st.write(
        f"### Death Probability: "
        f"{death_probability:.2f}%"
    )

    st.progress(int(death_probability))

    # -----------------------------------------------------
    # Prediction Confidence
    # -----------------------------------------------------

    confidence = max(
        survival_probability,
        death_probability
    )

    st.subheader("🎯 Prediction Confidence")

    if confidence >= 80:
        st.success(
            f"🟢 High Confidence: {confidence:.2f}%"
        )

    elif confidence >= 60:
        st.warning(
            f"🟡 Medium Confidence: {confidence:.2f}%"
        )

    else:
        st.error(
            f"🔴 Low Confidence: {confidence:.2f}%"
        )

    # -----------------------------------------------------
    # Save Prediction History
    # -----------------------------------------------------

    history_record = {
        "Pclass": pclass,
        "Sex": sex_input,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "Prediction": prediction_text,
        "Survival Probability":
            f"{survival_probability:.2f}%",
        "Confidence":
            f"{confidence:.2f}%"
    }

    st.session_state.history.append(history_record)

    # -----------------------------------------------------
    # SAVE LAST PREDICTION
    # -----------------------------------------------------

    st.session_state.last_prediction = {
        "Pclass": pclass,
        "Sex": sex_input,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "Prediction": int(prediction[0]),
        "Survival Probability":
            round(survival_probability, 2),
        "Confidence":
            round(confidence, 2)
    }

    st.success(
        "✅ Prediction saved. Now enter the actual outcome below."
    )

# =========================================================
# 9. PREDICTION HISTORY
# =========================================================

st.header("📋 Prediction History")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

else:

    st.info("No predictions made yet.")

    # =========================================================
# DOWNLOAD PREDICTION HISTORY
# =========================================================

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.download_button(
        label="📥 Download Prediction History",
        data=history_df.to_csv(index=False),
        file_name="titanic_prediction_history.csv",
        mime="text/csv"
    )


# =========================================================
# 10. CLEAR PREDICTION HISTORY
# =========================================================

if st.button("🗑️ Clear Prediction History"):

    st.session_state.history = []

    st.rerun()


# 11. RECORD ACTUAL OUTCOME

if st.session_state.last_prediction is not None:

    st.header("📝 Record Actual Outcome")

    # Show latest prediction
    latest = st.session_state.last_prediction

    st.info(
        f"Latest Prediction: "
        f"{'Survived' if latest['Prediction'] == 1 else 'Did Not Survive'} "
        f"| Survival Probability: "
        f"{latest['Survival Probability']:.2f}%"
    )

    actual_outcome = st.selectbox(
        "What was the actual outcome?",
        [
            "Select",
            "Survived",
            "Did Not Survive"
        ],
        key="actual_outcome"
    )

    if st.button("💾 Save Actual Outcome"):

        if actual_outcome == "Select":

            st.warning(
                "⚠️ Please select the actual outcome."
            )

        else:

            # Convert actual outcome to 0/1
            actual = (
                1
                if actual_outcome == "Survived"
                else 0
            )

            # Copy latest prediction
            record = st.session_state.last_prediction.copy()

            # Add actual outcome
            record["Actual"] = actual

            # Add actual text
            record["Actual Outcome"] = actual_outcome

            # -------------------------------------------------
            # SAVE RECORD
            # -------------------------------------------------

            st.session_state.live_records.append(record)

            st.success(
                "✅ Actual outcome saved successfully!"
            )

            # -------------------------------------------------
            # CLEAR LAST PREDICTION
            # -------------------------------------------------

            st.session_state.last_prediction = None

            # Refresh page so metrics update immediately
            st.rerun()


# =========================================================
# 12. MODEL PERFORMANCE
# =========================================================

st.header("📊 Model Performance")

accuracy = performance["accuracy"]
cm = performance["confusion_matrix"]
report = performance["classification_report"]


# ---------------------------------------------------------
# Model Accuracy
# ---------------------------------------------------------

st.subheader("Model Accuracy")

st.metric(
    "Random Forest Accuracy",
    f"{accuracy * 100:.2f}%"
)


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

st.subheader("Confusion Matrix")

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual: Did Not Survive",
        "Actual: Survived"
    ],
    columns=[
        "Predicted: Did Not Survive",
        "Predicted: Survived"
    ]
)

st.dataframe(
    cm_df,
    use_container_width=True
)

# Classification Report

st.subheader("Classification Report")

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True
)


# 13. LIVE MODEL PERFORMANCE

st.header("📈 Live Model Performance")

if len(st.session_state.live_records) > 0:

    live_df = pd.DataFrame(
        st.session_state.live_records
    )

    # -----------------------------------------------------
    # Actual and Predicted Values
    # -----------------------------------------------------

    y_true = live_df["Actual"].astype(int)
    y_pred = live_df["Prediction"].astype(int)

    # -----------------------------------------------------
    # Calculate Metrics
    # -----------------------------------------------------

    live_accuracy = accuracy_score(
        y_true,
        y_pred
    )

    live_precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    live_recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    live_f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # -----------------------------------------------------
    # Number of Records
    # -----------------------------------------------------

    st.write(
        f"**Live Predictions With Known Actual Outcomes: "
        f"{len(live_df)}**"
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Live Accuracy",
            f"{live_accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Live Precision",
            f"{live_precision * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Live Recall",
            f"{live_recall * 100:.2f}%"
        )

    with col4:
        st.metric(
            "Live F1 Score",
            f"{live_f1 * 100:.2f}%"
        )

    # -----------------------------------------------------
    # Live Confusion Matrix
    # -----------------------------------------------------

    st.subheader("📊 Live Confusion Matrix")

    live_cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    live_cm_df = pd.DataFrame(
        live_cm,
        index=[
            "Actual: Did Not Survive",
            "Actual: Survived"
        ],
        columns=[
            "Predicted: Did Not Survive",
            "Predicted: Survived"
        ]
    )

    st.dataframe(
        live_cm_df,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Live Performance Records
    # -----------------------------------------------------

    st.subheader("📋 Live Performance Records")

    display_df = live_df.copy()

    display_df["Prediction"] = (
        display_df["Prediction"]
        .map({
            0: "Did Not Survive",
            1: "Survived"
        })
    )

    display_df["Actual"] = (
        display_df["Actual"]
        .map({
            0: "Did Not Survive",
            1: "Survived"
        })
    )

    st.dataframe(
        display_df,
        use_container_width=True
    )

else:

    st.info(
        "No actual outcomes recorded yet. "
        "Make a prediction and enter its actual "
        "outcome to calculate live performance."
    )

    # =========================================================
# =========================================================
# 14. DOWNLOAD LIVE PERFORMANCE DATA
# =========================================================

st.header("📥 Download Live Performance Data")

if len(st.session_state.live_records) > 0:

    live_df = pd.DataFrame(
        st.session_state.live_records
    )

    csv_data = live_df.to_csv(index=False)

    st.download_button(
        label="⬇️ Download Live Performance CSV",
        data=csv_data,
        file_name="titanic_live_performance.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No live performance records available to download."
    )