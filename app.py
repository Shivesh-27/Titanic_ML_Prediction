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
    page_icon="🚢",
    layout="wide"
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

col1, col2 = st.columns(2)

with col1:

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


with col2:

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

if st.button("🚢 Predict Survival", use_container_width=True):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    survival_probability = probability[0][1] * 100
    death_probability = probability[0][0] * 100

    confidence = max(
        survival_probability,
        death_probability
    )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.header("🔮 Prediction Result")

    if prediction[0] == 1:

        st.success(
            "🎉 Passenger is likely to Survive"
        )

        prediction_text = "Survived"

    else:

        st.error(
            "❌ Passenger is not likely to Survive"
        )

        prediction_text = "Did Not Survive"


    # =====================================================
    # PROBABILITIES
    # =====================================================

    st.subheader("📊 Prediction Probabilities")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Survival Probability",
            f"{survival_probability:.2f}%"
        )

        st.progress(
            int(survival_probability)
        )


    with col2:

        st.metric(
            "Death Probability",
            f"{death_probability:.2f}%"
        )

        st.progress(
            int(death_probability)
        )


    # =====================================================
    # PREDICTION CONFIDENCE
    # =====================================================

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


    # =====================================================
    # SAVE PREDICTION HISTORY
    # =====================================================

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

    st.session_state.history.append(
        history_record
    )


    # =====================================================
    # SAVE LAST PREDICTION
    # =====================================================

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
            survival_probability,

        "Death Probability":
            death_probability,

        "Confidence":
            confidence
    }


    st.success(
        "✅ Prediction saved. "
        "Now enter the actual outcome below."
    )


# =========================================================
# 9. PREDICTION EXPLANATION
# =========================================================

st.header("🔬 Prediction Explanation")

feature_names = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_Q",
    "Embarked_S"
]

importance_values = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df["Importance (%)"] = (
    importance_df["Importance"] * 100
).round(2)


st.subheader("📊 Features Influencing the Model")

st.bar_chart(
    importance_df.head(5).set_index(
        "Feature"
    )["Importance (%)"]
)


top_feature = importance_df.iloc[0]

st.success(
    f"⭐ Most influential feature: "
    f"{top_feature['Feature']} "
    f"({top_feature['Importance (%)']:.2f}%)"
)

st.info(
    "Feature importance shows how much each feature "
    "contributes to the Random Forest model's decisions. "
    "It does not mean that the feature alone caused the prediction."
)


# =========================================================
# 10. PREDICTION HISTORY
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

    st.download_button(
        label="📥 Download Prediction History",
        data=history_df.to_csv(index=False),
        file_name="titanic_prediction_history.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No predictions made yet."
    )


# =========================================================
# 11. CLEAR PREDICTION HISTORY
# =========================================================

if st.button("🗑️ Clear Prediction History"):

    st.session_state.history = []

    st.rerun()


# =========================================================
# 12. RECORD ACTUAL OUTCOME
# =========================================================

if st.session_state.last_prediction is not None:

    st.header("📝 Record Actual Outcome")

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

            actual = (
                1
                if actual_outcome == "Survived"
                else 0
            )

            record = (
                st.session_state
                .last_prediction
                .copy()
            )

            record["Actual"] = actual

            record["Actual Outcome"] = (
                actual_outcome
            )

            st.session_state.live_records.append(
                record
            )

            st.success(
                "✅ Actual outcome saved successfully!"
            )

            st.session_state.last_prediction = None

            st.rerun()


# =========================================================
# 13. MODEL PERFORMANCE
# =========================================================

st.header("📊 Model Performance")

accuracy = performance["accuracy"]

cm = performance["confusion_matrix"]

report = performance["classification_report"]


# =========================================================
# MODEL ACCURACY
# =========================================================

st.subheader("🎯 Model Accuracy")

st.metric(
    "Random Forest Accuracy",
    f"{accuracy * 100:.2f}%"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("📊 Confusion Matrix")

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


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

st.subheader("📋 Classification Report")

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True
)


# =========================================================
# 14. LIVE MODEL PERFORMANCE
# =========================================================

st.header("📈 Live Model Performance")

if len(st.session_state.live_records) > 0:

    live_df = pd.DataFrame(
        st.session_state.live_records
    )

    y_true = live_df["Actual"].astype(int)

    y_pred = live_df["Prediction"].astype(int)


    # =====================================================
    # LIVE METRICS
    # =====================================================

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


    st.write(
        f"**Live Predictions With Known Actual Outcomes: "
        f"{len(live_df)}**"
    )


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


    # =====================================================
    # LIVE CONFUSION MATRIX
    # =====================================================

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


    # =====================================================
    # LIVE PERFORMANCE RECORDS
    # =====================================================

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
# 15. DOWNLOAD LIVE PERFORMANCE DATA
# =========================================================

st.header("📥 Download Live Performance Data")

if len(st.session_state.live_records) > 0:

    live_df = pd.DataFrame(
        st.session_state.live_records
    )

    csv_data = live_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Live Performance CSV",
        data=csv_data,
        file_name="titanic_live_performance.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No live performance records available "
        "to download."
    )


# =========================================================
# 16. LIVE ACCURACY TREND
# =========================================================

st.header("📈 Live Accuracy Trend")

if len(st.session_state.live_records) > 0:

    live_df = pd.DataFrame(
        st.session_state.live_records
    )

    correct = (
        live_df["Prediction"].astype(int)
        ==
        live_df["Actual"].astype(int)
    )

    cumulative_accuracy = (
        correct.cumsum()
        /
        pd.Series(
            range(1, len(correct) + 1),
            index=correct.index
        )
        * 100
    )

    trend_df = pd.DataFrame({

        "Prediction Number":
            range(1, len(live_df) + 1),

        "Accuracy":
            cumulative_accuracy.values
    })

    st.line_chart(
        trend_df.set_index(
            "Prediction Number"
        )
    )

    st.caption(
        "This chart shows cumulative live accuracy "
        "as actual outcomes are recorded."
    )

else:

    st.info(
        "Record actual outcomes to generate "
        "the live accuracy trend."
    )


# =========================================================
# 17. TEST VS LIVE PERFORMANCE
# =========================================================

st.header("⚖️ Test vs Live Performance")

if len(st.session_state.live_records) > 0:

    live_df = pd.DataFrame(
        st.session_state.live_records
    )

    y_true = live_df["Actual"]

    y_pred = live_df["Prediction"]


    # =====================================================
    # LIVE METRICS
    # =====================================================

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


    # =====================================================
    # TEST METRICS
    # =====================================================

    test_accuracy = performance["accuracy"]

    test_report = performance[
        "classification_report"
    ]


    test_precision = test_report.get(
        "1",
        test_report.get(1)
    )["precision"]

    test_recall = test_report.get(
        "1",
        test_report.get(1)
    )["recall"]

    test_f1 = test_report.get(
        "1",
        test_report.get(1)
    )["f1-score"]


    # =====================================================
    # COMPARISON TABLE
    # =====================================================

    comparison_df = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Test Data": [
            test_accuracy * 100,
            test_precision * 100,
            test_recall * 100,
            test_f1 * 100
        ],

        "Live Data": [
            live_accuracy * 100,
            live_precision * 100,
            live_recall * 100,
            live_f1 * 100
        ]
    })


    comparison_df["Test Data"] = (
        comparison_df["Test Data"]
        .round(2)
    )

    comparison_df["Live Data"] = (
        comparison_df["Live Data"]
        .round(2)
    )


    st.dataframe(
        comparison_df,
        use_container_width=True
    )

else:

    st.info(
        "Record actual outcomes to compare "
        "test and live performance."
    )


# =========================================================
# 18. PREDICTION CONFIDENCE DISTRIBUTION
# =========================================================

st.header("📊 Prediction Confidence Distribution")

if len(st.session_state.live_records) > 0:

    confidence_df = pd.DataFrame(
        st.session_state.live_records
    ).copy()


    if "Confidence" in confidence_df.columns:

        confidence_df["Confidence"] = (
            pd.to_numeric(
                confidence_df["Confidence"],
                errors="coerce"
            )
        )

        confidence_df = (
            confidence_df
            .dropna(
                subset=["Confidence"]
            )
        )


        # =================================================
        # CONFIDENCE CATEGORIES
        # =================================================

        high = (
            confidence_df["Confidence"] >= 80
        ).sum()

        medium = (
            (
                confidence_df["Confidence"] >= 60
            )
            &
            (
                confidence_df["Confidence"] < 80
            )
        ).sum()

        low = (
            confidence_df["Confidence"] < 60
        ).sum()


        confidence_summary = pd.DataFrame({

            "Confidence Level": [
                "High (80%+)",
                "Medium (60-79%)",
                "Low (<60%)"
            ],

            "Number of Predictions": [
                high,
                medium,
                low
            ]
        })


        st.bar_chart(
            confidence_summary.set_index(
                "Confidence Level"
            )
        )


        # =================================================
        # AVERAGE CONFIDENCE
        # =================================================

        average_confidence = (
            confidence_df["Confidence"].mean()
        )


        st.metric(
            "Average Prediction Confidence",
            f"{average_confidence:.2f}%"
        )


        # =================================================
        # CONFIDENCE STATISTICS
        # =================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "High Confidence",
                high
            )

        with col2:

            st.metric(
                "Medium Confidence",
                medium
            )

        with col3:

            st.metric(
                "Low Confidence",
                low
            )

    else:

        st.info(
            "Confidence data is not available "
            "in live records."
        )

else:

    st.info(
        "Make live predictions and record actual "
        "outcomes to see the confidence distribution."
    )


# =========================================================
# 19. PREDICTION ANALYTICS DASHBOARD
# =========================================================

st.header("📊 Prediction Analytics Dashboard")

if len(st.session_state.live_records) > 0:

    analytics_df = pd.DataFrame(
        st.session_state.live_records
    )


    # =====================================================
    # BASIC STATISTICS
    # =====================================================

    total_predictions = len(
        analytics_df
    )

    correct_predictions = (
        analytics_df["Prediction"]
        ==
        analytics_df["Actual"]
    ).sum()

    incorrect_predictions = (
        analytics_df["Prediction"]
        !=
        analytics_df["Actual"]
    ).sum()

    survival_predictions = (
        analytics_df["Prediction"] == 1
    ).sum()

    death_predictions = (
        analytics_df["Prediction"] == 0
    ).sum()

    average_confidence = (
        analytics_df["Confidence"].mean()
    )


    # =====================================================
    # DASHBOARD METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Predictions",
            total_predictions
        )

    with col2:

        st.metric(
            "Correct Predictions",
            correct_predictions
        )

    with col3:

        st.metric(
            "Incorrect Predictions",
            incorrect_predictions
        )

    with col4:

        st.metric(
            "Average Confidence",
            f"{average_confidence:.2f}%"
        )


    # =====================================================
    # PREDICTION DISTRIBUTION
    # =====================================================

    st.subheader("🚢 Prediction Distribution")

    prediction_distribution = pd.DataFrame({

        "Outcome": [
            "Predicted Survived",
            "Predicted Did Not Survive"
        ],

        "Count": [
            survival_predictions,
            death_predictions
        ]
    })


    st.bar_chart(
        prediction_distribution.set_index(
            "Outcome"
        )
    )


    # =====================================================
    # CORRECT VS INCORRECT
    # =====================================================

    st.subheader(
        "🎯 Prediction Accuracy Distribution"
    )

    accuracy_distribution = pd.DataFrame({

        "Result": [
            "Correct",
            "Incorrect"
        ],

        "Count": [
            correct_predictions,
            incorrect_predictions
        ]
    })


    st.bar_chart(
        accuracy_distribution.set_index(
            "Result"
        )
    )

else:

    st.info(
        "Record actual outcomes to generate "
        "the Prediction Analytics Dashboard."
    )


# =========================================================
# 20. FEATURE IMPORTANCE
# =========================================================

st.header("📊 Feature Importance")

feature_importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": model.feature_importances_
})


feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)


feature_importance_df["Importance (%)"] = (
    feature_importance_df["Importance"] * 100
).round(2)


# =========================================================
# FEATURE IMPORTANCE CHART
# =========================================================

st.subheader("🔍 Most Important Features")

st.bar_chart(
    feature_importance_df.set_index(
        "Feature"
    )["Importance (%)"]
)


# =========================================================
# FEATURE IMPORTANCE TABLE
# =========================================================

st.subheader("📋 Feature Importance Details")

st.dataframe(
    feature_importance_df[
        [
            "Feature",
            "Importance (%)"
        ]
    ],
    use_container_width=True
)


# =========================================================
# MOST IMPORTANT FEATURE
# =========================================================

top_feature = feature_importance_df.iloc[0]

st.success(
    f"⭐ Most Important Feature: "
    f"**{top_feature['Feature']}** "
    f"({top_feature['Importance (%)']:.2f}%)"
)