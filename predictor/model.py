import os
import joblib
import shap
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score, roc_curve, auc
from imblearn.over_sampling import ADASYN, SMOTE


def train_model(df, ticker):
    features = [col for col in df.columns if col not in ['Date', 'future_return', 'target']]
    X = df[features]
    y = df['target']

    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []

    os.makedirs("models/roc", exist_ok=True)
    os.makedirs("models/shap", exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # ADASYN with SMOTE fallback
        try:
            X_train_res, y_train_res = ADASYN(random_state=42).fit_resample(X_train, y_train)
        except ValueError:
            print(f"ADASYN failed for {ticker} fold {fold+1}; using SMOTE fallback.")
            X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

        model = XGBClassifier(
            objective='binary:logistic',
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            random_state=42
        )

        # Use callback for early stopping (required in xgboost >= 2.0)
        model.fit(
            X_train_res,
            y_train_res,
            eval_set=[(X_test, y_test)],
            verbose=False,
            callbacks=[xgb.callback.EarlyStopping(rounds=10)]
        )

        probas = model.predict_proba(X_test)[:, 1]
        best_thresh = 0.5
        best_f1 = 0

        for t in np.arange(0.4, 0.7, 0.01):
            preds = (probas > t).astype(int)
            f1 = f1_score(y_test, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        final_preds = (probas > best_thresh).astype(int)
        report = classification_report(y_test, final_preds, output_dict=True)
        print(f"\n{ticker} Fold {fold + 1} Report (Threshold={best_thresh:.2f}):")
        print(classification_report(y_test, final_preds))

        fold_metrics.append({
            "fold": fold + 1,
            "threshold": best_thresh,
            "f1_score": best_f1,
            "report": report
        })

        fpr, tpr, _ = roc_curve(y_test, probas)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
        plt.title(f"{ticker} ROC Curve - Fold {fold + 1}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(f"models/roc/{ticker}_fold{fold + 1}_roc.png")
        plt.close()

    # Final model on full data
    final_model = XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        random_state=42
    )
    final_model.fit(X, y)

    joblib.dump(final_model, f"models/{ticker}_model.pkl")

    # SHAP summary
    explainer = shap.TreeExplainer(final_model)
    sample_X = X.sample(min(500, len(X)), random_state=42)
    shap_values = explainer.shap_values(sample_X)

    shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    plt.title(f"{ticker} - SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(f"models/shap/{ticker}_shap_bar.png")
    plt.close()

    shap.summary_plot(shap_values, sample_X, plot_type="dot", show=False)
    plt.tight_layout()
    plt.savefig(f"models/shap/{ticker}_shap_summary.png")
    plt.close()

    return final_model, fold_metrics
