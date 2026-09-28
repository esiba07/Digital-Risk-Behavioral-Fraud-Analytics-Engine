import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report

def train_fraud_model(df):
    """
    Trains a LightGBM fraud detection model using time-based splitting 
    to prevent data leakage.
    """
    print("Preparing features and target...")
    
    # Define feature columns (dropping IDs, raw targets, and non-numeric fields for simplicity)
    drop_cols = ['TransactionID', 'isFraud', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo']
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df['isFraud']
    
    # Use TimeSeriesSplit to respect chronological flow (crucial for fraud models)
    tscv = TimeSeriesSplit(n_splits=3)
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        print(f"\n--- Training Fold {fold + 1} ---")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Initialize LightGBM Classifier with parameters tuned for imbalanced data
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            scale_pos_weight=10,  # Handles class imbalance for rare fraud events
            random_state=42,
            n_jobs=-1
        )
        
        # Train model with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        # Evaluate performance
        preds_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds_proba)
        print(f"Validation AUC-ROC for Fold {fold + 1}: {auc:.4f}")
        
    return model

if __name__ == "__main__":
    # Integration point with your feature-engineered dataframe:
    # model = train_fraud_model(processed_df)
    print("Model training template ready.")