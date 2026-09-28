import pandas as pd
import numpy as np

def load_and_merge_data(transaction_path, identity_path):
    """Loads and merges transaction and identity datasets on TransactionID."""
    print("Loading datasets...")
    train_trans = pd.read_csv(transaction_path)
    train_ident = pd.read_csv(identity_path)
    
    # Merge on TransactionID
    df = train_trans.merge(train_ident, on='TransactionID', how='left')
    return df

def engineer_features(df):
    """
    Engineers behavioral, velocity, and aggregate features 
    while avoiding target leakage.
    """
    print("Engineering features...")
    df = df.copy()
    
    # 1. Transaction Amount Log Transformation (handles skewness)
    df['LogTransactionAmt'] = np.log1p(df['TransactionAmt'])
    
    # 2. Email domain risk flag (e.g., matching missing or anonymous domains)
    df['is_p_email_missing'] = df['P_emaildomain'].isna().astype(int)
    
    # 3. Card-level historical aggregations (Velocity & Volume metrics)
    # Group by card1 to see how much a specific card profile typically spends
    card_stats = df.groupby('card1')['TransactionAmt'].agg(['mean', 'std', 'count']).reset_index()
    card_stats.columns = ['card1', 'card1_amt_mean', 'card1_amt_std', 'card1_transaction_count']
    df = df.merge(card_stats, on='card1', how='left')
    
    # Calculate deviation from the card's historical average amount
    df['amt_to_card_mean_ratio'] = df['TransactionAmt'] / (df['card1_amt_mean'] + 1e-5)
    
    # 4. Device and Network Risk Indicators (from identity table)
    if 'DeviceType' in df.columns:
        df['DeviceType_missing'] = df['DeviceType'].isna().astype(int)
        
    return df

if __name__ == "__main__":
    # Example execution flow:
    # df = load_and_merge_data('train_transaction.csv', 'train_identity.csv')
    # processed_df = engineer_features(df)
    print("Feature engineering template ready.")