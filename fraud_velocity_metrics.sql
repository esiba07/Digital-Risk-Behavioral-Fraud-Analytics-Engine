-- =====================================================================
-- Project: Digital Risk & Behavioral Fraud Analytics Engine
-- Query Purpose: Calculate rolling 1-hour transaction velocity and 
--                spending deviations by card profile to flag anomalies.
-- =====================================================================

WITH card_historical_baseline AS (
    -- Step 1: Compute historical mean and standard deviation of transaction amounts per card
    SELECT
        card1,
        AVG(TransactionAmt) AS historical_avg_amt,
        STDDEV(TransactionAmt) AS historical_std_amt,
        COUNT(TransactionID) AS total_historical_txns
    FROM
        `your_project.fraud_dataset.train_transaction`
    GROUP BY
        card1
),

rolling_velocity_window AS (
    -- Step 2: Track high-frequency transaction counts and total volume in a sliding time window
    SELECT
        t.TransactionID,
        t.card1,
        t.TransactionDT,
        t.TransactionAmt,
        COUNT(t.TransactionID) OVER (
            PARTITION BY t.card1 
            ORDER BY t.TransactionDT 
            RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
        ) AS txns_in_past_hour,
        SUM(t.TransactionAmt) OVER (
            PARTITION BY t.card1 
            ORDER BY t.TransactionDT 
            RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
        ) AS volume_in_past_hour
    FROM
        `your_project.fraud_dataset.train_transaction` t
)

-- Step 3: Combine baseline stats and velocity metrics to flag high-risk anomalies
SELECT
    v.TransactionID,
    v.card1,
    v.TransactionDT,
    v.TransactionAmt,
    v.txns_in_past_hour,
    v.volume_in_past_hour,
    b.historical_avg_amt,
    -- Calculate ratio deviation from the card's typical spend
    SAFE_DIVIDE(v.TransactionAmt, b.historical_avg_amt) AS amt_to_historical_mean_ratio,
    -- Risk Flag: Flag if more than 3 transactions occur within 1 hour OR amount is 5x higher than average
    CASE 
        WHEN v.txns_in_past_hour > 3 THEN 1
        WHEN v.TransactionAmt > (b.historical_avg_amt + (5 * COALESCE(b.historical_std_amt, 0))) THEN 1
        ELSE 0
    END AS high_risk_velocity_flag
FROM
    rolling_velocity_window v
LEFT JOIN
    card_historical_baseline b
    ON v.card1 = b.card1
ORDER BY
    v.TransactionDT DESC;