-- ============================================
-- DIMENSION: Customer
-- ============================================
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_key    SERIAL PRIMARY KEY,
    customer_id     INTEGER UNIQUE NOT NULL,
    full_name       VARCHAR(100),
    email           VARCHAR(100),
    city            VARCHAR(50),
    country         VARCHAR(50),
    active          BOOLEAN,
    create_date     DATE
);

-- ============================================
-- DIMENSION: Time
-- ============================================
CREATE TABLE IF NOT EXISTS dim_time (
    time_key        SERIAL PRIMARY KEY,
	full_date    	DATE UNIQUE NOT NULL,  -- snapshot date
    day_of_week  	VARCHAR(10),           -- "Monday", "Tuesday", dll
    day_num      	INTEGER,               -- 1-31
    week_num     	INTEGER,               -- Which week of the year?
    month_num    	INTEGER,               -- 1-12
    month_name  	VARCHAR(15),           -- "January", dll
    quarter      	INTEGER,               -- 1-4
    year            INTEGER,
    is_weekend      BOOLEAN
);

-- ============================================
-- ML Table
-- ============================================
CREATE TABLE ml_model_log (
    id SERIAL PRIMARY KEY,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    algorithm VARCHAR(50),
    mae FLOAT,
    rmse FLOAT,
    r2_score FLOAT,
    n_samples INTEGER,
    model_path VARCHAR(255),
    notes TEXT
);
-- ============================================
-- FACT TABLE: Customer CLV
-- ============================================
CREATE TABLE IF NOT EXISTS fact_customer_clv (
    fact_id             SERIAL PRIMARY KEY,
    customer_key        INTEGER REFERENCES dim_customer(customer_key),
    time_key            INTEGER REFERENCES dim_time(time_key),
    
    -- Transactional metrics
    total_payments      NUMERIC(10,2),
    payment_count       INTEGER,
    rental_count        INTEGER,
    avg_payment_amount  NUMERIC(6,2),
    
    -- RFM metrics (core for CLV)
    recency_days        INTEGER,   -- Days since last rental
    frequency           INTEGER,   -- Total number of rentals
    monetary_avg        NUMERIC(6,2), -- Average spend per rental
    
    -- CLV metrics
    customer_tenure_days INTEGER,
    clv_predicted       NUMERIC(10,2),  -- ML predicted CLV
    clv_segment         VARCHAR(20),    -- High / Medium / Low
    
    -- Time dimension
    snapshot_date       DATE,
    month_year          VARCHAR(7)      -- e.g. '2024-01'
);

-- ============================================
-- OLAP AGGREGATE VIEW: Monthly Revenue per Segment
-- ============================================
CREATE OR REPLACE VIEW vw_monthly_clv_segment AS
SELECT
    dt.month_name,
    dt.year,
    dt.month_num,
    fc.clv_segment,
    COUNT(DISTINCT fc.customer_key) AS customer_count,
    SUM(fc.total_payments)          AS total_revenue,
    AVG(fc.clv_predicted)           AS avg_predicted_clv,
    AVG(fc.recency_days)            AS avg_recency,
    AVG(fc.frequency)               AS avg_frequency
FROM fact_customer_clv fc
JOIN dim_time dt ON fc.time_key = dt.time_key
GROUP BY dt.month_name, dt.year, dt.month_num, fc.clv_segment
ORDER BY dt.year, dt.month_num;

-- ============================================
-- OLAP AGGREGATE VIEW: Top Customers by CLV
-- ============================================
CREATE OR REPLACE VIEW vw_top_customers AS
SELECT
    dc.customer_id,
    dc.full_name,
    dc.email,
    dc.city,
    dc.country,
    SUM(fc.total_payments)    AS lifetime_revenue,
    MAX(fc.clv_predicted)     AS predicted_clv,
    MAX(fc.frequency)         AS total_rentals,
    MIN(fc.recency_days)      AS last_seen_days_ago,
    MAX(fc.clv_segment)       AS clv_segment
FROM fact_customer_clv fc
JOIN dim_customer dc ON fc.customer_key = dc.customer_key
GROUP BY dc.customer_id, dc.full_name, dc.email, dc.city, dc.country
ORDER BY predicted_clv DESC;


-- Check range payment_date
SELECT 
    MIN(payment_date) AS earliest_payment,
    MAX(payment_date) AS latest_payment,
    MAX(payment_date) - MIN(payment_date) AS total_range
FROM payment;

-- Check range create_date customer
SELECT 
    MIN(create_date) AS earliest_customer,
    MAX(create_date) AS latest_customer,
    MAX(create_date) - MIN(create_date) AS total_range
FROM customer;


select * from dim_customer
select * from dim_time
select * from fact_customer_clv
select * from payment
select * from vw_top_customers
select * from vw_monthly_clv_segment