# Customer Lifetime Value Prediction System
A full-stack data analytics application that predicts Customer Lifetime Value (CLV) using machine learning, built on top of the PostgreSQL DVD Rental database. The system implements a complete data pipeline from raw OLTP transactions to an OLAP star schema, with an interactive Django web dashboard.

---

## Overview

The core question this system answers: **How much total revenue will each customer generate over their lifetime?**

Using RFM (Recency, Frequency, Monetary) analysis as the feature foundation, the system trains a Gradient Boosting model that predicts individual customer CLV, segments customers into value tiers, and surfaces actionable insights through an interactive dashboard.

**Stack:** Python · Django · PostgreSQL · scikit-learn · Plotly · pandas

---

## Features

**ETL Pipeline**
- Extracts ~14,000 transaction rows from 9 joined dvdrental tables
- Transforms into RFM metrics per customer (599 customers)
- Loads into OLAP star schema with upsert pattern 
- Dynamic segment thresholds using data percentiles (p33/p66) 

**ML Model**
- Compares 3 algorithms: Linear Regression, Random Forest, Gradient Boosting
- Trains with 80/20 chronological split + 5-fold TimeSeriesSplit cross-validation
- log1p target transformation to handle right-skewed CLV distribution
- Saves trained pipeline and thresholds as `.pkl` artifacts
- Logs every training run to `ml_model_log` table

**Dashboard**
- KPI cards: total revenue, avg predicted CLV, customer count, avg recency
- Revenue by segment bar chart, customer distribution donut chart
- CLV vs Recency scatter plot by segment
- Top 10 customers table by predicted CLV

**Prediction Form**
- Real-time CLV prediction with ±15% confidence interval
- R/F/M score breakdown (Good / Fair / Poor per dimension)
- Every prediction saved to `prediction_log` table
- Prediction history chart (last 20 predictions)

---

## Prerequisites

- Python 3.9+
- PostgreSQL with the [dvdrental database](https://www.postgresqltutorial.com/postgresql-getting-started/load-postgresql-sample-database/) loaded
- pip

---

## Getting Started

**1. Clone and set up environment**
```bash
git clone https://github.com/fasyanbls/Customer-Lifetime-Value-Prediction.git
cd clv_project
python -m venv django_venv
# Windows
django_venv\Scripts\activate
# Mac/Linux
source django_venv/bin/activate

pip install -r requirements.txt
```

**2. Configure database**

Edit `clv_project/settings.py`:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'dvdrental',
        'USER': 'postgres',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

TIME_ZONE = 'Asia/Jakarta'  # adjust to your timezone
```

**3. Create OLAP tables**

Run `CLVPROJECT.sql` in pgAdmin or psql against your dvdrental database.

**4. Apply Django migrations**
```bash
python manage.py migrate
python manage.py createsuperuser
```

**5. Run the full data pipeline**
```bash
# Extract, transform, load from dvdrental → OLAP
python manage.py run_etl

# Compare models and train the best one
python manage.py train_model --compare --show-importance

# Fill clv_predicted for all customers in the fact table
python manage.py predict_customers
```

**6. Start the server**
```bash
python manage.py runserver
```

Open `http://127.0.0.1:8000/`

---

## Project Structure

```
clv_project/
├── analytics/
│   ├── etl.py                  # ETL logic (extract, transform, load)
│   ├── ml_model.py             # ML training, comparison, prediction
│   ├── models.py               # Django ORM models (OLAP tables)
│   ├── views.py                # Page logic and chart generation
│   ├── urls.py
│   ├── admin.py
│   ├── templates/analytics/
│   │   ├── base.html           # Shared layout (vintage cream theme)
│   │   ├── dashboard.html
│   │   ├── etl.html
│   │   ├── model.html
│   │   └── prediction.html
│   └── management/commands/
│       ├── run_etl.py
│       ├── train_model.py
│       └── predict_customers.py
├── models/
│   ├── clv_model.pkl           # Trained model artifact
│   └── thresholds.pkl          # Segment thresholds (p33, p66)
├── CLVPROJECT.sql              # OLAP schema DDL
├── requirements.txt
└── manage.py
```

---

## Model Performance

The Gradient Boosting model is evaluated on a held-out 20% test set (chronological split):

| Metric | Meaning | Typical result |
|---|---|---|
| R² | Variance in CLV explained by the model | ~0.99 |
| MAE | Average prediction error in dollars | ~$0.83–$1.20 |
| CV R² | Cross-validated R² (5-fold TimeSeriesSplit) | ~0.96–0.97 |

The high R² reflects that RFM features are strong predictors of total spend in transactional data. Cross-validation uses TimeSeriesSplit to prevent data leakage — always training on past data and testing on future data.

---

## Business Insights

- **High-value customers (~33%)** generate disproportionately more revenue. Prioritize retention and loyalty programs for this group.
- **Recency is a strong signal** — customers who rented within the last 50 days have significantly higher predicted CLV across all segments.
- **Frequency drives CLV more than monetary average** — customers who rent often, even at lower price points, accumulate more lifetime value than infrequent high-spenders.
- **Segment thresholds are data-driven** — the Low/Medium/High split uses 33rd and 66th percentiles of actual spend, ensuring balanced and meaningful groupings.

---

## Limitations

The DVD Rental database is a static sample dataset (data from ~2005–2007) with no new transactions. This means:
- The model is trained and tested on historical data only, it cannot be validated against truly unseen future customers
- Recency values are calculated relative to the last recorded payment date in the dataset, not today's actual date
- In a production setting, the ETL pipeline would run on a schedule (e.g. daily cron job) and the model would be retrained periodically as new data arrives

---

## Demo Application
You can view the CircleBloom demo application, including an app walkthrough and feature showcase, via the following Google Drive link:

🔗 Demo App (Google Drive): https://drive.google.com/file/d/1c_kvC3XDROVDodSc1YHXyE_tDWvLAXfV/view?usp=sharing

---

## Screenshots

*Dashboard — KPI overview and customer segment analysis*
<img width="1896" height="1005" alt="image" src="https://github.com/user-attachments/assets/cea02307-1324-4677-b574-a14ac26f7f2c" />

*CLV Prediction form with RFM breakdown and business recommendations*
<img width="1895" height="995" alt="image" src="https://github.com/user-attachments/assets/8ff73411-ef8d-40ec-bc4a-1e32dfd651ec" />

*Model comparison and actual vs predicted accuracy chart*
<img width="1893" height="994" alt="image" src="https://github.com/user-attachments/assets/816f4c5e-8290-405b-aea7-bd1990fa0c04" />

*ETL pipeline result after processing 14,000+ transactions*
<img width="1918" height="998" alt="image" src="https://github.com/user-attachments/assets/e7e281c6-40ff-45bf-918b-cc40158cd917" />

---

## Author
Fasya Nabila Salim


