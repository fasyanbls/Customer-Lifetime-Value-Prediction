"""
- compare_models()      → test 3 algorithms, pick the best
- train_clv_model()     → train the best model (Gradient Boosting)
- predict_clv()         → predict CLV for ONE customer (used by prediction form)
- predict_all_customers() → predict CLV for ALL customers and save to DB
- get_feature_importance() → which features matter most

"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model  import LinearRegression
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score

from django.conf import settings
import logging

logger = logging.getLogger(__name__)

MODEL_DIR  = os.path.join(settings.BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'clv_model.pkl')
THRESH_PATH= os.path.join(MODEL_DIR, 'thresholds.pkl')  

FEATURES = [
    'recency_days',
    'frequency',
    'monetary_avg',
    'customer_tenure_days',
    'payment_count',
]

# Load data from OLAP
def _load_olap_data():
    from analytics.models import FactCustomerCLV
    qs = FactCustomerCLV.objects.select_related('customer').values(
        'recency_days', 'frequency', 'monetary_avg',
        'customer_tenure_days', 'payment_count', 'total_payments',
        'customer__customer_id', 'customer__full_name',
    )
    return pd.DataFrame(list(qs))

# Feature engineering
def prepare_features(df):
    """
    Converts the OLAP data into X (features) and y (target).
    """
    df_clean = df.dropna(subset=FEATURES + ['total_payments']).copy()
    df_clean['total_payments'] = df_clean['total_payments'].apply(float)
    df_clean['target_log']     = np.log1p(df_clean['total_payments'])

    X = df_clean[FEATURES].astype(float)
    y = df_clean['target_log']
    return X, y, df_clean 


# Calculate segment thresholds from data
def _get_thresholds(df_clean):
    """
    Bottom 33% of customers  → Low
    Middle 33% of customers  → Medium
    Top    33% of customers  → High
    """
    p33 = float(df_clean['total_payments'].quantile(0.33))
    p66 = float(df_clean['total_payments'].quantile(0.66))
    return p33, p66


def _segment(pred_clv, p33, p66):
    if   pred_clv >= p66: return 'High'
    elif pred_clv >= p33: return 'Medium'
    else:                 return 'Low'


# COMPARE MODELS
def compare_models():
    df = _load_olap_data()
    if len(df) < 10:
        return {'success': False, 'error': 'Not enough data. Run ETL first.'}

    X, y, df_clean = prepare_features(df)
    p33, p66 = _get_thresholds(df_clean)

    split_idx     = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    tscv = TimeSeriesSplit(n_splits=5)

    candidates = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model',  LinearRegression()),
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model',  RandomForestRegressor(
                n_estimators=100, max_depth=6,
                min_samples_leaf=3, random_state=42, n_jobs=-1,
            )),
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('model',  GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                min_samples_split=5, min_samples_leaf=3,
                subsample=0.8, random_state=42,
            )),
        ]),
    }

    results = {}
    for name, pipeline in candidates.items():
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='r2')
        pipeline.fit(X_train, y_train)

        y_pred   = np.expm1(pipeline.predict(X_test))
        y_actual = np.expm1(y_test)

        mae  = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2   = r2_score(y_actual, y_pred)

        results[name] = {
            'mae':        round(mae, 2),
            'rmse':       round(rmse, 2),
            'r2':         round(r2, 4),
            'cv_r2_mean': round(float(cv_scores.mean()), 4),
            'cv_r2_std':  round(float(cv_scores.std()), 4),
        }
        logger.info(f"[COMPARE] {name}: MAE={mae:.2f}, R2={r2:.4f}")

    return {
        'success':   True,
        'results':   results,
        'n_samples': len(df_clean),
        'p33':       round(p33, 2),
        'p66':       round(p66, 2),
    }


# TRAIN THE FINAL MODEL
def train_clv_model():
    from analytics.models import MLModelLog

    df = _load_olap_data()
    if len(df) < 10:
        return {'success': False, 'error': 'Not enough data. Run ETL first.'}

    X, y, df_clean = prepare_features(df)
    p33, p66 = _get_thresholds(df_clean)

    split_idx       = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('model',  GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,   
            max_depth=4,          
            min_samples_split=5,  
            min_samples_leaf=3,   
            subsample=0.8,        
            random_state=42,      
        )),
    ])

    # Cross-validation: train on 5 different train/test splits
    tscv      = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='r2')


    # Final training
    pipeline.fit(X_train, y_train) 


    # Evaluate 
    y_pred_log = pipeline.predict(X_test)
    y_pred     = np.expm1(y_pred_log)  
    y_actual   = np.expm1(y_test)

    mae  = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2   = r2_score(y_actual, y_pred)


    # Prepare data for visualization 
    actual_list    = [round(float(v), 2) for v in list(y_actual)[:80]]
    predicted_list = [round(float(v), 2) for v in list(y_pred)[:80]]

    logger.info(f"[ML Train] MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f} CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")


    # Save model & thresholds
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump({'p33': p33, 'p66': p66}, THRESH_PATH)


    # Store model performance in database
    log = MLModelLog.objects.create(
        algorithm  = 'GradientBoostingRegressor',
        mae        = round(mae, 4),
        rmse       = round(rmse, 4),
        r2_score   = round(r2, 4),
        n_samples  = len(df_clean),
        model_path = MODEL_PATH,
        notes      = (
            f"CV R²: {cv_scores.mean():.4f}±{cv_scores.std():.4f} | "
            f"Train: {len(X_train)} | Test: {len(X_test)} | "
            f"Thresholds: p33=${p33:.2f} p66=${p66:.2f}"
        ),
    )

    return {
        'success':    True,
        'mae':        round(mae, 2),
        'rmse':       round(rmse, 2),
        'r2':         round(r2, 4),
        'cv_r2_mean': round(float(cv_scores.mean()), 4),
        'cv_r2_std':  round(float(cv_scores.std()), 4),
        'n_samples':  len(df_clean),
        'n_train':    len(X_train),
        'n_test':     len(X_test),
        'model_id':   log.id,
        'p33':        round(p33, 2),
        'p66':        round(p66, 2),
        'actual':     actual_list,
        'predicted':  predicted_list,
    }


# PREDICT ONE CUSTOMER
def predict_clv(recency_days, frequency, monetary_avg, tenure_days, payment_count):
    if not os.path.exists(MODEL_PATH):
        return {'error': 'Model not trained yet. Please train the model first.'}

    pipeline = joblib.load(MODEL_PATH)

    # Load saved thresholds 
    thresholds = joblib.load(THRESH_PATH) if os.path.exists(THRESH_PATH) else {'p33': 60, 'p66': 100}
    p33, p66   = thresholds['p33'], thresholds['p66']

    X_new = pd.DataFrame([{
        'recency_days':         float(recency_days),
        'frequency':            float(frequency),
        'monetary_avg':         float(monetary_avg),
        'customer_tenure_days': float(tenure_days),
        'payment_count':        float(payment_count),
    }])

    pred_log = pipeline.predict(X_new)[0]
    pred_clv = max(0.0, float(np.expm1(pred_log))) 

    ci_lower = round(pred_clv * 0.85, 2)
    ci_upper = round(pred_clv * 1.15, 2)
    segment  = 'High Value' if pred_clv >= p66 else ('Medium Value' if pred_clv >= p33 else 'Low Value')

    # RFM breakdown cards
    rfm_recency   = 'Good' if recency_days < 50   else ('Fair' if recency_days < 100 else 'Poor')
    rfm_frequency = 'Good' if frequency   > 25    else ('Fair' if frequency   > 10   else 'Poor')
    rfm_monetary  = 'Good' if monetary_avg > 5.0  else ('Fair' if monetary_avg > 3.0  else 'Poor')
    
    return {
        'predicted_clv': round(pred_clv, 2),
        'ci_lower':       ci_lower,
        'ci_upper':       ci_upper,
        'segment':        segment,
        'rfm_recency':    rfm_recency,
        'rfm_frequency':  rfm_frequency,
        'rfm_monetary':   rfm_monetary,
        'p33':            round(p33, 2),
        'p66':            round(p66, 2),
    }


# FEATURE IMPORTANCE
def get_feature_importance():
    if not os.path.exists(MODEL_PATH):
        return None
    pipeline   = joblib.load(MODEL_PATH)
    model      = pipeline.named_steps['model']
    importance = dict(zip(FEATURES, model.feature_importances_))
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# BATCH PREDICT ALL CUSTOMERS
def predict_all_customers():
    from analytics.models import FactCustomerCLV

    if not os.path.exists(MODEL_PATH):
        return {'error': 'Train model first. Run: python manage.py train_model'}

    pipeline   = joblib.load(MODEL_PATH)
    thresholds = joblib.load(THRESH_PATH) if os.path.exists(THRESH_PATH) else {'p33': 60, 'p66': 100}
    p33, p66   = thresholds['p33'], thresholds['p66']

    facts = list(FactCustomerCLV.objects.all())
    if not facts:
        return {'error': 'No data in OLAP. Run ETL first.'}

    rows = [{
        'recency_days':         float(f.recency_days),
        'frequency':            float(f.frequency),
        'monetary_avg':         float(f.monetary_avg),
        'customer_tenure_days': float(f.customer_tenure_days),
        'payment_count':        float(f.payment_count),
    } for f in facts]

    X_all = pd.DataFrame(rows, columns=FEATURES)
    preds = np.expm1(pipeline.predict(X_all))

    to_update = []
    for fact, pred in zip(facts, preds):
        pred = max(0.0, float(pred))
        fact.clv_predicted = round(pred, 2)
        fact.clv_segment   = _segment(pred, p33, p66)
        to_update.append(fact)

    FactCustomerCLV.objects.bulk_update(to_update, ['clv_predicted', 'clv_segment'])

    logger.info(f"[ML Predict All] Updated {len(to_update)} customers | p33=${p33:.2f} p66=${p66:.2f}")
    return {
        'success': True,
        'updated': len(to_update),
        'p33':     round(p33, 2),
        'p66':     round(p66, 2),
    }
