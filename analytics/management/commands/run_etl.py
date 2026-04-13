"""
Management command: Run the full ETL pipeline
Usage: python manage.py run_etl
       python manage.py run_etl --verbose
"""

import time
import pandas as pd
import numpy as np
from django.db import connection
from django.core.management.base import BaseCommand, CommandError
import logging

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════
# STEP 1: EXTRACT
# Pull every transaction row from dvdrental source DB
# ══════════════════════════════════════════════════════
def extract_source_data():
    query = """
    SELECT
        c.customer_id,
        c.first_name || ' ' || c.last_name  AS full_name,
        c.email,
        c.active,
        c.create_date,
        ct.city,
        co.country,
        r.rental_id,
        r.rental_date,
        r.return_date,
        p.payment_id,
        p.amount                             AS payment_amount,
        p.payment_date,
        f.film_id,
        f.title                              AS film_title,
        cat.name                             AS film_category,
        f.rental_rate,
        f.rental_duration,
        f.rating
    FROM customer c
    JOIN address        a   ON c.address_id   = a.address_id
    JOIN city           ct  ON a.city_id      = ct.city_id
    JOIN country        co  ON ct.country_id  = co.country_id
    JOIN rental         r   ON c.customer_id  = r.customer_id
    JOIN payment        p   ON r.rental_id    = p.rental_id
    JOIN inventory      i   ON r.inventory_id = i.inventory_id
    JOIN film           f   ON i.film_id      = f.film_id
    JOIN film_category  fc  ON f.film_id      = fc.film_id
    JOIN category       cat ON fc.category_id = cat.category_id
    ORDER BY c.customer_id, p.payment_date;
    """
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        rows    = cursor.fetchall()

    df = pd.DataFrame(rows, columns=columns)

    df['payment_amount'] = df['payment_amount'].astype(float)
    df['rental_rate']    = df['rental_rate'].astype(float)
  
    df['payment_date'] = pd.to_datetime(df['payment_date'])
    df['rental_date']  = pd.to_datetime(df['rental_date'])

    logger.info(f"[ETL Extract] {len(df):,} rows | {df['customer_id'].nunique()} customers")
    return df

# ══════════════════════════════════════════════════════
# STEP 2: TRANSFORM
# Collapse all rows into one RFM row per customer
# ══════════════════════════════════════════════════════
def transform_rfm(df):
    snapshot_date = df['payment_date'].max().date()     # 2007-05-14
 
    rfm = df.groupby('customer_id').agg(
        full_name      = ('full_name',      'first'),
        email          = ('email',          'first'),
        city           = ('city',           'first'),
        country        = ('country',        'first'),
        active         = ('active',         'first'),
        create_date    = ('create_date',    'first'),
        last_purchase  = ('payment_date',   'max'),     # date of the last transaction
        first_purchase = ('payment_date',   'min'),     # date of the first transaction
        frequency      = ('rental_id',      'nunique'), # Count unique rentals
        total_payments = ('payment_amount', 'sum'),
        payment_count  = ('payment_id',     'nunique'), # count unique payments
        rental_count   = ('rental_id',      'nunique'),
        avg_payment    = ('payment_amount', 'mean'),
    ).reset_index()

    rfm['last_purchase']  = pd.to_datetime(rfm['last_purchase']).dt.date
    rfm['first_purchase'] = pd.to_datetime(rfm['first_purchase']).dt.date

    rfm['recency_days']         = (snapshot_date - rfm['last_purchase']).apply(lambda x: x.days)
    rfm['customer_tenure_days'] = (snapshot_date - rfm['first_purchase']).apply(lambda x: x.days)
    rfm['monetary_avg']         = rfm['avg_payment'].round(2)
    rfm['snapshot_date']        = snapshot_date
    rfm['month_year']           = snapshot_date.strftime('%Y-%m')

    rfm = rfm[rfm['frequency'] > 0].copy()  
    
    p33 = rfm['total_payments'].quantile(0.33)
    p66 = rfm['total_payments'].quantile(0.66) 

    def assign_segment(val):
        if   val >= p66: return 'High'
        elif val >= p33: return 'Medium'
        else:            return 'Low'

    rfm['clv_segment']   = rfm['total_payments'].apply(assign_segment)
    rfm['clv_predicted'] = 0.0 

    logger.info(
        f"[ETL Transform] {len(rfm)} customers | "
        f"p33=${p33:.2f} p66=${p66:.2f} | "
        f"snapshot={snapshot_date}"
    )
    return rfm, snapshot_date, float(p33), float(p66)


"""
Imagine the customer Eleanor Hunt:
Raw data from the extract (multiple rows):
customer_id=1 | payment_date=2007-02-14 | amount=2.99 | rental_id=101
customer_id=1 | payment_date=2007-03-21 | amount=4.99 | rental_id=205
customer_id=1 | payment_date=2007-05-01 | amount=5.99 | rental_id=387
... (45 rows total)

After Transformation (1 row):
last_purchase  = 2007-05-01  → recency_days = 14 - 01 = 13 days
first_purchase = 2007-02-14  → tenure_days  = many days
frequency      = 45 unique rentals
total_payments = sum of all amounts = $211.55
monetary_avg   = $211.55 / 45 = $4.70
clv_segment    = High (because $211.55 > p66 $110.39)
clv_predicted  = 0.0 (not yet ML-processed)
"""
    
# ══════════════════════════════════════════════════════
# STEP 3: LOAD
# Insert/update OLAP star schema tables
# ══════════════════════════════════════════════════════
def load_to_olap(rfm_df, snapshot_date):
    from analytics.models import DimCustomer, DimTime, FactCustomerCLV

    dt_obj, _ = DimTime.objects.get_or_create(
        full_date=snapshot_date,
        defaults={
            'day_of_week': snapshot_date.strftime('%A'),
            'day_num':     snapshot_date.day,
            'week_num':    snapshot_date.isocalendar()[1],
            'month_num':   snapshot_date.month,
            'month_name':  snapshot_date.strftime('%B'),
            'quarter':     (snapshot_date.month - 1) // 3 + 1,
            'year':        snapshot_date.year,
            'is_weekend':  snapshot_date.weekday() >= 5,
        }
    )

    created = 0
    
    for _, row in rfm_df.iterrows():
        cust, _ = DimCustomer.objects.update_or_create(
            customer_id=int(row['customer_id']),
            defaults={
                'full_name':   row['full_name'],
                'email':       row['email'],
                'city':        row.get('city', ''),
                'country':     row.get('country', ''),
                'active':      bool(row['active']),
                'create_date': row['create_date'],
            }
        )

        FactCustomerCLV.objects.update_or_create(
            customer=cust,
            time=dt_obj,
            month_year=row['month_year'],
            defaults={
                'total_payments':       float(row['total_payments']),
                'payment_count':        int(row['payment_count']),
                'rental_count':         int(row['rental_count']),
                'avg_payment_amount':   float(row['monetary_avg']),
                'recency_days':         int(row['recency_days']),
                'frequency':            int(row['frequency']),
                'monetary_avg':         float(row['monetary_avg']),
                'customer_tenure_days': int(row['customer_tenure_days']),
                'clv_predicted':        float(row['clv_predicted']),
                'clv_segment':          row['clv_segment'],
                'snapshot_date':        snapshot_date,
            }
        )
        created += 1

    logger.info(f"[ETL Load] {created} fact rows written to OLAP")
    return created


# ══════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════
def run_full_etl():
    try:
        logger.info("[ETL] Pipeline started")
        raw_df                          = extract_source_data()
        rfm_df, snapshot_date, p33, p66 = transform_rfm(raw_df)
        loaded                          = load_to_olap(rfm_df, snapshot_date)

        return {
            'success':               True,
            'records_extracted':     len(raw_df),
            'customers_transformed': len(rfm_df),
            'facts_loaded':          loaded,
            'snapshot_date':         str(snapshot_date),
            'p33':                   round(p33, 2),
            'p66':                   round(p66, 2),
        }
    except Exception as e:
        logger.error(f"[ETL] Pipeline failed: {e}")
        return {'success': False, 'error': str(e)}


class Command(BaseCommand):
    help = 'Run the full ETL pipeline: Extract from dvdrental → Transform RFM → Load to OLAP star schema'

    def add_arguments(self, parser):
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed step-by-step output',
        )

    def handle(self, *args, **options):
        verbose = options['verbose']

        self.stdout.write(self.style.HTTP_INFO(
            '\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
        )
        self.stdout.write(self.style.HTTP_INFO(
            '   CLV Analytics — ETL Pipeline')
        )
        self.stdout.write(self.style.HTTP_INFO(
            '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n')
        )

        start_time = time.time()

        # Step 1
        self.stdout.write('  [1/3] Extracting data from dvdrental source tables...')
        if verbose:
            self.stdout.write(self.style.WARNING(
                '        → Joining: customer, rental, payment, address, city, country, film, category'
            ))

        # Step 2
        self.stdout.write('  [2/3] Transforming: computing RFM features per customer...')
        if verbose:
            self.stdout.write(self.style.WARNING(
                '        → Recency, Frequency, Monetary avg, Tenure, Payment count'
            ))

        # Step 3
        self.stdout.write('  [3/3] Loading into OLAP star schema...')
        if verbose:
            self.stdout.write(self.style.WARNING(
                '        → Upserting: dim_customer, dim_time, fact_customer_clv'
            ))

        # Run the actual pipeline
        result = run_full_etl()

        elapsed = round(time.time() - start_time, 2)

        if result['success']:
            self.stdout.write('')
            self.stdout.write(self.style.SUCCESS(
                f'  ✓ ETL completed in {elapsed}s'
            ))
            self.stdout.write(self.style.SUCCESS(
                f'    Records extracted  : {result["records_extracted"]:,}'
            ))
            self.stdout.write(self.style.SUCCESS(
                f'    Customers processed: {result["customers_transformed"]:,}'
            ))
            self.stdout.write(self.style.SUCCESS(
                f'    Fact rows loaded   : {result["facts_loaded"]:,}'
            ))
            self.stdout.write(self.style.SUCCESS(
                f'    Snapshot date      : {result["snapshot_date"]}'
            ))
            self.stdout.write(self.style.HTTP_INFO(
                '\n  Next step: python manage.py train_model\n'
            ))
        else:
            raise CommandError(f'ETL failed: {result["error"]}')