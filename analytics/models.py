from django.db import models

class DimCustomer(models.Model):
    customer_key = models.AutoField(primary_key=True)
    customer_id  = models.IntegerField(unique=True)
    full_name    = models.CharField(max_length=100)
    email        = models.CharField(max_length=100)
    city         = models.CharField(max_length=50, blank=True)
    country      = models.CharField(max_length=50, blank=True)
    active       = models.BooleanField(default=True)
    create_date  = models.DateField(null=True)

    class Meta:
        db_table = 'dim_customer'
        managed  = False       

    def __str__(self):
        return self.full_name


class DimTime(models.Model):
    time_key    = models.AutoField(primary_key=True)   
    full_date   = models.DateField(unique=True)
    day_of_week = models.CharField(max_length=10)
    day_num     = models.IntegerField()
    week_num    = models.IntegerField()
    month_num   = models.IntegerField()
    month_name  = models.CharField(max_length=15)
    quarter     = models.IntegerField()
    year        = models.IntegerField()
    is_weekend  = models.BooleanField()

    class Meta:
        db_table = 'dim_time'
        managed  = False        

    def __str__(self):
        return str(self.full_date)



class FactCustomerCLV(models.Model):
    fact_id     = models.AutoField(primary_key=True)
    customer    = models.ForeignKey(
        DimCustomer,
        on_delete=models.CASCADE,
        db_column='customer_key'
    )
    time        = models.ForeignKey(
        DimTime,
        on_delete=models.CASCADE,
        db_column='time_key'
    )

    total_payments     = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    payment_count      = models.IntegerField(default=0)
    rental_count       = models.IntegerField(default=0)
    avg_payment_amount = models.DecimalField(max_digits=6, decimal_places=2, default=0)

    recency_days          = models.IntegerField(default=0)
    frequency             = models.IntegerField(default=0)
    monetary_avg          = models.DecimalField(max_digits=6, decimal_places=2, default=0)

    customer_tenure_days  = models.IntegerField(default=0)
    clv_predicted         = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    clv_segment           = models.CharField(max_length=20, default='Low')

    snapshot_date = models.DateField()
    month_year    = models.CharField(max_length=7)

    class Meta:
        db_table = 'fact_customer_clv'
        managed  = False

    def __str__(self):
        return f"{self.customer.full_name} — {self.month_year}"


class MLModelLog(models.Model):
    trained_at  = models.DateTimeField(auto_now_add=True)
    algorithm   = models.CharField(max_length=50)
    mae         = models.FloatField()
    rmse        = models.FloatField()
    r2_score    = models.FloatField()
    n_samples   = models.IntegerField()
    model_path  = models.CharField(max_length=255)
    notes       = models.TextField(blank=True)

    class Meta:
        db_table = 'ml_model_log'
        managed  = True        
        ordering = ['-trained_at']

    def __str__(self):
        return f"{self.algorithm} @ {self.trained_at}"