from django.contrib import admin
from .models import DimCustomer, DimTime, FactCustomerCLV, MLModelLog

@admin.register(FactCustomerCLV)
class FactCLVAdmin(admin.ModelAdmin):
    list_display  = ['customer', 'clv_predicted', 'clv_segment', 'recency_days','frequency', 'monetary_avg', 'customer_tenure_days', 'payment_count', 'month_year']
    list_filter   = ['clv_segment', 'month_year']
    search_fields = ['customer__full_name', 'customer__email']

@admin.register(DimCustomer)
class CustomerAdmin(admin.ModelAdmin):
    list_display  = ['customer_id', 'full_name', 'email', 'city', 'country', 'active']
    search_fields = ['full_name', 'email']

@admin.register(MLModelLog)
class MLLogAdmin(admin.ModelAdmin):
    list_display = ['trained_at', 'algorithm', 'mae', 'r2_score', 'n_samples']
    readonly_fields = ['trained_at']

admin.site.register(DimTime)