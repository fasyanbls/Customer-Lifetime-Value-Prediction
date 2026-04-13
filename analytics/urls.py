from django.urls import path
from . import views

urlpatterns = [
    path('',            views.dashboard,       name='dashboard'),
    path('etl/',        views.etl_view,         name='etl'),
    path('model/',      views.model_view,       name='model'),
    path('predict/',    views.prediction_view,  name='prediction'),
]