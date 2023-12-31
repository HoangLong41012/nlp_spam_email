from django.urls import path

from . import views

urlpatterns = [
    path("predict", views.index, name="index"),
]