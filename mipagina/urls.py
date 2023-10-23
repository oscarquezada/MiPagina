from django.urls import path, re_path  # Importa la función re_path para usar expresiones regulares
from . import views

urlpatterns = [
    path('', views.inicio, name="inicio"),
]