from django.conf.urls import url
from . import views

urlpatterns = [
                url(r'^$', views.index, name='index'),
                url(r'^wells/', views.wells, name='wells'),
                url(r'^home/', views.home, name='home')
              ]
