"""fakenewsdetect URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from xml.etree.ElementInclude  import include
from django.contrib import admin
from django.urls import path
from app import views
from django.urls import path, re_path
urlpatterns = [
    path('', views.landing, name='root'),
    path('admin/', admin.site.urls),
    path('home/', views.home, name='home'),
	path('nvb/', views.nvb, name='nvb'),
	path('pac/', views.pac, name='pac'),
	path('svm/', views.svm, name='svm'),
	path('dec/', views.dec, name='dec'),
	path('randomf/', views.randomf, name='randomf'),
	path('mnb/', views.mnb, name='mnb'),
	path('graph/', views.graph, name='graph'),
	path('accuracy/', views.accuracy, name='accuracy'),
	path('loginCheck/', views.loginCheck, name='loginCheck'),
	path('reg/', views.reg, name='reg'),
	path('login/', views.login, name='login'),
	path('save/', views.save, name='save'),
	path('logout/', views.logout, name='logout'),
]
