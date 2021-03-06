"""project_natty URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from django.contrib import admin
from django.urls import path, include
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

from neural_net.views import setup_net, initiate_net, finalize_net, to_net, home_page, signup, login
urlpatterns = [
    path('admin/', admin.site.urls),
    path('signup/setup/',setup_net),
    path('signup/setup/initiate/',initiate_net),
    path('signup/setup/initiate/finalize/',finalize_net),
    path('',home_page),
    path('signup/to_net/',to_net),
    path('signup/',signup),
    path('login/',login),
    path('login/to_net/',to_net),
    path('login/setup/',setup_net),
    path('login/setup/initiate/',initiate_net),
    path('login/setup/initiate/finalize/',finalize_net),
    path('api/', include('neural_net.api.urls')),
    path('api-auth/', include('rest_framework.urls'))
]
urlpatterns += staticfiles_urlpatterns()
