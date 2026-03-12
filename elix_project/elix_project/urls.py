
# chatbot/urls.py

# elix_project/urls.py

from django.contrib import admin
from django.urls import path, include
from chatbot import views as chatbot_views

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Authentication URLs
    path('login/', chatbot_views.login_view, name='login'),
    path('logout/', chatbot_views.logout_view, name='logout'),
    
    # Chatbot URLs
    path('', include('chatbot.urls')),
]