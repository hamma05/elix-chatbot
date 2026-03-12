from django.contrib import admin

# Register your models here.
# chatbot/admin.py

from django.contrib import admin
from .models import Project, ChatMessage, ElixServiceCall

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = [
        'project_id', 
        'name', 
        'product_owner', 
        'status', 
        'advancement_display',
        'estimation_time',
        'created_date'
    ]
    list_filter = ['status', 'product_owner', 'created_date']
    search_fields = ['name', 'project_id', 'product_owner', 'description']
    ordering = ['-created_date']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('project_id', 'name', 'description', 'product_owner')
        }),
        ('Progress & Timeline', {
            'fields': ('status', 'advancement', 'estimation_time', 'start_date', 'end_date')
        }),
        ('Budget', {
            'fields': ('budget', 'spent'),
            'classes': ('collapse',)
        }),
    )
    
    def advancement_display(self, obj):
        return f"{obj.advancement}%"
    advancement_display.short_description = 'Progress'
    advancement_display.admin_order_field = 'advancement'


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['user', 'user_message_short', 'intent', 'project', 'timestamp']
    list_filter = ['intent', 'timestamp', 'user']
    search_fields = ['user__username', 'user_message', 'bot_response']
    date_hierarchy = 'timestamp'
    
    def user_message_short(self, obj):
        return obj.user_message[:50] + '...' if len(obj.user_message) > 50 else obj.user_message
    user_message_short.short_description = 'Message'


@admin.register(ElixServiceCall)
class ElixServiceCallAdmin(admin.ModelAdmin):
    list_display = ['service_id', 'user', 'method', 'status_code', 'success', 'called_at']
    list_filter = ['success', 'method', 'called_at']
    search_fields = ['service_id', 'service_name', 'user__username']
    date_hierarchy = 'called_at'