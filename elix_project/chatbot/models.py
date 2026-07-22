from django.db import models

# Create your models here.
# chatbot/models.py
from elix_project.settings import MCP_BASE_URL, MCP_API_KEY, MCP_WORKFLOW_SERVICE_PATH
from django.db import models
from django.contrib.auth.models import User

class Project(models.Model):
    """Store project information from database"""
    
    # Basic Info
    project_id = models.IntegerField(unique=True, help_text="Unique project identifier")
    name = models.CharField(max_length=255, help_text="Project name")
    description = models.TextField(blank=True, help_text="Project description")
    product_owner = models.CharField(max_length=255, help_text="Product owner name")
    advancement = models.IntegerField(default=0, help_text="Project progress percentage (0-100)")
    estimation_time = models.IntegerField(help_text="Estimated time in days")
    
    # Additional useful fields
    status = models.CharField(
        max_length=50,
        choices=[
            ('Planning', 'Planning'),
            ('In Progress', 'In Progress'),
            ('On Hold', 'On Hold'),
            ('Completed', 'Completed'),
            ('Cancelled', 'Cancelled'),
        ],
        default='Planning'
    )
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    budget = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    spent = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    equipe= models.CharField(max_length=255, blank=True)
    
    def __str__(self):
        return f"{self.name} (ID: {self.project_id})"
    
    @property
    def progress_percentage(self):
        """Return advancement as percentage"""
        return f"{self.advancement}%"
    
    @property
    def remaining_time(self):
        """Calculate remaining time based on advancement"""
        if self.advancement >= 100:
            return 0
        completed_time = (self.estimation_time * self.advancement) / 100
        return int(self.estimation_time - completed_time)
    
    class Meta:
        ordering = ['-created_date']
    
    equipe = models.CharField(max_length=255, blank=True)


class ChatMessage(models.Model):
    """Store chat conversation history"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.SET_NULL, null=True, blank=True)
    
    # The user's message
    user_message = models.TextField()
    
    # The bot's response
    bot_response = models.TextField()
    
    # Intent detected by NLU
    intent = models.CharField(max_length=50, blank=True)
    
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username}: {self.user_message[:50]}..."
    
    class Meta:
        ordering = ['-timestamp']




class ElixServiceCall(models.Model):
    """Log API calls (optional - for future use)"""
    service_id ="this can be used to track specific service calls, e.g., project details retrieval"
    """url = f"{MCP_BASE_URL}{MCP_WORKFLOW_SERVICE_PATH}/{service_id}"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=True, blank=True)
    
    service_name = models.CharField(max_length=255, blank=True)
    
    # Request details
    method = models.CharField(max_length=10)
    parameters = models.JSONField(default=dict, blank=True)
    
    # Response details
    status_code = models.IntegerField()
    response_data = models.JSONField(default=dict, blank=True)
    success = models.BooleanField(default=True)
    
    called_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Service {self.service_id} - {self.called_at}"
    
    class Meta:
        ordering = ['-called_at']