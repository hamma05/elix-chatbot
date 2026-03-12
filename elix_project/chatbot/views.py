# chatbot/views.py

import json

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt

from .chatbot_nlu import ChatbotNLU
from .models import ChatMessage, Project

# Initialize the chatbot NLU
chatbot = ChatbotNLU()


def login_view(request):
    """User login page."""
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("chat")
    else:
        form = AuthenticationForm()

    return render(request, "registration/login.html", {"form": form})


def logout_view(request):
    """User logout."""
    logout(request)
    return redirect("login")


@login_required
def chat_view(request):
    """Main chat interface."""
    # Reset conversation history whenever the chat page is loaded/refreshed.
    ChatMessage.objects.filter(user=request.user).delete()
    recent_messages = ChatMessage.objects.none()

    total_projects = Project.objects.count()
    active_projects = Project.objects.filter(status="In Progress").count()

    context = {
        "recent_messages": recent_messages,
        "total_projects": total_projects,
        "active_projects": active_projects,
    }

    return render(request, "chatbot/chat.html", context)


@login_required
@csrf_exempt
def send_message(request):
    """Handle chat messages via AJAX."""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "").strip()

            if not user_message:
                return JsonResponse({"error": "Empty message"}, status=400)

            intent_data = chatbot.parse_intent(user_message)
            intent = intent_data["intent"]

            bot_response, mentioned_project = process_intent(request.user, intent_data)

            chat_msg = ChatMessage.objects.create(
                user=request.user,
                user_message=user_message,
                bot_response=bot_response,
                intent=intent,
                project=mentioned_project,
            )

            return JsonResponse(
                {
                    "response": bot_response,
                    "intent": intent,
                    "confidence": intent_data.get("confidence", 0.0),
                    "nlu_source": intent_data.get("source", "rules"),
                    "fallback_used": intent_data.get("fallback_used", False),
                    "entities": intent_data.get("entities", {}),
                    "timestamp": chat_msg.timestamp.isoformat(),
                }
            )

        except Exception as exc:  # noqa: BLE001
            return JsonResponse({"error": str(exc)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)


def process_intent(user, intent_data):
    """Process the detected intent and generate a response."""
    intent = intent_data["intent"]
    entities = intent_data.get("entities", {})
    mentioned_project = None

    if intent == "search_project":
        project_name = entities.get("project")
        if project_name:
            try:
                project = Project.objects.get(name__icontains=project_name)
                mentioned_project = project

                response = f"Project: {project.name}\n\n"
                response += f"Product Owner: {project.product_owner}\n"
                response += f"Status: {project.status}\n"
                response += f"Progress: {project.advancement}%\n"
                response += f"Estimated Time: {project.estimation_time} days\n"
                response += f"Remaining Time: ~{project.remaining_time} days\n"

                if project.description:
                    response += f"\nDescription: {project.description}\n"

                if project.budget is not None:
                    response += f"\nBudget: ${project.budget:,.2f}"
                    if project.spent is not None:
                        response += f"\nSpent: ${project.spent:,.2f}"

                return response, mentioned_project

            except Project.DoesNotExist:
                return (
                    f"Project '{project_name}' not found in database. "
                    "Try 'list all projects' to see available projects.",
                    None,
                )
            except Project.MultipleObjectsReturned:
                projects = Project.objects.filter(name__icontains=project_name)
                response = f"Found multiple projects matching '{project_name}':\n\n"
                for proj in projects:
                    response += f"- {proj.name} (ID: {proj.project_id})\n"
                response += "\nPlease be more specific."
                return response, None
        return "Please specify a project name.", None

    if intent == "list_projects":
        projects = Project.objects.all()[:15]
        if projects:
            response = f"Available Projects ({Project.objects.count()} total):\n\n"
            for proj in projects:
                response += f"{proj.name}\n"
                response += (
                    f"  Owner: {proj.product_owner} | "
                    f"Status: {proj.status} | Progress: {proj.advancement}%\n\n"
                )
            return response, None
        return "No projects found in database.", None

    if intent == "generate_report":
        last_project_msg = ChatMessage.objects.filter(user=user, project__isnull=False).first()

        if last_project_msg:
            project = last_project_msg.project
            mentioned_project = project
            return generate_project_report(project), mentioned_project
        return "Please specify a project first. Say 'show me project [name]'.", None

    if intent == "check_status":
        active_projects = Project.objects.filter(status="In Progress")
        if active_projects:
            response = "Active Projects Status:\n\n"
            for proj in active_projects:
                response += f"{proj.name} - {proj.advancement}% complete\n"
                response += f"  Owner: {proj.product_owner}\n"
                response += f"  Time remaining: ~{proj.remaining_time} days\n\n"
            return response, None
        return "No active projects at the moment.", None

    if intent == "search_by_owner":
        owner_name = entities.get("owner")
        if owner_name:
            projects = Project.objects.filter(product_owner__icontains=owner_name)
            if projects:
                response = f"Projects owned by '{owner_name}':\n\n"
                for proj in projects:
                    response += f"- {proj.name} - {proj.status} ({proj.advancement}%)\n"
                return response, None
            return f"No projects found for owner '{owner_name}'.", None
        return "Please specify an owner name.", None

    if intent == "greeting":
        return (
            "Hello! I can help you with project information. Try asking:\n"
            "- List all projects\n"
            "- Show me project [name]\n"
            "- Generate a report\n"
            "- Check project status"
        ), None

    if intent == "help":
        return (
            "Available Commands:\n\n"
            "- List all projects - Show all projects\n"
            "- Show me project [name] - Get project details\n"
            "- Generate a report - Create detailed report\n"
            "- Check project status - See active projects\n"
            "- Projects by [owner name] - Filter by owner"
        ), None

    if intent == "thanks":
        return "You're welcome. Ask for another project update any time.", None

    if intent == "goodbye":
        return "Goodbye. I am here whenever you need project insights.", None

    return "I am not sure I understood that. Type 'help' to see available commands.", None


def generate_project_report(project):
    """Generate a detailed report for a project."""
    report = "PROJECT REPORT\n"
    report += "=" * 50 + "\n\n"

    report += f"Project Name: {project.name}\n"
    report += f"Project ID: {project.project_id}\n"
    report += f"Product Owner: {project.product_owner}\n"
    report += f"Status: {project.status}\n\n"

    report += "PROGRESS\n"
    report += f"- Current Progress: {project.advancement}%\n"
    report += f"- Estimated Total Time: {project.estimation_time} days\n"
    report += f"- Time Remaining: ~{project.remaining_time} days\n\n"

    if project.start_date:
        report += f"Start Date: {project.start_date.strftime('%Y-%m-%d')}\n"
    if project.end_date:
        report += f"Target End Date: {project.end_date.strftime('%Y-%m-%d')}\n"
    report += "\n"

    if project.budget is not None:
        report += "BUDGET\n"
        report += f"- Total Budget: ${project.budget:,.2f}\n"
        if project.spent is not None:
            report += f"- Amount Spent: ${project.spent:,.2f}\n"
            remaining = float(project.budget) - float(project.spent)
            report += f"- Remaining: ${remaining:,.2f}\n"
            percentage = (float(project.spent) / float(project.budget)) * 100
            report += f"- Budget Used: {percentage:.1f}%\n"
        report += "\n"

    if project.description:
        report += "DESCRIPTION\n"
        report += f"{project.description}\n\n"

    report += "=" * 50 + "\n"
    report += f"Report generated: {project.updated_date.strftime('%Y-%m-%d %H:%M')}"

    return report