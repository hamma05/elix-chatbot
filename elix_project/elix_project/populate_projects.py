# populate_projects.py
"""
Run this script to populate your Django database with project data.
Usage: python manage.py shell < populate_projects.py
"""

from datetime import datetime, timedelta
from decimal import Decimal

from chatbot.models import Project


PROJECTS_DATA = [
    {
        "project_id": 1001,
        "name": "E-Commerce Platform",
        "description": "Development of new online shopping platform with mobile support",
        "product_owner": "Amira Ben Salah",
        "advancement": 75,
        "estimation_time": 180,
        "status": "In Progress",
        "budget": Decimal("250000.00"),
        "spent": Decimal("187500.00"),
        "start_date": datetime.now().date() - timedelta(days=135),
    },
    {
        "project_id": 1002,
        "name": "Mobile Banking App",
        "description": "iOS and Android banking application with biometric authentication",
        "product_owner": "Youssef Trabelsi",
        "advancement": 100,
        "estimation_time": 120,
        "status": "Completed",
        "budget": Decimal("180000.00"),
        "spent": Decimal("175000.00"),
        "start_date": datetime.now().date() - timedelta(days=150),
        "end_date": datetime.now().date() - timedelta(days=30),
    },
    {
        "project_id": 1003,
        "name": "Customer Portal Redesign",
        "description": "UI/UX redesign of customer self-service portal",
        "product_owner": "Leila Ben Youssef",
        "advancement": 45,
        "estimation_time": 90,
        "status": "In Progress",
        "budget": Decimal("75000.00"),
        "spent": Decimal("33750.00"),
        "start_date": datetime.now().date() - timedelta(days=40),
    },
    {
        "project_id": 1004,
        "name": "Data Analytics Dashboard",
        "description": "Real-time business intelligence dashboard for executives",
        "product_owner": "Omar Ben Salem",
        "advancement": 30,
        "estimation_time": 150,
        "status": "In Progress",
        "budget": Decimal("200000.00"),
        "spent": Decimal("60000.00"),
        "start_date": datetime.now().date() - timedelta(days=45),
    },
    {
        "project_id": 1005,
        "name": "Cloud Migration",
        "description": "Migration of on-premise infrastructure to AWS cloud",
        "product_owner": "Fatma Aloui",
        "advancement": 60,
        "estimation_time": 240,
        "status": "In Progress",
        "budget": Decimal("500000.00"),
        "spent": Decimal("300000.00"),
        "start_date": datetime.now().date() - timedelta(days=144),
    },
    {
        "project_id": 1006,
        "name": "API Integration Platform",
        "description": "Unified API gateway for third-party integrations",
        "product_owner": "Noureddine Gharbi",
        "advancement": 15,
        "estimation_time": 200,
        "status": "Planning",
        "budget": Decimal("300000.00"),
        "spent": Decimal("45000.00"),
        "start_date": datetime.now().date() - timedelta(days=30),
    },
    {
        "project_id": 1007,
        "name": "Security Audit System",
        "description": "Automated security scanning and vulnerability detection",
        "product_owner": "Karim Mansouri",
        "advancement": 85,
        "estimation_time": 100,
        "status": "In Progress",
        "budget": Decimal("120000.00"),
        "spent": Decimal("102000.00"),
        "start_date": datetime.now().date() - timedelta(days=85),
    },
    {
        "project_id": 1008,
        "name": "Inventory Management System",
        "description": "Warehouse inventory tracking with RFID integration",
        "product_owner": "Leila Ben Youssef",
        "advancement": 100,
        "estimation_time": 160,
        "status": "Completed",
        "budget": Decimal("220000.00"),
        "spent": Decimal("215000.00"),
        "start_date": datetime.now().date() - timedelta(days=180),
        "end_date": datetime.now().date() - timedelta(days=20),
    },
    {
        "project_id": 1009,
        "name": "Employee Training Portal",
        "description": "Online learning management system for employee development",
        "product_owner": "Omar Ben Salem",
        "advancement": 50,
        "estimation_time": 120,
        "status": "In Progress",
        "budget": Decimal("95000.00"),
        "spent": Decimal("47500.00"),
        "start_date": datetime.now().date() - timedelta(days=60),
    },
    {
        "project_id": 1010,
        "name": "Marketing Automation",
        "description": "Email and social media marketing automation platform",
        "product_owner": "Noureddine Gharbi",
        "advancement": 20,
        "estimation_time": 130,
        "status": "Planning",
        "budget": Decimal("150000.00"),
        "spent": Decimal("30000.00"),
        "start_date": datetime.now().date() - timedelta(days=26),
    },
]


def populate_database():
    """Add projects to Django database."""
    print("=" * 60)
    print("POPULATING DATABASE WITH PROJECT DATA")
    print("=" * 60)
    print()

    created_count = 0
    updated_count = 0

    for project_data in PROJECTS_DATA:
        project, created = Project.objects.update_or_create(
            project_id=project_data["project_id"],
            defaults={
                "name": project_data["name"],
                "description": project_data["description"],
                "product_owner": project_data["product_owner"],
                "advancement": project_data["advancement"],
                "estimation_time": project_data["estimation_time"],
                "status": project_data["status"],
                "budget": project_data.get("budget"),
                "spent": project_data.get("spent"),
                "start_date": project_data.get("start_date"),
                "end_date": project_data.get("end_date"),
            },
        )

        if created:
            print(f"Created: {project.name}")
            print(f"   Owner: {project.product_owner} | Progress: {project.advancement}%")
            created_count += 1
        else:
            print(f"Updated: {project.name}")
            updated_count += 1

    print()
    print("=" * 60)
    print("SUMMARY:")
    print(f"  - Created: {created_count} projects")
    print(f"  - Updated: {updated_count} projects")
    print(f"  - Total in database: {Project.objects.count()} projects")
    print("=" * 60)
    print()

    print("PROJECT STATISTICS:")
    print(f"  - In Progress: {Project.objects.filter(status='In Progress').count()}")
    print(f"  - Completed: {Project.objects.filter(status='Completed').count()}")
    print(f"  - Planning: {Project.objects.filter(status='Planning').count()}")
    print()

    owners = (
        Project.objects.order_by("product_owner")
        .values_list("product_owner", flat=True)
        .distinct()
    )
    print("PRODUCT OWNERS:")
    for owner in owners:
        count = Project.objects.filter(product_owner=owner).count()
        print(f"  - {owner}: {count} project(s)")

    print()
    print("Database populated successfully!")


if __name__ == "__main__":
    populate_database()
