import requests
import json
import os
from flask import current_app as app

class MantisBTAPI:
    def __init__(self, api_url, api_token):
        self.api_url = api_url
        self.api_token = api_token
        self.headers = {
            "Authorization": api_token,
            "Content-Type": "application/json"
        }

    def create_ticket(self, summary, description, category="General", project="IT Support", priority="normal"):
        """
        Create a new ticket in Mantis BT

        Args:
            summary (str): Brief summary of the issue
            description (str): Detailed description of the issue
            category (str): Issue category
            project (str): Project name
            priority (str): Issue priority

        Returns:
            dict: Response from Mantis BT API
        """
        issue_data = {
            "summary": summary,
            "description": description,
            "category": {"name": category},
            "project": {"name": project},
            "priority": {"name": priority}
        }

        try:
            response = requests.post(
                f"{self.api_url}/issues",
                headers=self.headers,
                json=issue_data
            )

            if response.status_code == 201:
                return {"success": True, "issue": response.json()}
            else:
                return {"success": False, "error": f"Failed to create ticket: {response.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_ticket(self, ticket_id):
        """
        Get ticket details from Mantis BT

        Args:
            ticket_id (int): ID of the ticket to retrieve

        Returns:
            dict: Ticket details
        """
        try:
            response = requests.get(
                f"{self.api_url}/issues/{ticket_id}",
                headers=self.headers
            )

            if response.status_code == 200:
                issue_data = response.json()["issues"][0]
                return {
                    "success": True,
                    "ticket": {
                        "id": issue_data["id"],
                        "status": issue_data["status"]["name"],
                        "summary": issue_data["summary"],
                        "description": issue_data["description"],
                        "category": issue_data["category"]["name"],
                        "project": issue_data["project"]["name"],
                        "priority": issue_data["priority"]["name"],
                        "assigned_to": issue_data.get("handler", {}).get("name", "Unassigned"),
                        "created_at": issue_data["created_at"],
                        "updated_at": issue_data["updated_at"]
                    }
                }
            else:
                return {"success": False, "error": f"Failed to retrieve ticket: {response.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_ticket(self, ticket_id, **kwargs):
        """
        Update an existing ticket in Mantis BT

        Args:
            ticket_id (int): ID of the ticket to update
            **kwargs: Fields to update (status, priority, etc.)

        Returns:
            dict: Response from Mantis BT API
        """
        update_data = {}

        # Map kwargs to Mantis BT API fields
        if "status" in kwargs:
            update_data["status"] = {"name": kwargs["status"]}
        if "priority" in kwargs:
            update_data["priority"] = {"name": kwargs["priority"]}
        if "summary" in kwargs:
            update_data["summary"] = kwargs["summary"]
        if "description" in kwargs:
            update_data["description"] = kwargs["description"]
        if "category" in kwargs:
            update_data["category"] = {"name": kwargs["category"]}
        if "notes" in kwargs:
            update_data["notes"] = [{"text": kwargs["notes"]}]

        try:
            response = requests.patch(
                f"{self.api_url}/issues/{ticket_id}",
                headers=self.headers,
                json=update_data
            )

            if response.status_code == 200:
                return {"success": True, "issue": response.json()}
            else:
                return {"success": False, "error": f"Failed to update ticket: {response.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_tickets(self, query_params):
        """
        Search for tickets in Mantis BT

        Args:
            query_params (dict): Parameters for the search

        Returns:
            dict: Search results
        """
        try:
            response = requests.get(
                f"{self.api_url}/issues",
                headers=self.headers,
                params=query_params
            )

            if response.status_code == 200:
                issues = response.json()["issues"]
                return {"success": True, "tickets": issues}
            else:
                return {"success": False, "error": f"Failed to search tickets: {response.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}