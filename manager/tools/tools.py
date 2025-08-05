from google.adk.tools import FunctionTool
from datetime import datetime

def log_message(message):
    """Logs a message to the console."""
    print(f"[LOG] {message}")

def format_data(data):
    """Formats data for better readability."""
    return str(data).replace(",", ", ")

def validate_input(data, expected_type):
    """Validates the input data against the expected type."""
    if not isinstance(data, expected_type):
        raise ValueError(f"Expected data of type {expected_type}, got {type(data)}")

def generate_report(data):
    """Generates a report from the provided data."""
    report = "Report:\n"
    for key, value in data.items():
        report += f"{key}: {value}\n"
    return report.strip()

def get_current_time() -> str:
    """Returns the current date and time as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

get_current_time_tool = FunctionTool(func=get_current_time)
