import requests

def test_api():
    # Test data
    test_input = {
        "device_type": "Mobile",
        "operating_system": "iOS",
        "web_vs_native": "Native",
        "age_group": "Adults",
        "content_type": "Text-Heavy",
        "primary_goal": "Productivity",
        "interaction_type": "Touchscreen"
    }
    
    # Make request to your API
    response = requests.post(
        'http://localhost:5501/api/recommendations',
        json=test_input
    )
    
    # Print results
    print("Status Code:", response.status_code)
    print("\nResponse:", response.json())

if __name__ == "__main__":
    test_api()