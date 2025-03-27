# DesignGenie: A UX Design Recommendation System

Welcome to the DesignGenie repository! 🎨✨

## Table of Contents

- [About the Project](#about-the-project)
  - [Overview](#overview)
  - [Goals](#goals)
- [Technologies Used](#technologies-used)
  - [What I Learned](#what-i-learned)
  - [Continued Development](#continued-development)
  - [The Design](#the-design)
- [Features](#features)
- [Database Structure](#database-structure)
- [Setup and Usage](#setup-and-usage)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
  - [Using the Project](#using-the-project)
  - [Troubleshooting](#troubleshooting)
- [Useful Resources](#useful-resources)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

## About the Project

### Overview

DesignGenie is a UX design recommendation system that uses theoretical design principles to help designers and marketers create optimized user interfaces tailored to specific user needs. Users input project information, and the system generates actionable, data-driven recommendations grounded in established UX theories.

### Goals

The key objectives of this project include:

- Providing personalized design recommendations based on user inputs such as target audience, platform, content type, and accessibility needs.
- Bridging the gap between UX theory and practice by offering actionable design guidelines.
- Enhancing the UX design process with a hybrid recommendation model that combines theoretical data and dynamic user inputs.
- Supporting designers and marketers in creating designs that are accessible, engaging, and user-friendly.

## Technologies Used

This system was built using:

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Python (Flask framework)
- **Database**: SQLite (`design_recommendations.db`)
- **Other Tools**: Jupyter Notebooks for model development, Git for version control

### What I Learned

During the development of DesignGenie, I gained valuable insights, such as:

- Encoding theoretical UX design principles into a structured database.
- Implementing a hybrid recommendation model to generate actionable design suggestions.
- Creating a backend in Python that integrates seamlessly with a dynamic frontend interface.
- Developing systems with accessibility and inclusivity as core principles.

### Continued Development

Future development plans include:

- Adding support for advanced input options, such as brand guidelines and aesthetic preferences.
- Expanding the database with more detailed UX principles and real-world examples.
- Implementing a color-matching tool to recommend accessible color schemes.
- Connecting the recommendation system to the DesignGenie web interface for a seamless user experience.

### The Design

DesignGenie is built with a focus on:

- A clean, intuitive interface for both novice and experienced users.
- Accessibility-first design principles to ensure inclusivity.
- Dynamic outputs that adapt to user preferences and project constraints.

The color scheme and layout are inspired by user-friendly design standards, balancing functionality with aesthetics.

## Features

- **Personalized Recommendations**: Tailored design suggestions based on detailed user inputs.
- **Holistic Design Database**: A repository of theoretical UX design principles.
- **Hybrid Recommendation Model**: Combines static theory with dynamic user criteria.
- **Accessibility Focus**: Ensures outputs align with inclusive design standards.
- **Color Matching Tool (Planned)**: Assists with accessible and visually appealing color palette creation.

## Database Structure

The SQLite database (`design_recommendations.db`) is the foundation of DesignGenie, consisting of:

1. **Guidelines**: Stores theoretical design principles with categories, subsections, details, examples, and keywords.
2. **UserInputs**: Captures user-provided data such as target audience, platform, content type, and accessibility needs.
3. **RecommendationsMapping**: Maps user inputs to guidelines, enabling refined and dynamic recommendation generation.

## Setup and Usage

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python** (version 3.8 or above)
- **SQLite** (pre-installed with Python)
- **Git** (for cloning the repository)
- A code editor (e.g., VSCode, PyCharm) or Jupyter Notebook environment

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repository/designgenie.git
   cd designgenie

2. **Set Up a Virtual Environment**
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate

3. **Install Dependencies**
    pip install -r requirements.txt

4. **Initialise the Database**
    python setup_database.py

5. **Run the Application**
    flask run

### Using the Project

1. Access the interface
Open your browser and navigate to http://127.0.0.1:5500.

2. Click 'Get Started' button and complete the form

3. Generate Recommendations

4. View Recommendations

## Acknowledgments

A special thanks to:

- The UX design and machine learning communities for providing invaluable research and resources.
- My supervisor and peers for their continuous support and feedback.

---

Thank you for exploring DesignGenie!
