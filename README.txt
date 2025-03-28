# 🎨 DesignGenie: A UX Design Recommendation System

Welcome to the DesignGenie repository — your intelligent assistant for turning user insights into accessible, aesthetically sound, and data-driven design decisions.

---

## 📚 Table of Contents
- [About the Project](#about-the-project)
  - [Overview](#overview)
  - [Goals](#goals)
- [Technologies Used](#technologies-used)
  - [What I Learned](#what-i-learned)
  - [Continued Development](#continued-development)
  - [The Design](#the-design)
- [Features](#features)
- [Database Structure](#database-structure)
- [Useful Resources](#useful-resources)
- [Acknowledgments](#acknowledgments)
- [Author](#author)

---

## 💡 About the Project

### Overview
DesignGenie is a UX design recommendation system that bridges the gap between UX theory and creative practice. With just a few user inputs, the system leverages a curated database of theoretical principles to generate actionable, accessible, and relevant UI/UX suggestions. It's ideal for marketers, designers, project managers, designers and developers looking to make informed design choices.

### Goals
- Deliver **personalized design recommendations** based on user-defined project attributes.
- Apply **design theory at scale** using a structured recommendation engine.
- Promote **accessibility-first principles** through inclusive guidelines.
- Support designers and marketers by reducing decision fatigue and increasing confidence.

---

## 🛠 Technologies Used

### Core Stack
- **Frontend:** HTML5, CSS3, JavaScript
- **Backend:** Python (Flask)
- **Database:** SQLite (`design_recommendations.db`)
- **Dev Tools:** Git, Jupyter Notebooks

### What I Learned
- Translating UX theory into machine-readable structures
- Designing and managing modular databases for scalable recommendations
- Building a Flask application with dynamic front-end integration
- Prioritizing accessibility and usability at every development stage

### Continued Development
- 🎨 Integrate a color-matching tool for WCAG-compliant palettes
- 🧠 Expand the rule engine to account for brand identity and visual preferences
- 📈 Connect recommendations to performance feedback metrics for iterative improvement

### The Design
DesignGenie focuses on clarity, simplicity, and functionality. Its design language follows:
- A clean UI for intuitive exploration
- Accessibility-driven spacing, contrast, and input prompts
- Modular UI components prepared for future integrations (e.g., dashboards, data tracking)

---

## ✨ Features
- **Tailored Recommendations:** Design guidance based on platform, audience, goals, and content
- **Hybrid Recommendation Engine:** Mixes static UX theory with real-time user input
- **Inclusive Design First:** Recommends with accessibility guidelines baked in
- **Expandable Architecture:** Easy to grow with new principles or patterns
- **User-First Interface:** Beginner-friendly and visually focused

---

## 🗂 Database Structure
`design_recommendations.db` includes:

### 1. Guidelines Table
Stores theoretical UX principles with fields:
- Category
- Section
- Subsection
- Guideline & Description
- Examples
- Keywords (for smarter matching)

### 2. UserInputs Table
Captures detailed project data:
- Target Audience
- Platform
- Content Type
- Visual Style
- Accessibility Needs
- Device & Navigation Info

### 3. RecommendationsMapping Table
Maps `UserInputs` to matching `Guidelines`, forming personalized suggestions.

---

## 🔗 Useful Resources
- [Laws of UX](https://lawsofux.com/)
- [Web Content Accessibility Guidelines (WCAG)](https://www.w3.org/WAI/WCAG21/quickref/)
- [Material Design](https://m3.material.io/)

---

## 🙏 Acknowledgments
Thanks to the open-source design and accessibility communities for the foundational research. Special appreciation to my mentors and peers who guided this work.

---

## 👩‍💻 Author
Created by **Nicolette Woolery**  
LinkedIn: [linkedin.com/in/nicolettewoolery](https://linkedin.com/in/nicolettewoolery)

---

Thank you for exploring **DesignGenie**! 🎨💡
