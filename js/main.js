document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("onboardingForm");
    const steps = document.querySelectorAll(".form-step");
    const progressBar = document.getElementById("progressBar");
    const totalSteps = steps.length;
    let currentStepIndex = 0;
    const userInputs = JSON.parse(localStorage.getItem("userInputs"));
    const recommendations = JSON.parse(localStorage.getItem("recommendations"));

    function updateProgressBar(stepIndex) {
        if (!progressBar) {
            console.warn("Progress bar element not found.");
            return;
        }
        const progress = ((stepIndex + 1) / totalSteps) * 100;
        progressBar.style.width = `${progress}%`;
    }

    // Function to show a specific step
    window.showStep = function (stepIndex) {
        // Hide all steps
        steps.forEach((step) => {
            step.style.display = "none";
        });

        // Show the target step
        if (stepIndex >= 0 && stepIndex < steps.length) {
            steps[stepIndex].style.display = "block";
            currentStepIndex = stepIndex;
            updateProgressBar(currentStepIndex);
        }
    };

    // Show the first step on page load
    showStep(currentStepIndex);

    // Validate the current step
    function validateStep() {
        const currentStep = steps[currentStepIndex];
        const inputs = currentStep.querySelectorAll("input[required], select[required], textarea[required]");
        let isValid = true;

        inputs.forEach((input) => {
            if (!input.checkValidity()) {
                isValid = false;
                input.classList.add("error");
                const errorMessage = input.parentElement.querySelector(".error-message");
                if (errorMessage) {
                    errorMessage.textContent = input.validationMessage;
                }
            } else {
                input.classList.remove("error");
                const errorMessage = input.parentElement.querySelector(".error-message");
                if (errorMessage) {
                    errorMessage.textContent = "";
                }
            }
        });

        return isValid;
    }

    // Handle "Next" button click
    const nextButtons = document.querySelectorAll(".next-btn");
    nextButtons.forEach((button) => {
        button.addEventListener("click", () => {
            if (validateStep()) {
                showStep(currentStepIndex + 1);
            }
        });
    });

    // Handle "Back" button click
    const backButtons = document.querySelectorAll(".back-btn");
    backButtons.forEach((button) => {
        button.addEventListener("click", () => {
            showStep(currentStepIndex - 1);
        });
    });

    localStorage.clear();

    // Form submission handler
    form.addEventListener("submit", (e) => {
        e.preventDefault();
    
        const formData = new FormData(form);

        const colorValue = document.getElementById("colorPicker").value;
        formData.set("color_code", colorValue);

        const formObject = {};
        formData.forEach((value, key) => {
            console.log(`Collecting form data - ${key}: ${value}`);
            formObject[key] = value;
        });
        formObject["color_code"] = colorValue;
    
        console.log("Collected Form Data:", formObject);
    
        // Save locally and send to backend
        localStorage.setItem("userInputs", JSON.stringify(formObject));
    
        fetch("https://designgenie.glitch.me/submit", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(formObject),
          })
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then((data) => {
                console.log("Success:", data);
                localStorage.setItem("recommendations", JSON.stringify(data.recommendations || []));
                window.location.href = "ResultsCard.html";
            })
            .catch((error) => {
                console.error("Error:", error);
                alert(`Submission failed: ${error.message}`);
            });
    });


    // Color Picker Integration
    const colorInput = document.getElementById("colorPicker");
    const colorDisplay = document.createElement("div");
    colorDisplay.className = "color-display";
    colorDisplay.style.width = "50px";
    colorDisplay.style.height = "50px";
    colorDisplay.style.marginTop = "10px";
    colorDisplay.style.border = "1px solid #000";

    // Append the display below the color picker
    colorInput.parentNode.appendChild(colorDisplay);

    // Update the display and form hidden value when a color is selected
    colorInput.addEventListener("input", (event) => {
        const selectedColor = event.target.value;
        colorDisplay.style.backgroundColor = selectedColor; // Update the display
        console.log("Selected Color:", selectedColor); // Debugging log
    });

    function handleOptionSelection(inputName, inputValue) {
        console.log(`Handling selection for ${inputName}: ${inputValue}`); // Debug log
        
        try {
            // 1. Update the radio button
            const radioInput = document.querySelector(`input[type="radio"][name="${inputName}"][value="${inputValue}"]`);
            if (radioInput) {
                radioInput.checked = true;
            } else {
                console.warn(`Radio input not found for ${inputName} = ${inputValue}`);
            }
    
            // 2. Update the hidden input
            const hiddenInput = document.querySelector(`input[type="hidden"][name="${inputName}"]`);
            if (hiddenInput) {
                hiddenInput.value = inputValue;
                console.log(`Updated hidden input ${inputName} with value: ${inputValue}`);
            } else {
                console.warn(`Hidden input not found for ${inputName}`);
            }
    
            // 3. Update visual selection
            const allOptionBoxes = document.querySelectorAll(`.option-box[onclick*="${inputName}"]`);
            allOptionBoxes.forEach(box => box.classList.remove('selected'));
    
            const selectedBox = radioInput?.closest('.option-box');
            if (selectedBox) {
                selectedBox.classList.add('selected');
            }
    
        } catch (error) {
            console.error('Error in handleOptionSelection:', error);
        }
    }
    
    // Make it globally available
    window.handleOptionSelection = handleOptionSelection;
    

    if (!userInputs) {
        console.error("No user inputs found in localStorage.");
        document.querySelector(".dashboard-title").innerHTML += "<p class='error'>User inputs not found. Please complete the form.</p>";
        return;
    }

    console.log("User Inputs:", userInputs);
    console.log("Recommendations:", recommendations);

    // Populate user inputs
    const populateField = (id, value, fallback = "N/A") => {
        const field = document.getElementById(id);
        if (field) field.textContent = value || fallback;
    };

    populateField("age_group", userInputs.age_group);
    populateField("occupation", userInputs.occupation);
    populateField("location", userInputs.location);
    populateField("technical-proficiency", userInputs.technical_proficiency);
    populateField("user-preferences", userInputs.user_preferences);
    populateField("goal", userInputs.primary_goal);
    populateField("industry-type", userInputs.industry_type);
    populateField("brand-personality", userInputs.brand_personality);
    populateField("visual-style", userInputs.visual_style);
    populateField("content-type", userInputs.content_type);
    populateField("platform", userInputs.device_type);
    populateField("screen-resolution", userInputs.screen_resolution);
    populateField("interaction-type", userInputs.interaction_type);
    populateField("navigation-type", userInputs.navigation_type);
    populateField("cognitive-load", userInputs.cognitive_load);
    populateField("visual-impairment", userInputs.visual_impairment);
    populateField("motor-impairment", userInputs.motor_impairment);
    populateField("hearing-impairment", userInputs.hearing_impairment);

    // Populate color palette
    const colors = [
        userInputs.color_code || "#FF5733", // Main color from user input or default
        adjustBrightness(userInputs.color_code || "#FF5733", 20), // Lighter variation
        adjustBrightness(userInputs.color_code || "#FF5733", -20), // Darker variation
        generateComplementaryColor(userInputs.color_code || "#FF5733"), // Complementary color
        "#4CAF50" // A fallback static color
    ];

    ["color1", "color2", "color3", "color4", "color5"].forEach((colorId, index) => {
        const colorElement = document.getElementById(colorId);
        if (colorElement) {
            colorElement.style.backgroundColor = colors[index];
        }
    });


    // function displayRecommendations(recommendations) {
    //     const recommendationsContainer = document.getElementById("recommendations-by-category");
    //     recommendationsContainer.innerHTML = ""; // Clear existing content
    
    //     if (!recommendations || recommendations.length === 0) {
    //         recommendationsContainer.innerHTML = "<p>No recommendations available.</p>";
    //         return;
    //     }
    
    //     // Deduplicate recommendations by guideline or ID
    //     // const uniqueRecommendations = Array.from(
    //     //     new Map(recommendations.map(rec => [rec.guideline, rec])).values()
    //     // );
    
    //     const recommendationsByCategory = uniqueRecommendations.reduce((acc, rec) => {
    //         const category = rec.category || "Uncategorized";
    //         acc[category] = acc[category] || [];
    //         acc[category].push(rec);
    //         return acc;
    //     }, {});
    
    //     Object.entries(recommendationsByCategory).forEach(([category, recs]) => {
    //         const categorySection = document.createElement("div");
    //         categorySection.className = "category-section";
    //         categorySection.innerHTML = `<h3 class="category-title">${category}</h3>`;
    
    //         recs.forEach((rec) => {
    //             const recommendationCard = document.createElement("div");
    //             recommendationCard.className = "recommendation-card";
    //             recommendationCard.innerHTML = `
    //                 <h4>${rec.guideline}</h4>
    //                 <p><strong>Details:</strong> ${rec.details}</p>
    //                 ${rec.examples ? `<p><strong>Example:</strong> ${rec.examples}</p>` : ""}
    //                 <div class="recommendation-meta">
    //                     <p><strong>Category:</strong> ${rec.category || "Uncategorized"}</p>
    //                     <p><strong>Section:</strong> ${rec.section || "General"}</p>
    //                     <span class="score">Relevance: ${(rec.similarity_score * 100).toFixed(0)}%</span>
    //                 </div>
    //             `;
    
    //             // Add context matches if available
    //             // if (rec.context_matches) {
    //             //     const contextInfo = Object.entries(rec.context_matches)
    //             //         .filter(([key, value]) => value) // Only include matching contexts
    //             //         .map(([key]) => `<span class="match">${key}</span>`) // Format as badges
    //             //         .join(" ");
    
    //             //     recommendationCard.innerHTML += `
    //             //         <div class="context-info">
    //             //             <strong>Matches:</strong> ${contextInfo}
    //             //         </div>
    //             //     `;
    //             // }
    
    //             categorySection.appendChild(recommendationCard);
    //         });
    
    //         recommendationsContainer.appendChild(categorySection);
    //     });
    // }

    // Helper functions for color adjustments
    function adjustBrightness(hex, percent) {
        let rgb = hexToRgb(hex);
        rgb = rgb.map((color) => Math.min(255, Math.max(0, color + Math.round((percent / 100) * 255))));
        return rgbToHex(rgb[0], rgb[1], rgb[2]);
    }

    function generateComplementaryColor(hex) {
        let rgb = hexToRgb(hex);
        const complementary = rgb.map((color) => 255 - color);
        return rgbToHex(complementary[0], complementary[1], complementary[2]);
    }

    function hexToRgb(hex) {
        hex = hex.replace(/^#/, "");
        if (hex.length === 3) {
            hex = hex.split("").map((h) => h + h).join("");
        }
        const bigint = parseInt(hex, 16);
        return [(bigint >> 16) & 255, (bigint >> 8) & 255, bigint & 255];
    }

    function rgbToHex(r, g, b) {
        return `#${[r, g, b].map((x) => x.toString(16).padStart(2, "0")).join("")}`;
    }
    

});

// Sparkle Sparkle!
const header = document.getElementById("magicHeader");

header.addEventListener("mousemove", (e) => {
  // Generate multiple sparkles per move
  for (let i = 0; i < 3; i++) {
    const sparkle = document.createElement("div");
    sparkle.classList.add("sparkle");

    // Random color from a magical palette
    const colors = ["#ffffff", "#f5a3ff", "#a3f7ff", "#ffe29e", "#e2a3ff"];
    const color = colors[Math.floor(Math.random() * colors.length)];
    sparkle.style.background = `radial-gradient(circle, ${color}, transparent)`;

    // Random position around cursor
    const offsetX = (Math.random() - 0.5) * 40;
    const offsetY = (Math.random() - 0.5) * 40;
    sparkle.style.left = `${e.clientX + offsetX}px`;
    sparkle.style.top = `${e.clientY + offsetY}px`;

    // Random size
    const size = Math.random() * 8 + 4; // 4px to 12px
    sparkle.style.width = `${size}px`;
    sparkle.style.height = `${size}px`;

    // Optional glow
    sparkle.style.boxShadow = `0 0 8px ${color}, 0 0 16px ${color}`;

    document.body.appendChild(sparkle);

    // Clean up
    setTimeout(() => sparkle.remove(), 1200);
  }
});

document.addEventListener('mousemove', (e) => {
  const layers = document.querySelectorAll('.parallax-layer');
  layers.forEach((layer, index) => {
    const speed = (index + 1) * 0.02;
    layer.style.transform = `translate(${e.clientX * speed}px, ${e.clientY * speed}px)`;
  });
});

//Share button
const dashboardSummary = encodeURIComponent(document.getElementById("dashboard").innerText);
const mailto = `mailto:designer@example.com?subject=Design Genie Brief&body=${dashboardSummary}`;
document.getElementById("emailLink").href = mailto;


//Sidebar Functionality
const overviewLink = document.getElementById("overviewLink");
const recommendationsLink = document.getElementById("recommendationsLink");

const overviewSection = document.getElementById("overviewSection");
const recommendationsSection = document.getElementById("recommendationsSection");

overviewLink.addEventListener("click", (e) => {
    e.preventDefault();
    overviewSection.style.display = "block";
    recommendationsSection.style.display = "none";
    setActive(overviewLink);
});

recommendationsLink.addEventListener("click", (e) => {
    e.preventDefault();
    overviewSection.style.display = "none";
    recommendationsSection.style.display = "block";
    setActive(recommendationsLink);
});

function setActive(activeLink) {
    document.querySelectorAll(".sidebar nav ul li a").forEach(link => {
        link.classList.remove("active");
    });
    activeLink.classList.add("active");
}

const hamburger = document.querySelector('.hamburger');
const navLinks = document.querySelector('.nav-links');

hamburger.addEventListener('click', () => {
  navLinks.classList.toggle('active');
});

window.addEventListener("scroll", function () {
    const navbar = document.querySelector(".navbar");
    if (window.scrollY > 50) {
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }
  });

// document.addEventListener("DOMContentLoaded", () => {
//     // ==== INITIALIZATION ====
//     let currentStep = 1;
//     const steps = document.querySelectorAll(".form-step");
//     const progressBar = document.getElementById("progressBar");
//     const form = document.getElementById("onboardingForm");

//     // Check for critical elements
//     if (!steps.length || !progressBar) {
//         console.error("Critical elements are missing in the DOM");
//         return;
//     }

//     // ==== VALIDATION CONSTANTS ====
//     const VALID_VALUES = {
//         industry_type: [
//             'ecommerce', 'education', 'healthcare', 'finance', 'retail', 
//             'hospitality', 'real-estate', 'manufacturing', 'technology', 
//             'transportation', 'telecommunications', 'energy', 'entertainment', 
//             'government', 'nonprofit', 'legal', 'pharmaceutical', 'construction', 
//             'automotive', 'media'
//         ],
//         content_type: ['text', 'image', 'video', 'interactive', 'data-heavy', 'mixed-media'],
//         tone: ['Professional', 'Friendly', 'Informative', 'Entertaining'],
//         technical_proficiency: ['Beginner', 'Intermediate', 'Advanced'],
//         cognitive_load: ['High Cognitive Load', 'Low Cognitive Load'],
//         visual_impairment: ['None', 'Color Blindness', 'Low Vision', 'Blindness'],
//         motor_impairment: ['None', 'Full Keyboard Support', 'Voice Command Control', 'Large Touch Targets'],
//         hearing_impairment: ['None', 'Audio Alternatives'],
//         design_preferences: ['Minimalist Design', 'Bold and Vibrant Design', 'Classical And Traditional', 'Professional and Formal'],
//         mood_feel: ['Professional', 'Friendly', 'Fun', 'Modern', 'Minimalist'],
//         imagery_iconography: ['Realistic Images', 'Illustrative Images', 'Icons', 'Abstract Imagery', 'Minimalistic Imagery'],
//         device_type: ['Mobile', 'Tablet', 'Desktop', 'Web'],
//         operating_system: ['iOs', 'Android', 'Windows', 'MacOS'],
//         user_flow: ['Linear Navigation', 'Hierarchical Navigation', 'Interactive Navigation', 'Freeform Navigation'],
//         cta_placement: ['Homepage CTA', 'Footer CTA', 'Sidebar CTA', 'Popup or Modal CTA'],
//         accessibility_needs: [
//             'None', 'WCAG Compliance', 'High Contrast Mode', 
//             'Alternative Text for Images', 'Closed Captioning for Videos', 
//             'Keyboard Navigation'
//         ]
//     };

//     // ==== HELPER FUNCTIONS ====
//     function getRadioValue(name) {
//         const selectedOption = document.querySelector(`input[name="${name}"]:checked`);
//         return selectedOption ? selectedOption.value : "";
//     }

//     function getCheckboxValues(name) {
//         const checkboxes = document.querySelectorAll(`input[name="${name}"]:checked`);
//         return Array.from(checkboxes).map(cb => cb.value);
//     }

//     function showError(message) {
//         const formContainer = document.querySelector(".form-container");
//         if (!formContainer) {
//             console.error("Form container not found");
//             return;
//         }

//         let errorContainer = document.getElementById("error-container");
//         if (!errorContainer) {
//             errorContainer = document.createElement("div");
//             errorContainer.id = "error-container";
//             errorContainer.style.cssText = `
//                 background-color: #ffebee;
//                 color: #c62828;
//                 padding: 10px;
//                 margin: 10px 0;
//                 border-radius: 4px;
//                 border: 1px solid #ef5350;
//             `;
//             formContainer.insertBefore(errorContainer, formContainer.firstChild);
//         }

//         errorContainer.textContent = message;
//         errorContainer.style.display = "block";
//         errorContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });

//         setTimeout(() => errorContainer.style.display = "none", 5000);
//     }

//     // ==== FORM NAVIGATION FUNCTIONS ====
//     function updateProgressBar(step) {
//         const progressPercentage = (step / steps.length) * 100;
//         progressBar.style.width = `${progressPercentage}%`;
//     }

//     function showStep(step) {
//         steps.forEach((formStep, index) => {
//             formStep.style.display = (index + 1 === step) ? "block" : "none";
//             formStep.classList.toggle("fade-in", index + 1 === step);
//         });
//         updateProgressBar(step);
//     }

//     window.showStep = function(step) {
//         if (step < 1 || step > steps.length) {
//             console.error(`Invalid step: ${step}`);
//             return;
//         }
//         showStep(step);
//     };

//     window.nextStep = function() {
//         if (currentStep < steps.length) {
//             currentStep++;
//             showStep(currentStep);
//         }
//     };

//     window.prevStep = function() {
//         if (currentStep > 1) {
//             currentStep--;
//             showStep(currentStep);
//         }
//     };

//     // ==== FORM DATA HANDLING ====
//     function validateFormData(data) {
//         console.log("Starting validation for data:", JSON.stringify(data, null, 2));
        
//         const errors = [];
        
//         // Define exact format requirements
//         const requiredFields = {
//             industry_type: value => VALID_VALUES.industry_type.includes(value),
//             content_type: value => VALID_VALUES.content_type.includes(value),
//             personas: value => Array.isArray(value) && value.length > 0,
//             user_preferences: value => Array.isArray(value) && value.length > 0,
//             device_compatibility: value => Array.isArray(value) && value.length > 0,
//             browser_compatibility: value => Array.isArray(value) && value.length > 0,
//             core_features: value => Array.isArray(value) && value.length > 0
//         };
    
//         // Check each required field
//         Object.entries(requiredFields).forEach(([field, validator]) => {
//             if (!data[field] || !validator(data[field])) {
//                 errors.push(`Invalid or missing ${field}`);
//                 console.error(`Validation failed for ${field}:`, {
//                     value: data[field],
//                     type: typeof data[field],
//                     isArray: Array.isArray(data[field]),
//                     length: Array.isArray(data[field]) ? data[field].length : 'N/A'
//                 });
//             }
//         });
    
//         // Additional field validation
//         if (data.age_group && typeof data.age_group !== 'string') {
//             errors.push('age_group must be a string');
//         }
        
//         const result = {
//             isValid: errors.length === 0,
//             errors: errors
//         };
    
//         console.log("Validation result:", result);
//         return result;
//     }
    
//     function collectFormData() {
//         const data = {
//             // Required fields
//             industry_type: getRadioValue("industry_type") || '',
//             content_type: getRadioValue("content_type") || '',
//             personas: getCheckboxValues("personas"),
//             user_preferences: getCheckboxValues("user_preferences"),
//             device_compatibility: getCheckboxValues("device_compatibility"),
//             browser_compatibility: getCheckboxValues("browser_compatibility"),
//             core_features: getCheckboxValues("core_features"),
            
//             // Optional fields with explicit default values
//             age_group: document.getElementById("age_group")?.value || '',
//             gender: getRadioValue("gender") || '',
//             occupation: document.getElementById("occupation")?.value || '',
//             location: document.getElementById("location")?.value || '',
//             tone: getRadioValue("tone") || '',
//             content_structure: getRadioValue("content_structure") || '',
//             technical_proficiency: getRadioValue("technical_proficiency") || '',
//             cognitive_load: getRadioValue("cognitive_load") || '',
//             visual_impairment: getRadioValue("visual_impairment") || '',
//             motor_impairment: getRadioValue("motor_impairment") || '',
//             hearing_impairment: getRadioValue("hearing_impairment") || '',
//             accessibility_preferences: getCheckboxValues("accessibility_preferences"),
//             color_code: document.querySelector("input[name='color_code']")?.value || '',
//             branding_constraints: getCheckboxValues("brand_guidelines"),
//             device_type: getRadioValue("device_type") || '',
//             operating_system: getRadioValue("operating_system") || '',
//             user_flow: getRadioValue("user_flow") || '',
//             cta_placement: getRadioValue("cta_placement") || '',
//             interaction_requirements: getCheckboxValues("interaction_requirements"),
//             accessibility_needs: getRadioValue("accessibility_needs") || '',
//             design_preferences: getRadioValue("design_preferences") || '',
//             mood_feel: getRadioValue("mood_feel") || '',
//             imagery_iconography: getRadioValue("imagery_iconography") || ''
//         };
    
//         // Debug log the collected data
//         console.log("Collected form data:", JSON.stringify(data, null, 2));
    
//         // Validate before returning
//         const validation = validateFormData(data);
//         if (!validation.isValid) {
//             throw new Error(`Validation failed: ${validation.errors.join(', ')}`);
//         }
    
//         return data;
//     }

//     function submitColor() {
//         const color = document.getElementById('colorPicker').value;
//         console.log("Selected Color:", color);
//         // Submit the color code to the backend
//         fetch('/submit_color', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify({ color_code: color }),
//         })
//         .then(response => response.json())
//         .then(data => {
//             console.log("Response:", data);
//         })
//         .catch(error => {
//             console.error("Error:", error);
//         });
//     }

//     const pickr = Pickr.create({
//         el: '.color-picker',
//         theme: 'classic', // or 'monolith', 'nano'
//         swatches: ['#F44336', '#E91E63', '#9C27B0', '#673AB7'],
//         components: {
//             preview: true,
//             opacity: true,
//             hue: true,
//             interaction: {
//                 hex: true,
//                 rgba: true,
//                 input: true,
//                 save: true
//             }
//         }
//     });
    
//     pickr.on('save', (color, instance) => {
//         const colorHex = color.toHEXA().toString();
//         console.log("Selected Color:", colorHex);
//         // Send colorHex to backend
//     });

//     async function submitFormData(userInputs) {
//         try {
//             // Clean and validate the data before sending
//             const cleanedData = {
//                 industry_type: userInputs.industry_type,
//                 age_group: userInputs.age_group || "",
//                 gender: userInputs.gender || "",
//                 occupation: userInputs.occupation || "",
//                 location: userInputs.location || "",
//                 personas: Array.isArray(userInputs.personas) ? userInputs.personas : [],
//                 user_preferences: Array.isArray(userInputs.user_preferences) ? userInputs.user_preferences : [],
//                 content_type: userInputs.content_type,
//                 tone: userInputs.tone || "",
//                 content_structure: userInputs.content_structure || "",
//                 technical_proficiency: userInputs.technical_proficiency || "",
//                 cognitive_load: userInputs.cognitive_load || "",
//                 visual_impairment: userInputs.visual_impairment || "",
//                 motor_impairment: userInputs.motor_impairment || "",
//                 hearing_impairment: userInputs.hearing_impairment || "",
//                 accessibility_preferences: Array.isArray(userInputs.accessibility_preferences) ? userInputs.accessibility_preferences : [],
//                 color_code: userInputs.color_code || "",
//                 branding_constraints: Array.isArray(userInputs.branding_constraints) ? userInputs.branding_constraints : [],
//                 device_type: userInputs.device_type || "",
//                 operating_system: userInputs.operating_system || "",
//                 device_compatibility: Array.isArray(userInputs.device_compatibility) ? userInputs.device_compatibility : [],
//                 browser_compatibility: Array.isArray(userInputs.browser_compatibility) ? userInputs.browser_compatibility : [],
//                 user_flow: userInputs.user_flow || "",
//                 cta_placement: userInputs.cta_placement || "",
//                 core_features: Array.isArray(userInputs.core_features) ? userInputs.core_features : [],
//                 interaction_requirements: Array.isArray(userInputs.interaction_requirements) ? userInputs.interaction_requirements : [],
//                 accessibility_needs: userInputs.accessibility_needs || "",
//                 design_preferences: userInputs.design_preferences || "",
//                 mood_feel: userInputs.mood_feel || "",
//                 imagery_iconography: userInputs.imagery_iconography || ""
//             };
    
//             console.log("Original input data:", userInputs);
//             console.log("Cleaned data to send:", cleanedData);
    
//             // Verify required fields
//             const requiredFields = [
//                 'industry_type',
//                 'content_type',
//                 'personas',
//                 'user_preferences',
//                 'device_compatibility',
//                 'browser_compatibility',
//                 'core_features'
//             ];
    
//             const missingFields = requiredFields.filter(field => {
//                 const value = cleanedData[field];
//                 if (Array.isArray(value)) {
//                     return value.length === 0;
//                 }
//                 return !value || value === "";
//             });
    
//             if (missingFields.length > 0) {
//                 throw new Error(`Missing required fields: ${missingFields.join(', ')}`);
//             }
    
//             console.log("Making request to server with data:", JSON.stringify(cleanedData, null, 2));
    
//             const response = await fetch("http://localhost:5500/api/recommendations", {
//                 method: "POST",
//                 headers: { 
//                     "Content-Type": "application/json"
//                 },
//                 body: JSON.stringify(cleanedData)
//             });
    
//             const responseText = await response.text();
//             console.log("Full server response:", {
//                 status: response.status,
//                 statusText: response.statusText,
//                 headers: Object.fromEntries(response.headers.entries()),
//                 body: responseText
//             });
    
//             // Try to parse the response
//             let responseData;
//             try {
//                 responseData = JSON.parse(responseText);
//                 console.log("Parsed response data:", responseData);
//             } catch (e) {
//                 console.error("Failed to parse response:", e);
//                 throw new Error("Server returned invalid JSON");
//             }
    
//             if (!response.ok) {
//                 const errorMessage = responseData.error || 'Server error';
//                 console.error("Server error:", errorMessage);
//                 throw new Error(errorMessage);
//             }
    
//             return responseData.recommendations || [];
    
//         } catch (error) {
//             console.error("Submission error details:", {
//                 name: error.name,
//                 message: error.message,
//                 stack: error.stack
//             });
//             throw error;
//         }
//     }

//     // ==== EVENT LISTENERS ====
//     // Option box handling
//     document.querySelectorAll('.option-box').forEach(box => {
//         const input = box.querySelector('input[type="radio"]');
//         if (input) {
//             const groupName = input.name;
//             const value = input.value;
            
//             box.onclick = () => {
//                 const options = document.querySelectorAll(`[name="${groupName}"]`);
//                 options.forEach(opt => opt.parentElement.classList.remove('active'));
                
//                 input.checked = true;
//                 input.parentElement.classList.add('active');
//             };
//         }
//     });

//     // Navigation buttons
//     document.querySelectorAll(".next-btn").forEach(button => 
//         button.addEventListener("click", nextStep));
//     document.querySelectorAll(".back-btn").forEach(button => 
//         button.addEventListener("click", prevStep));

//     // Form submission
//     if (form) {
//         form.addEventListener("submit", async function(e) {
//             e.preventDefault();
//             try {
//                 // Debug form values before submission
//                 console.log("Form submission started");
//                 console.log("Form values:", {
//                     industry_type: getRadioValue("industry_type"),
//                     content_type: getRadioValue("content_type"),
//                     personas: getCheckboxValues("personas"),
//                     user_preferences: getCheckboxValues("user_preferences"),
//                     device_compatibility: getCheckboxValues("device_compatibility"),
//                     browser_compatibility: getCheckboxValues("browser_compatibility"),
//                     core_features: getCheckboxValues("core_features")
//                 });
    
//                 const formData = collectFormData();
//                 console.log("Collected and validated form data:", formData);
    
//                 const recommendations = await submitFormData(formData);
                
//                 localStorage.setItem("userInputs", JSON.stringify(formData));
//                 localStorage.setItem("recommendations", JSON.stringify(recommendations));
                
//                 window.location.href = "ResultsCard.html";
//             } catch (error) {
//                 console.error("Form submission failed:", error);
//                 showError(error.message);
//             }
//         });
//     }
// });