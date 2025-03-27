const express = require("express");
const cors = require("cors");
const app = express();
const PORT = 5501;

// Middleware to parse JSON
app.use(express.json());

// Enable CORS
app.use(cors());

// Define a POST route for `/submit`
app.post("/submit", (req, res) => {
    console.log("Form Data Received:", req.body);
    res.status(200).json({
        message: "Form submitted successfully!",
        recommendations: ["Recommendation 1", "Recommendation 2"],
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://127.0.0.1:${PORT}`);
});