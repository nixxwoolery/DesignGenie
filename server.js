const express = require("express");
const cors = require("cors");
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files from 'public' folder
app.use(express.static("public"));

// Handle form submission
app.post("/submit", (req, res) => {
  console.log("Received form data:", req.body);
  const recommendations = [
    { guideline: "Use high-contrast text", details: "Helps accessibility." },
    { guideline: "Simplify navigation", details: "Improves user experience." }
  ];

  res.json({ success: true, recommendations });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});