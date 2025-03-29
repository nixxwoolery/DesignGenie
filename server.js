const express = require("express");
const cors = require("cors");
const sqlite3 = require("sqlite3").verbose();
const path = require("path");
const dbPath = path.resolve(__dirname, "designgenie.db");
const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error("Database connection error:", err.message);
  } else {
    console.log("Connected to the DesignGenie database.");
  }
});
const app = express();
const PORT = 5501;

// Middleware to parse JSON
app.use(express.json());

// Enable CORS
app.use(cors());

// Define a POST route for `/submit`
app.post("/submit", (req, res) => {
  const userInputs = req.body;

  // Example: Query based on a specific user input, such as platform or accessibility_need
  const query = `
    SELECT guideline, details 
    FROM Guidelines 
    WHERE keywords LIKE '%' || ? || '%' 
    LIMIT 5
  `;

  const inputKeyword = userInputs.platform || "accessibility"; // fallback for example

  db.all(query, [inputKeyword], (err, rows) => {
    if (err) {
      console.error("Database query error:", err.message);
      return res.status(500).json({ error: "Database error" });
    }

    const recommendations = rows.map(row => `${row.guideline}: ${row.details}`);

    res.status(200).json({
      message: "Form submitted successfully!",
      recommendations: recommendations
    });
  });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://127.0.0.1:${PORT}`);
});