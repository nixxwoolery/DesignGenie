const express = require("express");
const cors = require("cors");
const sqlite3 = require("sqlite3").verbose();
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Connect to SQLite DB
const dbPath = path.join(__dirname, "designgenie.db");
const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error("Error connecting to DB:", err.message);
  } else {
    console.log("Connected to designgenie.db");
  }
});

// Submit route
app.post("/submit", (req, res) => {
  const userInputs = req.body;

  const inputKeyword = userInputs.platform || "accessibility"; // fallback

  const query = `
    SELECT guideline, details 
    FROM Guidelines 
    WHERE keywords LIKE '%' || ? || '%' 
    LIMIT 5
  `;

  db.all(query, [inputKeyword], (err, rows) => {
    if (err) {
      console.error("Database query error:", err.message);
      return res.status(500).json({ error: "Database error" });
    }
  
    console.log("Query returned rows:", rows); // <-- Add this
  
    const recommendations = rows.map((row) => {
      console.log("Row keys:", Object.keys(row)); // Logs exact property names
      const guideline = row.guideline || row.Guideline || '[No guideline]';
      const details = row.details || row.Details || '[No details]';
      return `${guideline}: ${details}`;
      console.log("Recommendations:", recommendations);
    });
  
    res.status(200).json({
      message: "Form submitted successfully!",
      recommendations: recommendations,
    });
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});