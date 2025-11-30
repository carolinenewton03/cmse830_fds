📘 Smart Resume Analyser — End-to-End NLP + Streamlit Project

The Smart Resume Analyser is an AI-powered application that reads resumes, extracts key information, compares skills to job roles, identifies gaps, recommends courses, and provides visual insights.
It is built with NLP, Python, spaCy, Plotly, and Streamlit, and includes an Admin Dashboard powered by real user activity logs.

⭐ Key Features

PDF Resume Text Extraction
Name, Email & Phone Number Parsing (supports international formats)
Skill Extraction using spaCy + Keyword Matching
Automatic Job Role Matching
Skill Gap Detection
Course Recommendations
Resume Scoring (0-100)
Experience Level Prediction
Visual Insights (Gauge, Donut Chart, Heatmap)
YouTube Resume + Interview Tips
Admin Analytics Dashboard

App-Generated Dataset (user_data.csv)
🗂 Data Sources (3 Total)

1️⃣ Self-Collected Resume Dataset
Real resumes collected locally and converted into structured CSV format.

2️⃣ Kaggle Resume Dataset
Public dataset added to diversify writing styles and job categories.

3️⃣ App-Generated Dataset
Every resume analyzed in the app is appended to:

user_data.csv
This powers the Admin Dashboard.

🔧 Data Cleaning & Processing

Performed on all datasets:
Cleaning missing values
Removing duplicates
Standardizing skill names
Normalizing text
Phone number unification
Email extraction patterns
Combining datasets for richer skill mapping

📊 Exploratory Data Analysis

The following analyses were performed:
Skill frequency distribution
Resume length analysis
Role-wise missing skill patterns
Heatmaps
Donut chart of matched/missing skills
Gauge chart for resume score

🤖 NLP Resume Analysis Pipeline

Text Extraction (pdfplumber)
Basic Info Detection
Skill Extraction (spaCy PhraseMatcher + manual keyword rules)
Experience Level Detection
Resume Scoring System
Best Role Suggestion
Skill Gap Analysis
Course Recommendations

🎓 Course Recommendation Engine

Based on missing skills, courses are recommended from:
Data Science
Web Dev
Android
iOS
UI/UX
Software Engineering
Resume/Interview YouTube videos

🧭 Admin Dashboard

Admin panel displays:
Total analyses
Unique users
Average resume score
Most popular roles
Most common missing skills
CSV download
All using live data from user_data.csv.

🚀 Tech Stack

Python
Streamlit
spaCy
Pandas
Plotly
pdfplumber
yt_dlp
HTML/CSS Theming
CSV as lightweight data store

🚀 Deployment

The app runs on:
Streamlit Cloud
Local execution
MongoDB support removed due to SSL issues; CSV logging is used for stability.

📝 Future Improvements

Add embeddings-based skill extraction
Add ML job role classifier
Add user login / personalized dashboard
Add resume rewrite suggestions
