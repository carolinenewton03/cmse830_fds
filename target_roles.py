# target_roles.py
# -----------------------------
# Role descriptions (shown in UI)
# -----------------------------
role_descriptions = {
    "Data Analyst": (
        "Data Analysts clean, transform, and analyze data to answer business questions. "
        "They work heavily with SQL, spreadsheets, and BI tools to build reports and dashboards."
    ),
    "Data Scientist": (
        "Data Scientists build statistical and machine learning models to solve complex problems. "
        "They work with Python, SQL, ML libraries, and experimentation to derive actionable insights."
    ),
    "Machine Learning Engineer": (
        "Machine Learning Engineers productionize ML models. They focus on scalable pipelines, "
        "APIs, MLOps, and monitoring, combining ML knowledge with strong software engineering skills."
    ),
    "Software Engineer": (
        "Software Engineers design, build, test, and maintain software systems. "
        "They work extensively with programming languages, data structures, algorithms, "
        "version control, and scalable application development."
    ),
    "Business Analyst": (
        "Business Analysts translate business needs into data and system requirements. "
        "They work with stakeholders, write specifications, and support decision-making using data."
    ),
    "Web Developer": (
        "Web Developers build and maintain web applications using HTML, CSS, JavaScript, and frameworks. "
        "They focus on responsive UI, APIs, performance, and browser compatibility."
    ),
}


# -----------------------------
# Full skill universe per role
# (used for match %, missing skills, suggestions)
# -----------------------------
role_skills = {
    "Data Analyst": [
        # Core tools
        "SQL",
        "Excel",
        "Google Sheets",
        "Power BI",
        "Tableau",
        # Programming / data manipulation
        "Python",
        "R",
        "Pandas",
        "NumPy",
        # Analytics / stats
        "Statistics",
        "Hypothesis Testing",
        "A/B Testing",
        "Data Cleaning",
        "Data Wrangling",
        "Exploratory Data Analysis",
        # Visualization
        "Data Visualization",
        "Dashboarding",
        # Extras
        "ETL",
        "Reporting",
        "Business Intelligence",
    ],
    "Data Scientist": [
        # Programming / data
        "Python",
        "R",
        "SQL",
        "Pandas",
        "NumPy",
        "SciPy",
        # ML / modelling
        "Machine Learning",
        "Supervised Learning",
        "Unsupervised Learning",
        "Scikit-learn",
        "TensorFlow",
        "PyTorch",
        "XGBoost",
        "Model Evaluation",
        "Cross Validation",
        # Statistics / math
        "Statistics",
        "Probability",
        "Linear Algebra",
        "Optimization",
        # Data prep / viz
        "Data Cleaning",
        "Feature Engineering",
        "Data Visualization",
        "Matplotlib",
        "Seaborn",
        # Extras
        "NLP",
        "Time Series",
        "Experimentation",
        "A/B Testing",
        "Git",
        "Version Control",
    ],
    "Machine Learning Engineer": [
        # Programming / core
        "Python",
        "SQL",
        "OOP",
        "APIs",
        "REST",
        "Git",
        "Version Control",
        # ML stack
        "Machine Learning",
        "Deep Learning",
        "Scikit-learn",
        "TensorFlow",
        "PyTorch",
        "Model Serving",
        # MLOps / infra
        "MLOps",
        "Docker",
        "Kubernetes",
        "CI/CD",
        "ML Pipelines",
        "Cloud Computing",
        "AWS",
        "GCP",
        "Azure",
        # Quality
        "Unit Testing",
        "Integration Testing",
        "Monitoring",
        "Logging",
    ],
    "Software Engineer": [
        # Languages (software, not analyst junk)
        "Python",
        "Java",
        "C++",
        "JavaScript",
        "TypeScript",
        # Fundamentals
        "Data Structures",
        "Algorithms",
        "Object Oriented Programming",
        "OOP",
        "Design Patterns",
        "System Design",
        # Web / backend
        "APIs",
        "REST",
        "GraphQL",
        "Databases",
        "SQL",
        "NoSQL",
        # Tooling
        "Git",
        "Version Control",
        "Linux",
        "Unit Testing",
        "Integration Testing",
        "CI/CD",
        "Docker",
        "Debugging",
        # Extras
        "Agile",
        "Scrum",
        "Microservices",
    ],
    "Business Analyst": [
        # Data tools
        "Excel",
        "Google Sheets",
        "SQL",
        "Power BI",
        "Tableau",
        # Analysis
        "Data Analysis",
        "Reporting",
        "Dashboarding",
        "Requirements Gathering",
        "Process Mapping",
        "Gap Analysis",
        # Soft / domain skills (kept minimal)
        "Stakeholder Management",
        "Documentation",
        "Business Requirements",
        "Use Cases",
        "User Stories",
    ],
    "Web Developer": [
        # Core web
        "HTML",
        "CSS",
        "JavaScript",
        "TypeScript",
        # Frontend frameworks
        "React",
        "Angular",
        "Vue.js",
        # Styling / UI
        "Responsive Design",
        "Bootstrap",
        "Tailwind CSS",
        # Backend basics
        "APIs",
        "REST",
        "Node.js",
        "Express",
        "Databases",
        "SQL",
        # Tooling
        "Git",
        "Version Control",
        "Webpack",
        "Vite",
        # Quality
        "Unit Testing",
        "Debugging",
        "Performance Optimization",
    ],
}


# -----------------------------
# Core / must-have skills per role
# (used for stronger eligibility checks)
# -----------------------------
target_roles_required_skills = {
    "Data Analyst": [
        "SQL",
        "Excel",
        "Power BI",
        "Tableau",
        "Data Cleaning",
        "Data Visualization",
        "Statistics",
    ],
    "Data Scientist": [
        "Python",
        "SQL",
        "Machine Learning",
        "Statistics",
        "Pandas",
        "NumPy",
        "Scikit-learn",
    ],
    "Machine Learning Engineer": [
        "Python",
        "Machine Learning",
        "APIs",
        "Docker",
        "Git",
        "CI/CD",
    ],
    "Software Engineer": [
        "Python",      # OR Java / C++ – we just treat any of these as strong signal
        "Java",
        "C++",
        "Data Structures",
        "Algorithms",
        "OOP",
        "Git",
    ],
    "Business Analyst": [
        "Excel",
        "SQL",
        "Power BI",
        "Requirements Gathering",
        "Stakeholder Management",
    ],
    "Web Developer": [
        "HTML",
        "CSS",
        "JavaScript",
        "React",   # or other modern framework, but React is the default
        "Git",
    ],
}
