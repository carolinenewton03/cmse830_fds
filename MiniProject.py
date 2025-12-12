import random
import yt_dlp as ytdlp
import streamlit as st
st.cache_data.clear()
st.cache_resource.clear()
import pdfplumber
import pandas as pd
import base64
import re
import os
import spacy
from spacy.matcher import PhraseMatcher
from streamlit_tags import st_tags
import plotly.express as px
import plotly.graph_objects as go
from Courses import (
    ds_course,
    web_course,
    android_course,
    ios_course,
    uiux_course,
    software_engineering_courses,
    resume_videos,
    interview_videos,
)
from ML_EDA import render_ml_eda_page
from target_roles import target_roles_required_skills, role_skills, role_descriptions
from pymongo import MongoClient
import unicodedata
import certifi
from Admin import render_admin_panel
DATA_PATH = "user_data.csv"


THEME_BG = "#050022"          # deep navy background
THEME_TEXT = "#F5F7FF"        # soft white text

COLOR_PRIMARY = "#FF6AD5"     # neon pink
COLOR_SECONDARY = "#8BE9FF"   # cyan
COLOR_TERTIARY = "#C4A2FF"    # lavender
COLOR_QUATERNARY = "#6BFFB8"  # mint
COLOR_MUTED = "#2D3250"       # muted navy for less important stuff

def set_custom_theme():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {THEME_BG};
            color: {THEME_TEXT};
        }}

        /* Sidebar background + text */
        section[data-testid="stSidebar"] {{
            background-color: #151826;   /* slightly lighter than main bg */
            color: {THEME_TEXT};
        }}
        section[data-testid="stSidebar"] * {{
            color: {THEME_TEXT};
        }}

        /* Headings */
        h1, h2, h3, h4, h5, h6 {{
            color: {THEME_TEXT};
        }}

        /* Buttons (Read Me etc.) */
        .stButton>button {{
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(6px);
            color: {THEME_TEXT};
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.15);
            transition: 0.2s all ease-in-out;
        }}

        .stButton>button:hover {{
            background: rgba(255, 255, 255, 0.14);
            border-color: rgba(255, 255, 255, 0.25);
        }}

        .stButton>button:hover, .stDownloadButton>button:hover {{
            filter: brightness(1.1);
        }}

        /* Select boxes / inputs */
        div[data-baseweb="select"] > div {{
            background-color: #202538;
            color: {THEME_TEXT};
            border-radius: 8px;
        }}
        div[data-baseweb="select"] svg {{
            color: {THEME_TEXT};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
# --- 1. MongoDB Initialization (Atlas via Streamlit secrets) ---

mongo_client = None
user_collection = None

try:
    mongo_uri = st.secrets["MONGO"]["URI"]
    db_name = st.secrets["MONGO"]["DB_NAME"]
    collection_name = st.secrets["MONGO"]["COLLECTION"]

    mongo_client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
    mongo_db = mongo_client[db_name]
    user_collection = mongo_db[collection_name]
except Exception:
    user_collection = None  # run fine even if DB fails


# --- 2. SpaCy and Skill Matcher Initialization ---

model_name = "en_core_web_sm"
try:
    nlp = spacy.load(model_name)
except OSError:
    nlp = None

skills_list = [
    "Python",
    "SQL",
    "Power BI",
    "Pandas",
    "NumPy",
    "MS Office",
    "Canva",
    "Data Cleaning",
    "Data Visualization",
    "Written Communication",
    "Visual Storytelling",
    "Content Planning",
    "Social Media Analytics",
    "Deep Learning",
    "Machine Learning",
    "Looker",
    "Data Analytics",
    "HTML",
    "CSS",
    "JavaScript",
    "Git",
    "OOP",
    "APIs",
    "Unit Testing",
    "Version Control",
    "Agile",
    "CI/CD",
    "Docker",
    "Problem Solving",
    "Data Structures",
    "Algorithms",
    "Excel",
    "Tableau",
    "Statistics",
    "ETL",
    "Data Wrangling",
    "Matplotlib",
    "Seaborn",
    "Scikit-learn",
    "NLP",
    "DAX",
    "JIRA",
    "CRM",
    "Cisco",
    "Firewalls",
    "Troubleshooting",
    "TCP/IP",
    "Routing",
    "Switching",
    "DNS",
    "DHCP",
]

if nlp:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in skills_list]
    matcher.add("SkillList", patterns)
else:
    matcher = None


# --- 3. Utility Functions ---

def show_pdf(file):
    """Display uploaded PDF inside the app."""
    try:
        file.seek(0)
        base64_pdf = base64.b64encode(file.read()).decode("utf-8")
        pdf_display = (
            f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
            f'width="100%" height="800" type="application/pdf"></iframe>'
        )
        st.markdown(pdf_display, unsafe_allow_html=True)
        file.seek(0)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")


def pdf_reader(file):
    """Extract text from PDF using pdfplumber, fallback to raw read."""
    try:
        file.seek(0)
    except Exception:
        pass

    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        text = ""

    if not text:
        try:
            file.seek(0)
            raw = file.read()
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            pass
    return text


def extract_basic_info(text):
    lines = text.split("\n")
    name = "Not Found"

    # ---- NAME ----
    for line in lines:
        clean_line = " ".join(line.strip().split())
        if (
            clean_line
            and re.match(r"^[A-Za-z\s\.]+$", clean_line)
            and not any(
                word in clean_line.lower()
                for word in [
                    "contact",
                    "education",
                    "profile",
                    "objective",
                    "experience",
                    "skills",
                    "summary",
                    "work",
                    "certifications",
                    "projects",
                ]
            )
        ):
            words = clean_line.split()
            if words:
                # Heuristic: mostly single letters -> join them (e.g., C A R O L I N E)
                if sum(1 for w in words if len(w) == 1) / len(words) > 0.5 and len(words) > 2:
                    name = "".join(words).title()
                elif len(words) <= 4 and all(w.isalpha() or w == "." for w in words):
                    name = clean_line.title()

            if name != "Not Found":
                break

    # ---- EMAIL ----
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    email = email_match.group() if email_match else "Not Found"

    # ---- PHONE (more robust, international-friendly) ----
    # Strip emails and URLs first
    cleaned_text = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", text
    )
    cleaned_text = re.sub(
        r"(https?|ftp)://[^\s/$.?#].[^\s]*", "", cleaned_text
    )

    # Find sequences that look like phone numbers:
    # start with optional +, then digits with spaces/dashes/() allowed, at least ~10 chars total
    candidates = re.findall(r"\+?\d[\d\s\-\(\)]{8,}\d", cleaned_text)

    mobile = "Not Found"
    best_candidate = None
    best_len = 0

    for cand in candidates:
        digits = re.sub(r"\D", "", cand)
        # accept typical phone lengths (10–15 digits works for most countries)
        if 10 <= len(digits) <= 15 and len(digits) > best_len:
            best_len = len(digits)
            best_candidate = cand.strip()

    if best_candidate:
        mobile = best_candidate

    return {
        "name": name if name != "Not Found" else "N/A",
        "email": email if email != "Not Found" else "N/A",
        "mobile_number": mobile if mobile != "Not Found" else "N/A",
    }



def calculate_resume_score(basic_info, extracted_skills, total_keywords, total_structure_criteria):
    score = 0
    if basic_info["name"] != "N/A":
        score += 10
    if basic_info["email"] != "N/A":
        score += 5
    if basic_info["mobile_number"] != "N/A":
        score += 5
    score += min(len(extracted_skills) * 5, 50)
    score += total_structure_criteria * 10
    return min(score, 100)


def display_score_gauge(score):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": "Overall Resume Score",
                "font": {"size": 20, "color": THEME_TEXT},
            },
            gauge={
                "axis": {
                    "range": [None, 100],
                    "tickwidth": 1,
                    "tickcolor": "#A3A7C2",  # light grey ticks so they stand out
                },
                "bar": {"color": COLOR_PRIMARY},  # neon pink needle/bar
                "bgcolor": THEME_BG,
                "borderwidth": 2,
                "bordercolor": COLOR_MUTED,
                "steps": [
                    {"range": [0, 50], "color": "#4B1433"},   # dark magenta
                    {"range": [50, 75], "color": "#233B73"},  # dark indigo
                    {"range": [75, 100], "color": "#116451"}, # dark teal
                ],
                "threshold": {
                    "line": {"color": COLOR_SECONDARY, "width": 4},  # cyan tick at score
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        height=250,
        margin=dict(t=50, b=0, l=10, r=10),
        font=dict(color=THEME_TEXT),
    )

    st.plotly_chart(fig, use_container_width=True)


def display_skill_match_chart(match_score, missing_count, matched_count):
    total = matched_count + missing_count
    if total == 0:
        st.warning("No target skills defined for this role.")
        return

    # Percent values
    missing_score = 100 - match_score
    labels = ["Matched Skills", "Missing Skills"]
    values = [match_score, missing_score]

    # Colors (lavender for matched, mint for missing)
    colors = [COLOR_TERTIARY, COLOR_QUATERNARY]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker_colors=colors,
                textinfo="percent",          # show only percent inside
                textfont=dict(
                    color="#1A1A1A",        # <-- darker color (almost black)
                    size=18,                # bigger = more readable
                    family="Arial"          # clean readable font
                ),
                insidetextorientation="horizontal",
                hoverinfo="label+value+percent",
                showlegend=True,
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Skill Match Breakdown",
            "y": 0.92,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"color": THEME_TEXT},
        },
        height=350,
        margin=dict(t=60, b=40, l=10, r=10),
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        font=dict(color=THEME_TEXT),

        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def fetch_yt_thumbnail(link):
    try:
        if "youtube.com/watch?v=" in link:
            video_id = link.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in link:
            video_id = link.split("youtu.be/")[-1].split("?")[0]
        else:
            return None, None
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
        return thumbnail_url, link
    except Exception:
        return None, None


def course_recommender(extracted_skills, role):
    st.subheader("Courses & Certificates🎓 Recommendations")
    rec_course = []

    required_skills = role_skills.get(role, [])
    required_skills_lower = [s.lower() for s in required_skills]
    extracted_skills_lower = [s.lower() for s in extracted_skills]

    missing_skills_lower = [
        skill for skill in required_skills_lower if skill not in extracted_skills_lower
    ]

    course_set = set()

    for skill in missing_skills_lower:
        if skill in ["data analysis", "machine learning", "deep learning", "statistics", "tableau"]:
            course_set.update(tuple(course) for course in ds_course)
        if skill in ["web development", "javascript", "html", "css", "react.js"]:
            course_set.update(tuple(course) for course in web_course)
        if skill in ["android development", "java"]:
            course_set.update(tuple(course) for course in android_course)
        if skill in ["ios development", "swift"]:
            course_set.update(tuple(course) for course in ios_course)
        if skill in ["ui/ux", "design"]:
            course_set.update(tuple(course) for course in uiux_course)

        for key, courses in software_engineering_courses.items():
            if skill in [k.lower() for k in key.split(" & ")]:
                course_set.update(tuple(course) for course in courses)

    course_list = list(course_set)

    if not course_list:
        st.warning("No courses found for the missing skills.")
    else:
        no_of_reco = st.slider(
            "Choose Number of Course Recommendations:", 1, min(10, len(course_list)), 4
        )
        random.shuffle(course_list)

        for i, (c_name, c_link) in enumerate(course_list[:no_of_reco], 1):
            st.markdown(f"({i}) [{c_name}]({c_link})")
            rec_course.append(c_name)

    return rec_course


def extract_relevant_sections(text):
    text_upper = text.upper()
    start_keyword = "SKILLS"
    end_keyword = "PROJECTS"

    start_index = text_upper.find(start_keyword)
    end_index = text_upper.find(end_keyword)

    if start_index == -1:
        return text_upper

    if end_index != -1 and end_index > start_index:
        relevant_text = text_upper[start_index:end_index]
    else:
        relevant_text = text_upper[start_index:]

    return relevant_text


def extract_skills(resume_text):
    if nlp is None or matcher is None:
        return []

    doc = nlp(resume_text.lower())
    matches = matcher(doc)
    extracted_skills_set = set()

    for match_id, start, end in matches:
        span = doc[start:end]
        extracted_skills_set.add(span.text.title())

    text_lower = resume_text.lower()
    for skill in skills_list:
        if skill.lower() in text_lower:
            extracted_skills_set.add(skill)

    cleaned_skills = set()
    for skill in extracted_skills_set:
        if "Ms Office" in skill:
            cleaned_skills.add("MS Office")
        elif "Num Py" in skill:
            cleaned_skills.add("NumPy")
        else:
            cleaned_skills.add(skill)

    return sorted(list(cleaned_skills))


def determine_level(text, skills):
    text = text.lower()
    years = 0

    patterns = [
        r"(\d+)\s+years?\s+of\s+experience",
        r"experience\s+of\s+(\d+)\s+years?",
        r"(\d+)\s+years?\s+experience",
        r"(\d+)\s+yrs",
        r"(\d+)\+?\s+years",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            years = int(match.group(1))
            break

    if years >= 5:
        return "Advanced"
    elif 2 <= years < 5:
        return "Intermediate"

    skill_count = len(skills)
    if skill_count >= 10:
        return "Intermediate"
    elif skill_count >= 5:
        return "Fresher"
    else:
        return "Fresher"


def match_skills_for_role(extracted_skills, role):
    required_skills = role_skills.get(role, [])
    required_skills_normalized = [skill.lower() for skill in required_skills]
    extracted_skills_normalized = [skill.lower() for skill in extracted_skills]

    matched_skills = [
        skill for skill in extracted_skills_normalized if skill in required_skills_normalized
    ]
    missing_skills = [
        skill for skill in required_skills_normalized if skill not in extracted_skills_normalized
    ]

    matched_skills_original = [skill for skill in required_skills if skill.lower() in matched_skills]
    missing_skills_original = [skill for skill in required_skills if skill.lower() in missing_skills]

    match_score = (
        (len(matched_skills) / len(required_skills_normalized)) * 100
        if required_skills_normalized
        else 0
    )

    return matched_skills_original, match_score, missing_skills_original


def display_videos():
    st.subheader("Resume Building Tips 📋")
    resume_columns = st.columns(2)
    for idx, link in enumerate(resume_videos):
        thumbnail_url, video_url = fetch_yt_thumbnail(link)
        if thumbnail_url:
            with resume_columns[idx % 2]:
                st.markdown(
                    f'<a href="{video_url}" target="_blank"><img src="{thumbnail_url}" width="100%"></a>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning(f"Thumbnail not found for link: {link}")

    st.subheader("Interview Preparation 🎥")
    interview_columns = st.columns(2)
    for idx, link in enumerate(interview_videos):
        thumbnail_url, video_url = fetch_yt_thumbnail(link)
        if thumbnail_url:
            with interview_columns[idx % 2]:
                st.markdown(
                    f'<a href="{video_url}" target="_blank"><img src="{thumbnail_url}" width="100%"></a>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning(f"Thumbnail not found for link: {link}")


def is_resume(text):
    resume_keywords = [
        "experience",
        "education",
        "skills",
        "certifications",
        "projects",
        "summary",
        "contact",
    ]
    return any(keyword in text.lower() for keyword in resume_keywords)


def suggest_best_role(extracted_skills):
    if not extracted_skills:
        return None, 0.0
    best_role = None
    best_score = -1.0
    for role_name in target_roles_required_skills.keys():
        _, score, _ = match_skills_for_role(extracted_skills, role_name)
        if score > best_score:
            best_score = score
            best_role = role_name
    return best_role, best_score


# --- 4. Main Streamlit Application ---

def run():
    set_custom_theme()


    st.title("Smart Resume Analyser")

    st.sidebar.markdown("# Choose User")
    activities = ["Normal User", "Admin", "ML & EDA"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    st.sidebar.markdown("---")
    readme_clicked = st.sidebar.button("📘 Read Me")

    if "view" not in st.session_state:
        st.session_state["view"] = "Normal User"

    if readme_clicked:
        st.session_state["view"] = "Read Me"
    else:
        st.session_state["view"] = choice

    view = st.session_state["view"]

    try:
        # ---------- READ ME ----------
        if view == "Read Me":
            st.title("📘 Project Overview & Data Collection")
            st.markdown(
                """
            ### 📘 Smart Resume Analyser — Overview

            This app analyses your resume and provides:
            - Extracted skills  
            - Missing skills  
            - Best-fit job role  
            - Course recommendations  
            - Resume score  
            - Experience level  
            - Visual reports (donut chart, gauge chart, heatmap)  

            ---

            ### 🗂 Datasets Used

            This project is powered by **three datasets**:

            **1️⃣ Self-Collected Resume Dataset**  
            Real resumes collected from peers and converted into a structured CSV format.  
            Used to understand practical, modern resume styles.

            **2️⃣ Kaggle Resume Dataset**  
            A public dataset added to diversify:
            - resume writing patterns  
            - skill categories  
            - role structures  

            **3️⃣ App-Generated Dataset (`user_data.csv`)**  
            Every time a user analyses a resume, a new row is added containing:
            - Extracted information  
            - Role prediction  
            - Skill match %  
            - Missing skills  
            - Recommended courses  
            - Timestamp  

            This becomes the backend for the **Admin Dashboard**.

            ---

            ### 🧹 What the System Does

            - Reads PDF resumes  
            - Extracts Name, Email, and Phone (supports international formats)  
            - Uses NLP (spaCy) + keyword matching to detect skills  
            - Predicts best job role based on skill overlap  
            - Calculates:
            - Missing skills  
            - Skill match percentage  
            - Experience level  
            - Resume score  
            - Recommends courses (Data Science, Web Dev, Android, iOS, UI/UX)  
            - Displays:
            - Gauge chart  
            - Donut chart  
            - Skill heatmap  
            - Offers curated YouTube videos for resume & interview prep  

            ---

            ### 🧭 Admin Dashboard

            The Admin panel shows:
            - Unique users  
            - Role popularity  
            - Average resume scores  
            - Skill match distribution  
            - Experience level trends  
            - Filters + downloadable CSV  

            All analytics come directly from `user_data.csv`.

            ---

            ### 🎯 Goal of the Project

            To provide a **smarter, faster, and more personalized** way to evaluate resumes using NLP, visualization, and real user data — building a strong end-to-end Data Science project.

            More enhancements can be added as the project evolves!
            """
            )

        # ---------- NORMAL USER ----------
        elif view == "Normal User":
            pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])

            if pdf_file:
                st.subheader("Uploaded Resume")
                show_pdf(pdf_file)

                pdf_file.seek(0)
                resume_text = pdf_reader(pdf_file)

                st.header("Resume Analysis")

                if not is_resume(resume_text):
                    st.error("Uploaded file does not appear to be a resume. Please upload a valid resume.")
                else:
                    st.success("Resume successfully read!")

                    st.subheader("Extracted Text Preview")
                    st.text_area(
                        "Raw Text Used for Analysis",
                        value=resume_text,
                        height=200,
                        help="This is the raw text extracted from your PDF.",
                    )

                    basic_info = extract_basic_info(resume_text)

                    if basic_info:
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.subheader("Basic Info")
                            st.markdown(f"*Name*: **{basic_info['name']}**")
                            st.markdown(f"*Email*: **{basic_info['email']}**")
                            st.markdown(f"*Mobile Number*: **{basic_info['mobile_number']}**")

                            relevant_text = extract_relevant_sections(resume_text)
                            extracted_skills = extract_skills(
                                relevant_text if relevant_text else resume_text
                            )

                            total_keywords = 20
                            total_structure_criteria = 3
                            resume_score = calculate_resume_score(
                                basic_info,
                                extracted_skills,
                                total_keywords,
                                total_structure_criteria,
                            )
                            experience_level = determine_level(resume_text, extracted_skills)

                        with col2:
                            display_score_gauge(resume_score)
                            st.subheader("Experience Level")
                            st.markdown(f"**{experience_level}**")

                        st.markdown("---")
                        st.subheader("Choose Target Role")

                        role = st.selectbox(
                            "Select Role for Analysis",
                            list(target_roles_required_skills.keys()),
                        )

                        best_role, best_role_score = suggest_best_role(extracted_skills)
                        if best_role:
                            st.caption(
                                f"Suggested best match based on your skills: "
                                f"**{best_role}** ({best_role_score:.1f}% skill overlap)"
                            )

                        st.subheader("Role Description")
                        st.write(
                            role_descriptions.get(
                                role,
                                f"No description available for **{role}**.",
                            )
                        )

                        st.markdown("---")
                        st.subheader("Role Analysis")

                        matched_skills, match_score, missing_skills = match_skills_for_role(
                            extracted_skills, role
                        )

                        col3, col4 = st.columns([1.5, 1])

                        with col3:
                            st.subheader("Skills Overview")
                            st.markdown(
                                f"**Extracted Skills** ({len(extracted_skills)}): "
                                f"{', '.join(extracted_skills)}"
                            )
                            st.markdown(
                                f"**Matched Skills** ({len(matched_skills)}): "
                                f"{', '.join(matched_skills)}"
                            )
                            st.markdown(
                                f"**Skill Match Percentage**: **{match_score:.2f}%**"
                            )

                            st.subheader("Missing Skills (Recommended Focus)")
                            if missing_skills:
                                st.markdown(", ".join(missing_skills))
                            else:
                                st.success("You possess all target skills for this role!")

                        with col4:
                            st.markdown("<div style='padding-left: 15px;'>", unsafe_allow_html=True)
                            display_skill_match_chart(
                                match_score,
                                len(missing_skills),
                                len(matched_skills),
                            )
                            st.markdown("</div>", unsafe_allow_html=True)



                        # =============================
                        # Skill Presence Heatmap
                        # =============================
                        st.markdown("---")
                        st.subheader("Skill Presence Heatmap")

                        required = role_skills.get(role, [])
                        required_lower = [s.lower() for s in required]
                        extracted_lower = [s.lower() for s in extracted_skills]

                        presence = [
                            1 if skill.lower() in extracted_lower else 0
                            for skill in required
                        ]

                        if required:
                            fig_heat = px.imshow(
                            [presence],
                            labels=dict(x="Skills", y="Resume", color="Present"),
                            x=required,
                            y=["Resume"],
                            color_continuous_scale=[COLOR_PRIMARY, COLOR_SECONDARY],  
                        )



                            fig_heat.update_layout(
                                title="Skill Presence Heatmap",
                                height=250,
                                margin=dict(l=20, r=20, t=40, b=20),
                                paper_bgcolor=THEME_BG,
                                plot_bgcolor=THEME_BG,
                                font=dict(color=THEME_TEXT),
                            )

                            st.plotly_chart(fig_heat, use_container_width=True)

                            st.caption(
                                "This heatmap shows which required skills for the selected role are present "
                                "(cyan) and which are missing (pink) in your resume."
                            )
                        else:
                            st.info("No required skills defined for this role to build a heatmap.")

                        st.markdown("---")

                        rec_courses = course_recommender(extracted_skills, role)
                        st.markdown("---")
                        display_videos()

                                                # Try to save, but don't show any DB message to the user
                        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        user_data = {
                            "Name": basic_info["name"],
                            "Email_ID": basic_info["email"],
                            "resume_score": resume_score,
                            "matching_score": f"{match_score:.2f}%",
                            "Timestamp": timestamp,
                            "Page_no": "N/A",
                            "Predicted_Field": role,
                            "User_level": experience_level,
                            "Actual_skills": extracted_skills,
                            "Recommended_skills": list(set(matched_skills)),
                            "Recommended_courses": rec_courses,
                        }

                        # ---- SAVE TO CSV FOR ADMIN DASHBOARD ----
                        df_row = pd.DataFrame([user_data])

                        if os.path.exists(DATA_PATH):
                            try:
                                existing = pd.read_csv(DATA_PATH)
                                combined = pd.concat([existing, df_row], ignore_index=True)
                            except Exception:
                                # if file is corrupted, just overwrite with latest row
                                combined = df_row
                        else:
                            combined = df_row

                        combined.to_csv(DATA_PATH, index=False)

                        # (Optional) also try Mongo, but only for debug—NOT required for admin
                        if user_collection is not None:
                            try:
                                result = user_collection.insert_one(user_data)
                                if st.session_state.get("debug"):
                                    st.caption(
                                        f"(debug) Saved analysis to Mongo with id: {result.inserted_id}"
                                    )
                            except Exception as e:
                                if st.session_state.get("debug"):
                                    st.error(f"(debug) Failed to save data to Mongo: {e}")

                # ---------- ADMIN DASHBOARD ----------
        elif view == "Admin":
            render_admin_panel(user_collection)

            # ---------- ML + EDA PAGE ----------
        elif view == "ML & EDA":
            render_ml_eda_page()

    except Exception as e:
        st.error(f"An application error occurred: {e}")

if __name__ == "__main__":
    run()
