import random
import yt_dlp as ytdlp
import streamlit as st
import pdfplumber
import pandas as pd
import base64
import re
import spacy
from spacy.matcher import PhraseMatcher
from streamlit_tags import st_tags
import plotly.express as px # Import Plotly for Gauge/Donut charts
import plotly.graph_objects as go # For Gauge chart
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
from Admin import admin_panel  # Import admin_panel from admin.py
from target_roles import target_roles_required_skills, role_skills, role_descriptions
import firebase_admin
from firebase_admin import credentials, firestore
import json
import tempfile
import unicodedata

# --- 1. Firebase Initialization ---
if not firebase_admin._apps:
    try:
        # Load Firebase config from Streamlit secrets
        firebase_config = {
            "type": st.secrets["FIREBASE"]["type"],
            "project_id": st.secrets["FIREBASE"]["project_id"],
            "private_key_id": st.secrets["FIREBASE"]["private_key_id"],
            "private_key": st.secrets["FIREBASE"]["private_key"].replace("\\n", "\n"),
            "client_email": st.secrets["FIREBASE"]["client_email"],
            "client_id": st.secrets["FIREBASE"]["client_id"],
            "auth_uri": st.secrets["FIREBASE"]["auth_uri"],
            "token_uri": st.secrets["FIREBASE"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["FIREBASE"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["FIREBASE"]["client_x509_cert_url"],
            "universe_domain": st.secrets["FIREBASE"]["universe_domain"]
        }
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}. Check your `secrets.toml` file.")
db = firestore.client()

# --- 2. SpaCy and Skill Matcher Initialization ---
model_name = "en_core_web_sm"
try:
    nlp = spacy.load(model_name)
except OSError:
    st.warning(f"{model_name} not found. Please ensure it’s preinstalled.")
    nlp = None
    
# Master skills list for spaCy matching (Expanded for robustness)
skills_list = [
    'Python', 'SQL', 'Power BI', 'Pandas', 'NumPy', 'MS Office', 'Canva',
    'Data Cleaning', 'Data Visualization', 'Written Communication', 'Visual Storytelling',
    'Content Planning', 'Social Media Analytics', 'Deep Learning', 'Machine Learning', 
    'Looker', 'Data Analytics', 'HTML', 'CSS', 'JavaScript', 'Git', 'OOP', 'APIs',
    'Unit Testing', 'Version Control', 'Agile', 'CI/CD', 'Docker', 'Problem Solving',
    'Data Structures', 'Algorithms', 'Excel', 'Tableau', 'Statistics', 'ETL', 'Data Wrangling',
    'Matplotlib', 'Seaborn', 'Scikit-learn', 'NLP', 'DAX', 'JIRA', 'CRM', 'Cisco', 'Firewalls',
    'Troubleshooting', 'TCP/IP', 'Routing', 'Switching', 'DNS', 'DHCP' # Added from target_roles
]

# Initialize PhraseMatcher and add patterns
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
if nlp:
    patterns = [nlp.make_doc(text) for text in skills_list]
    matcher.add("SkillList", patterns)

# --- 3. Utility Functions ---

# Function to read and display PDF safely
def show_pdf(file):
    try:
        file.seek(0)
        base64_pdf = base64.b64encode(file.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        file.seek(0)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

# Extract text from PDF using pdfplumber
def pdf_reader(file):
    try:
        file.seek(0)
    except Exception:
        pass

    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    # final fallback
    if not text:
        try:
            file.seek(0)
            raw = file.read()
            text = raw.decode('utf-8', errors='ignore')
        except Exception:
            pass
    return text


def extract_basic_info(text):
    lines = text.split("\n")
    name = "Not Found"

    for line in lines:
        clean_line = ' '.join(line.strip().split())
        if (clean_line and re.match(r"^[A-Za-z\s\.]+$", clean_line)
                and not any(word in clean_line.lower() for word in ["contact", "education", "profile", "objective", "experience", "skills", "summary", "work", "certifications", "projects"])):
            
            words = clean_line.split()
            if words:
                # Heuristic: If line contains mostly single letters, join them (e.g., C A R O L I N E)
                if sum(1 for w in words if len(w) == 1) / len(words) > 0.5 and len(words) > 2:
                    name = ''.join(words).title()
                # Otherwise, treat it as a standard name line
                elif len(words) <= 4 and all(w.isalpha() or w == '.' for w in words):
                    name = clean_line.title()
            
            if name != "Not Found":
                break
            
    # Email extraction (most reliable)
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    email = email_match.group() if email_match else "Not Found"

    # Mobile number extraction (more robust: exclude emails and URLs first)
    cleaned_text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text) # Remove emails
    cleaned_text = re.sub(r'(https?|ftp)://[^\s/$.?#].[^\s]*', '', cleaned_text) # Remove URLs

    phone_match = re.search(
        r'(\+?\d{1,4}[.\-\s]?)?(\(?\d{3}\)?[.\-\s]?\d{3}[.\-\s]?\d{4})|(\d{10,15})',
        cleaned_text
    )
    mobile = "Not Found"
    if phone_match:
        number = next((g for g in phone_match.groups() if g), None)
        if number:
            # Clean up number and check length
            digits = re.sub(r'[^0-9]', '', number)
            if 10 <= len(digits) <= 15:
                mobile = number
    
    return {
        "name": name if name != "Not Found" else "N/A",
        "email": email if email != "Not Found" else "N/A",
        "mobile_number": mobile if mobile != "Not Found" else "N/A"
    }


# Function to calculate resume score
def calculate_resume_score(basic_info, extracted_skills, total_keywords, total_structure_criteria):
  score = 0
  if basic_info['name'] != "N/A":
      score += 10
  if basic_info['email'] != "N/A":
      score += 5
  if basic_info['mobile_number'] != "N/A":
      score += 5
      
  # 5 points per extracted skill up to 10 skills (Max 50 points)
  score += min(len(extracted_skills) * 5, 50)
  
  # Bonus for structure/keywords (Max 30 points)
  # Assuming 3 criteria (e.g., Education found, Experience found, Projects found)
  score += total_structure_criteria * 10 
  
  max_score = 100
  normalized_score = min(score, max_score)
  return normalized_score


# Function to display the Resume Score Gauge (Visualization 1)
def display_score_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Resume Score", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'red'},
                {'range': [50, 75], 'color': 'yellow'},
                {'range': [75, 100], 'color': 'green'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score}}))

    fig.update_layout(height=250, margin=dict(t=50, b=0, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

# Function to display the Skill Match Donut Chart (Visualization 2)
def display_skill_match_chart(match_score, missing_count, matched_count):
    
    # Calculate percentage for missing skills
    total = matched_count + missing_count
    if total == 0:
        st.warning("No target skills defined for this role.")
        return

    missing_score = 100 - match_score

    labels = ['Matched Skills', 'Missing Skills']
    values = [match_score, missing_score]
    
    # Custom colors: green for matched, grey/red for missing
    colors = ['#1f77b4', '#d62728'] if missing_count > 0 else ['#1f77b4', 'lightgray']

    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.5, 
        marker_colors=colors,
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title={
            'text': "Skill Match Breakdown",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        height=350,
        margin=dict(t=50, b=10, l=10, r=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)


# Function to fetch the thumbnail URL and video URL from the YouTube link
def fetch_yt_thumbnail(link):
  try:
      # Extract video ID from the link
      if "youtube.com/watch?v=" in link:
          video_id = link.split("v=")[-1].split("&")[0]
      elif "youtu.be/" in link:
          video_id = link.split("youtu.be/")[-1].split("?")[0]
      else:
          return None, None # Invalid link format

      thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
      return thumbnail_url, link
  except Exception:
      return None, None

# Recommending Courses based on Missing Skills
def course_recommender(extracted_skills, role):
  st.subheader("Courses & Certificates🎓 Recommendations")
  rec_course = []

  required_skills = role_skills.get(role, [])
  required_skills_lower = [s.lower() for s in required_skills]
  extracted_skills_lower = [s.lower() for s in extracted_skills]

  # Identify missing skills by comparing lowercased lists
  missing_skills_lower = [skill for skill in required_skills_lower if skill not in extracted_skills_lower]

  course_set = set()

  for skill in missing_skills_lower:
      if skill in ['data analysis', 'machine learning', 'deep learning', 'statistics', 'tableau']:
          course_set.update(tuple(course) for course in ds_course)

      if skill in ['web development', 'javascript', 'html', 'css', 'react.js']:
          course_set.update(tuple(course) for course in web_course)

      if skill in ['android development', 'java']:
          course_set.update(tuple(course) for course in android_course)

      if skill in ['ios development', 'swift']:
          course_set.update(tuple(course) for course in ios_course)

      if skill in ['ui/ux', 'design']:
          course_set.update(tuple(course) for course in uiux_course)

      # Check for software engineering relevant skills
      for key, courses in software_engineering_courses.items():
          if skill in [k.lower() for k in key.split(' & ')]: # Check against normalized keys
              course_set.update(tuple(course) for course in courses)

  course_list = list(course_set)

  if not course_list:
      st.warning("No courses found for the missing skills.")
  else:
      no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, min(10, len(course_list)), 4)
      random.shuffle(course_list)

      for i, (c_name, c_link) in enumerate(course_list[:no_of_reco], 1):
          st.markdown(f"({i}) [{c_name}]({c_link})")
          rec_course.append(c_name)

  return rec_course


def extract_relevant_sections(text):
    """
    Extracts text from SKILLS through CERTIFICATIONS by capturing all content 
    between 'SKILLS' and 'PROJECTS' to ensure all skills are included.
    """
    text = text.upper() # Standardize case for searching
    start_keyword = 'SKILLS'
    end_keyword = 'PROJECTS'

    start_index = text.find(start_keyword)
    end_index = text.find(end_keyword)

    if start_index == -1:
        return text # Fallback to full text if SKILLS isn't found
        
    # Capture text from start_keyword up to end_keyword (if found) or end of text
    if end_index != -1 and end_index > start_index:
        relevant_text = text[start_index:end_index]
    else:
        relevant_text = text[start_index:]
        
    return relevant_text


def extract_skills(resume_text):
    """Extracts skills using the pre-initialized spaCy PhraseMatcher."""
    global skills_list # Use the global list for fallback checking
    
    # Process the text once (all matching is case-insensitive due to attr="LOWER")
    doc = nlp(resume_text.lower())
    
    # 1. Use the PhraseMatcher to find multi-word and complex skill matches
    matches = matcher(doc)
    extracted_skills_set = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        extracted_skills_set.add(span.text.title())

    # 2. Fallback for single-token skills (like 'SQL' or 'Python') that might be missed 
    # and clean up case (e.g., ensure 'sql' becomes 'SQL')
    text_lower = resume_text.lower()
    for skill in skills_list:
        if skill.lower() in text_lower:
            extracted_skills_set.add(skill) # Add the capitalized/correct case version

    # Clean up results (e.g., normalize 'Ms Office' to 'MS Office')
    # This is a bit manual but ensures consistent output presentation
    cleaned_skills = set()
    for skill in extracted_skills_set:
        if 'Ms Office' in skill:
            cleaned_skills.add('MS Office')
        elif 'Num Py' in skill:
            cleaned_skills.add('NumPy')
        else:
            cleaned_skills.add(skill)

    # Return sorted list of unique skills
    return sorted(list(cleaned_skills))


# Function to determine experience level (Fresher, Intermediate, Advanced)
def determine_level(text, skills):
    import re
    from datetime import datetime

    text = text.lower()
    years = 0

    # 1. Look for explicit years of experience
    patterns = [
        r"(\d+)\s+years?\s+of\s+experience",
        r"experience\s+of\s+(\d+)\s+years?",
        r"(\d+)\s+years?\s+experience",
        r"(\d+)\s+yrs",
        r"(\d+)\+?\s+years"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            years = int(match.group(1))
            break
            
    # 2. If explicit years found, use them
    if years >= 5:
        return "Advanced"
    elif 2 <= years < 5:
        return "Intermediate"
        
    # 3. Fallback to skill count and project-based experience if years are low or not found
    skill_count = len(skills)
    
    # Prioritize skill depth for Fresher/Intermediate split
    if skill_count >= 10:
        return "Intermediate" # High skill volume suggests preparedness, but cap at Intermediate without tenure
    elif skill_count >= 5:
        return "Fresher" # More than a handful of skills
    else:
        return "Fresher" # Default


# Function to calculate skill match for a specified role
def match_skills_for_role(extracted_skills, role):
    required_skills = role_skills.get(role, [])

    # Normalize both required skills and extracted skills to ensure matching
    required_skills_normalized = [skill.lower() for skill in required_skills]
    extracted_skills_normalized = [skill.lower() for skill in extracted_skills]

    # Identify matched and missing skills
    matched_skills = [skill for skill in extracted_skills_normalized if skill in required_skills_normalized]
    
    # Identify missing skills only from the required list
    missing_skills = [skill for skill in required_skills_normalized if skill not in extracted_skills_normalized]

    # Preserve original casing for output
    # Map matched skills back to original casing used in the required_skills list for clean output
    matched_skills_original = [skill for skill in required_skills if skill.lower() in matched_skills]
    
    # Map missing skills back to original casing
    missing_skills_original = [skill for skill in required_skills if skill.lower() in missing_skills]

    # Calculate skill match score
    match_score = (len(matched_skills) / len(required_skills_normalized)) * 100 if required_skills_normalized else 0

    return matched_skills_original, match_score, missing_skills_original


# Display resume tips and interview videos with thumbnails in a two-column layout
def display_videos():
  st.subheader("Resume Building Tips 📋")
  resume_columns = st.columns(2)
  for idx, link in enumerate(resume_videos):
      thumbnail_url, video_url = fetch_yt_thumbnail(link)
      if thumbnail_url:
          with resume_columns[idx % 2]:
              st.markdown(
                  f'<a href="{video_url}" target="_blank"><img src="{thumbnail_url}" width="100%"></a>',
                  unsafe_allow_html=True
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
                  unsafe_allow_html=True
              )
      else:
          st.warning(f"Thumbnail not found for link: {link}")


def is_resume(text):
  # Simple heuristic: checking for common sections in resumes
  resume_keywords = ["experience", "education", "skills", "certifications", "projects", "summary", "contact"]
  return any(keyword in text.lower() for keyword in resume_keywords)


# --- 4. Main Streamlit Application ---
def run():
   st.title("Smart Resume Analyser")
   st.sidebar.markdown("# Choose User")
   activities = ["Normal User", "Admin"]
   choice = st.sidebar.selectbox("Choose among the given options:", activities)

   try:
    users_ref = db.collection("user_data")

    if choice == 'Normal User':
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        
        if pdf_file:
            # 1. Display PDF Preview (Under Drag and Drop)
            with st.expander("View Uploaded Resume"):
                 show_pdf(pdf_file)
                 
            pdf_file.seek(0)
            resume_text = pdf_reader(pdf_file)
            
            st.header("Resume Analysis")

            if not is_resume(resume_text):
                st.error("Uploaded file does not appear to be a resume. Please upload a valid resume.")
            else:
                st.success("Resume successfully read!")
                
                # 2. Display Extracted Resume Text (Under Success Message)
                st.subheader("Extracted Text Preview")
                st.text_area("Raw Text Used for Analysis", value=resume_text, height=200, help="This is the raw text extracted from your PDF.")

                basic_info = extract_basic_info(resume_text)

                if basic_info:
                    # --- Basic Info and Score Gauge in Columns ---
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.subheader("Basic Info")
                        st.markdown(f"*Name*: **{basic_info['name']}**")
                        st.markdown(f"*Email*: **{basic_info['email']}**")
                        st.markdown(f"*Mobile Number*: **{basic_info['mobile_number']}**")
                        
                        relevant_text = extract_relevant_sections(resume_text)
                        # Pass only the resume_text to extract_skills, which uses the global matcher
                        extracted_skills = extract_skills(relevant_text if relevant_text else resume_text)
                        
                        total_keywords = 20
                        total_structure_criteria = 3 # Placeholder for checking sections like Education, Experience, etc.
                        resume_score = calculate_resume_score(basic_info, extracted_skills, total_keywords, total_structure_criteria)
                        experience_level = determine_level(resume_text, extracted_skills)

                    with col2:
                        # Visualization 1: Resume Score Gauge
                        display_score_gauge(resume_score)
                        st.subheader("Experience Level")
                        st.markdown(f"**{experience_level}**")
                        
                    st.markdown("---")
                    
                    # --- Role Selection and Analysis Header ---
                    role = st.selectbox("Select Role for Analysis", list(target_roles_required_skills.keys()))
                    st.header("Role Analysis") # Simplified header as requested
                    st.write(role_descriptions.get(role, f"No description available for **{role}**."))

                    required_skills = role_skills.get(role, [])
                    matched_skills, match_score, missing_skills = match_skills_for_role(extracted_skills, role)
                    
                    st.markdown("---")
                    
                    # --- Skills Overview and Donut Chart in Columns ---
                    col3, col4 = st.columns([1.5, 1])
                    
                    with col3:
                        st.subheader("Skills Overview")
                        st.markdown(f"**Extracted Skills** ({len(extracted_skills)}): {', '.join(extracted_skills)}")
                        st.markdown(f"**Matched Skills** ({len(matched_skills)}): {', '.join(matched_skills)}")
                        st.markdown(f"**Skill Match Percentage**: **{match_score:.2f}%**")
                        
                        st.subheader("Missing Skills (Recommended Focus)")
                        if missing_skills:
                             st.markdown(", ".join(missing_skills))
                        else:
                             st.success("You possess all target skills for this role!")


                    with col4:
                        # Visualization 2: Skill Match Donut Chart
                        display_skill_match_chart(match_score, len(missing_skills), len(matched_skills))


                    st.markdown("---")
                    
                    # --- Recommendations ---
                    rec_courses = course_recommender(extracted_skills, role)
                    st.markdown("---")
                    display_videos()

                    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

                    # ✅ Save data to Firestore
                    user_data = {
                        "Name": basic_info['name'],
                        "Email_ID": basic_info['email'],
                        "resume_score": resume_score,
                        "matching_score": f"{match_score:.2f}%",
                        "Timestamp": timestamp,
                        "Page_no": "N/A",
                        "Predicted_Field": role,
                        "User_level": experience_level,
                        "Actual_skills": extracted_skills,
                        "Recommended_skills": list(set(matched_skills)),
                        "Recommended_courses": rec_courses
                    }
                    try:
                        users_ref.add(user_data)
                        st.success("✅ Analysis complete and data saved successfully!")
                    except Exception as firebase_error:
                        st.error(f"Failed to save data to Firestore: {firebase_error}")

                else:
                    st.error("Unable to extract basic info from resume.")
    elif choice == 'Admin':
        # Pass the Firestore database client 'db' to the admin panel
        admin_panel(db)

   except Exception as e:
    st.error(f"An application error occurred: {e}")


if __name__ == "__main__":
   run()
