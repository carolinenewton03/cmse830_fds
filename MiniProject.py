import random
import yt_dlp as ytdlp
import streamlit as st
import pdfplumber
import pymysql
import pandas as pd
import base64
import re
import spacy
from spacy.matcher import PhraseMatcher
from streamlit_tags import st_tags
import plotly.graph_objects as go # Import Plotly Graph Objects for the Gauge
import plotly.express as px        # Import Plotly Express for the Pie/Donut
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

# ----------------------------------------------------------------------
# FIREBASE/SPACY SETUP (Remains unchanged)
# ----------------------------------------------------------------------
if not firebase_admin._apps:
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
db = firestore.client()

model_name = "en_core_web_sm"
try:
    nlp = spacy.load(model_name)
except OSError:
    st.warning(f"{model_name} not found. Please ensure it’s preinstalled.")
    nlp = None
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# ----------------------------------------------------------------------
# UTILITY FUNCTIONS (Remains largely unchanged)
# ----------------------------------------------------------------------

def show_pdf(file):
    try:
        file.seek(0)
    except Exception:
        pass
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
    try:
        file.seek(0)
    except Exception:
        pass

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
    if not text:
        try:
            file.seek(0)
            raw = file.read()
            text = raw.decode('utf-8', errors='ignore')
        except Exception:
            pass
    return text


def extract_basic_info(text):
    import re
    lines = text.split("\n")
    name = "Not Found"

    for line in lines:
        clean_line = ' '.join(line.strip().split())
        if (clean_line and re.match(r"^[A-Za-z\s\.]+$", clean_line)
                and not any(word in clean_line.lower() for word in ["contact", "education", "profile", "objective", "experience", "skills", "summary", "work", "certifications", "projects"])):
            words = clean_line.split()
            if sum(1 for w in words if len(w) == 1) / len(words) > 0.5 and len(words) > 1:
                name = ''.join(words).title()
            elif len(clean_line) > 2:
                name = clean_line.title()
            break

    phone_match = re.search(
        r'(\+?\d{1,3}[\s\-\.]?\(?\d{1,4}\)?[\s\-\.]?\d{3,4}[\s\-\.]?\d{3,4})|(\d{10})',
        text
    )
    mobile = next((g for g in phone_match.groups() if g), "Not Found") if phone_match else "Not Found"

    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    email = email_match.group() if email_match else "Not Found"

    return {
        "name": name,
        "email": email,
        "mobile_number": mobile
    }


def calculate_resume_score(basic_info, extracted_skills, total_keywords, total_structure_criteria):
  score = 0
  if basic_info['name'] != "Not Found":
      score += 10
  if basic_info['email'] != "Not Found":
      score += 5
  score += len(extracted_skills) * 2
  score += (total_keywords / 2)
  score += total_structure_criteria * 5
  max_score = 100
  normalized_score = min(score, max_score)
  return normalized_score

def fetch_yt_thumbnail(link):
    # This function remains the same, assuming it's working for video display
    try:
        if "youtube.com/watch?v=" in link:
            video_id = link.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in link:
            video_id = link.split("youtu.be/")[-1].split("?")[0]
        else:
            raise ValueError("Invalid YouTube link format.")

        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
        return thumbnail_url, link
    except Exception:
        return None, None

def course_recommender(extracted_skills, role):
  st.subheader("Courses & Certificates🎓 Recommendations")
  rec_course = []
  required_skills = role_skills.get(role, [])
  missing_skills = [skill for skill in required_skills if skill not in extracted_skills]
  course_set = set()

  for skill in missing_skills:
      if skill in ['Data Analysis', 'Machine Learning', 'Deep Learning']:
          course_set.update(tuple(course) for course in ds_course)
      if skill in ['Web Development', 'JavaScript', 'HTML', 'CSS']:
          course_set.update(tuple(course) for course in web_course)
      if skill in ['Android Development', 'Java']:
          course_set.update(tuple(course) for course in android_course)
      if skill in ['iOS Development', 'Swift']:
          course_set.update(tuple(course) for course in ios_course)
      if skill in ['UI/UX', 'Design']:
          course_set.update(tuple(course) for course in uiux_course)
      for key, courses in software_engineering_courses.items():
          if skill in key:
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
    skills_keywords = ["skills", "technical skills", "certifications"]
    lines = text.split('\n')
    filtered_lines = []
    capture = False

    for line in lines:
        if any(k in line.lower() for k in skills_keywords):
            capture = True
        elif capture and (line.strip() == "" or len(line.strip()) < 3):
            capture = False
        if capture:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def extract_skills(resume_text, skills_list):
    text = unicodedata.normalize("NFKD", resume_text).encode("ascii", "ignore").decode("utf-8").lower()
    text = re.sub(r"[^a-z0-9\s\+]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text_nospace = text.replace(" ", "")

    extracted = set()
    for skill in skills_list:
        s_low = skill.lower().strip()
        s_nospace = s_low.replace(" ", "")
        if re.search(r"\b" + re.escape(s_low) + r"\b", text) or s_nospace in text_nospace:
            extracted.add(skill)

    return sorted(list(extracted))

def determine_level(text, skills):
    import re
    from datetime import datetime

    text = text.lower()
    years = 0

    patterns = [r"(\d+)\s+years?\s+of\s+experience", r"experience\s+of\s+(\d+)\s+years?", r"(\d+)\s+years?\s+experience", r"(\d+)\s+yrs", r"(\d+)\+?\s+years"]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            years = int(match.group(1))
            break

    if years >= 5 or len(skills) > 10:
        return "Advanced"
    elif 2 <= years < 5 or 5 <= len(skills) <= 10:
        return "Intermediate"
    else:
        return "Fresher"


def match_skills_for_role(extracted_skills, role):
    required_skills = role_skills.get(role, [])
    required_skills_normalized = [skill.lower() for skill in required_skills]
    extracted_skills_normalized = [skill.lower() for skill in extracted_skills]

    matched_skills = [skill for skill in extracted_skills_normalized if skill in required_skills_normalized]
    missing_skills = [skill for skill in required_skills_normalized if skill not in extracted_skills_normalized]

    matched_skills_original = [skill.capitalize() for skill in matched_skills]
    missing_skills_original = [skill.capitalize() for skill in missing_skills]

    match_score = (len(matched_skills) / len(required_skills_normalized)) * 100 if required_skills_normalized else 0

    return matched_skills_original, match_score, missing_skills_original


def display_videos():
  st.subheader("Resume Building Tips 📋")
  resume_columns = st.columns(2)
  for idx, link in enumerate(resume_videos):
      thumbnail_url, video_url = fetch_yt_thumbnail(link)
      if thumbnail_url:
          with resume_columns[idx % 2]:
              st.markdown(
                  f'<a href="{video_url}" target="_blank"><img src="{thumbnail_url}" width="400"></a>',
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
                  f'<a href="{video_url}" target="_blank"><img src="{thumbnail_url}" width="400"></a>',
                  unsafe_allow_html=True
              )
      else:
          st.warning(f"Thumbnail not found for link: {link}")


def is_resume(text):
  resume_keywords = ["experience", "education", "skills", "certifications", "projects", "summary", "contact"]
  return any(keyword in text.lower() for keyword in resume_keywords)


# ----------------------------------------------------------------------
# NEW VISUALIZATION FUNCTIONS FOR NORMAL USER
# ----------------------------------------------------------------------

def display_score_gauge(score):
    """Displays a Plotly gauge chart for the Resume Score."""
    st.subheader("Resume Score")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Resume Quality Score", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#2A9D8F"}, # Teal color for a professional look
                'steps': [
                    {'range': [0, 50], 'color': "#E9C46A"}, # Yellow/orange for low
                    {'range': [50, 80], 'color': "#F4A261"}, # Medium orange
                    {'range': [80, 100], 'color': "#264653"} # Dark color for high
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        )
    )
    fig.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)


def display_skill_match_chart(matched_skills, missing_skills, role):
    """Displays a Plotly pie/donut chart for Skill Match breakdown."""

    total_required = len(matched_skills) + len(missing_skills)

    if total_required == 0:
        st.warning(f"No required skills defined for **{role}**. Cannot generate skill match chart.")
        return

    # Data for the chart
    data = {
        'Category': ['Matched Skills', 'Missing Skills'],
        'Count': [len(matched_skills), len(missing_skills)]
    }
    df = pd.DataFrame(data)

    fig = px.pie(
        df,
        values='Count',
        names='Category',
        title=f'Skill Match Breakdown for {role}',
        hole=0.5, # Makes it a donut chart
        color_discrete_map={'Matched Skills': '#2A9D8F', 'Missing Skills': '#E76F51'} # Professional color scheme
    )

    # Customize layout for better appearance in Streamlit
    fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
    fig.update_layout(legend_title_text="Skill Status")

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------

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
            pdf_file.seek(0)
            resume_text = pdf_reader(pdf_file)

            if not is_resume(resume_text):
                st.error("Uploaded file does not appear to be a resume. Please upload a valid resume.")
            else:
                st.header("Resume Analysis")
                st.success("Resume successfully read!")

                basic_info = extract_basic_info(resume_text)

                if basic_info:

                    # Dummy skills list for initial extraction
                    skills_list = [
                        'Python', 'Java', 'SQL', 'Excel', 'Power BI', 'Git', 'HTML', 'CSS', 'JavaScript',
                        'React.js', 'OOP', 'APIs', 'Unit Testing', 'Version Control', 'Agile', 'CI/CD',
                        'Data Structures', 'Algorithms', 'Communication', 'CRM', 'Problem Solving'
                    ]
                    relevant_text = extract_relevant_sections(resume_text)
                    extracted_skills = extract_skills(relevant_text if relevant_text else resume_text, skills_list)

                    # --- RENDER ROLE SELECTION & ANALYSIS ---
                    role = st.selectbox("Select Target Role for Analysis", list(target_roles_required_skills.keys()))
                    st.write(role_descriptions.get(role, f"No description available for **{role}**."))

                    required_skills = role_skills.get(role, [])
                    matched_skills, match_score, missing_skills = match_skills_for_role(extracted_skills, role)
                    
                    # CORRECTION APPLIED HERE: Pass extracted_skills list correctly as the second positional argument
                    experience_level = determine_level(resume_text, extracted_skills) 

                    total_keywords = 20
                    total_structure_criteria = 3
                    resume_score = calculate_resume_score(basic_info, extracted_skills, total_keywords, total_structure_criteria)

                    # --- RENDER VISUALIZATIONS ---
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Basic Profile Overview")
                        st.write(f"**Name**: {basic_info['name']}")
                        st.write(f"**Email**: {basic_info['email']}")
                        st.write(f"**Mobile**: {basic_info['mobile_number']}")
                        st.write(f"**Experience Level**: {experience_level}")

                    with col2:
                        # Display the new Gauge Chart
                        display_score_gauge(resume_score)


                    st.subheader("Skill Alignment")
                    # Display the new Skill Match Chart
                    display_skill_match_chart(matched_skills, missing_skills, role)


                    st.markdown("#### Skill Details")
                    st.write(f"**Skill Match Percentage**: **{match_score:.2f}%**")
                    st.write("Extracted Skills:", ", ".join(extracted_skills) if extracted_skills else "No skills found.")
                    st.write("Matched Skills:", ", ".join(matched_skills))
                    st.markdown("**:red[Missing Skills (Recommended Focus)]**:", unsafe_allow_html=True)
                    st.write(", ".join(missing_skills))


                    rec_courses = course_recommender(extracted_skills, role)
                    display_videos()

                    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

                    # ✅ Save data to Firestore
                    user_data = {
                        "Name": basic_info['name'],
                        "Email_ID": basic_info['email'],
                        "resume_score": resume_score,
                        "matching_score": match_score,
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
                        st.success("Analysis complete and data saved successfully!")
                    except Exception as firebase_error:
                        st.error(f"Failed to save data to Firestore: {firebase_error}")

                else:
                    st.error("Unable to extract basic info from resume.")
    elif choice == 'Admin':
        st.warning("The Admin Panel currently uses an incompatible database connection (MySQL cursor). It needs to be updated to read data from Firestore (`db` object) to function.")
        # admin_panel(db) # Uncomment this if you update Admin.py to work with Firestore

   except Exception as e:
    st.error(f"An application error occurred: {e}")


if __name__ == "__main__":
   run()
