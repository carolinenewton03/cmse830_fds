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
# ✅ Load the English spaCy model safely
model_name = "en_core_web_sm"
try:
    nlp = spacy.load(model_name)
except OSError:
    st.warning(f"{model_name} not found. Please ensure it’s preinstalled.")
    nlp = None

# Function to read and display PDF safely
def show_pdf(file):
    # Read file and encode to base64
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embed PDF using iframe instead of <embed>
    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}" 
            width="700" height="1000" type="application/pdf"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

# Extract text from PDF using pdfplumber
def pdf_reader(file):
   with pdfplumber.open(file) as pdf:
       text = ""
       for page in pdf.pages:
           text += page.extract_text() + '\n'
   return text

def extract_basic_info(text):
    import re

    lines = text.split("\n")
    name = "Not Found"

    for line in lines:
        clean_line = ' '.join(line.strip().split())  # normalize multiple spaces to one
        if (clean_line and re.match(r"^[A-Za-z\s\.]+$", clean_line)
                and not any(word in clean_line.lower() for word in ["contact", "education", "profile", "objective", "experience", "skills", "summary", "work", "certifications", "projects"])):
            # remove spaces *only* if line looks like S T E F ... (lots of single letters)
            letters_only = clean_line.replace(" ", "")
            if len(letters_only) >= 2 and letters_only.isalpha():
                # check if >70% of words are single letters → join them
                words = clean_line.split()
                if sum(1 for w in words if len(w) == 1) / len(words) > 0.5:
                    name = ''.join(words).title()
                else:
                    name = clean_line.title()
            else:
                name = clean_line.title()
            break

    # phone detection (better)
    phone_match = re.search(
        r'(\+?\d{1,3}[\s\-\.]?\(?\d{1,4}\)?[\s\-\.]?\d{3,4}[\s\-\.]?\d{3,4})|(\d{10})',
        text
    )
    if phone_match:
        mobile = next((g for g in phone_match.groups() if g), "Not Found")
    else:
        mobile = "Not Found"

    # email
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    email = email_match.group() if email_match else "Not Found"

    return {
        "name": name,
        "email": email,
        "mobile_number": mobile
    }




# Function to calculate resume score
def calculate_resume_score(basic_info, extracted_skills, total_keywords, total_structure_criteria):
  score = 0
  if basic_info['name'] != "N/A":
      score += 10
  if basic_info['email'] != "N/A":
      score += 5
  score += len(extracted_skills) * 2
  score += (total_keywords / 2)
  score += total_structure_criteria * 5
  max_score = 100
  normalized_score = min(score, max_score)
  return normalized_score

# Function to fetch the thumbnail URL and video URL from the YouTube link
def fetch_yt_thumbnail(link):
  try:
      # Check if the link is a valid YouTube link
      if "youtube.com/watch?v=" not in link and "youtu.be/" not in link:
          raise ValueError("Invalid YouTube link format.")

      # Extract video ID from the link
      if "youtube.com/watch?v=" in link:
          video_id = link.split("v=")[-1].split("&")[0]  # Split on 'v=' and take the first part
      elif "youtu.be/" in link:
          video_id = link.split("youtu.be/")[-1].split("?")[0]  # Handle the short link format

      thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"  # Construct thumbnail URL

      # Debug output
      print(f"Fetching thumbnail for video ID: {video_id} - Thumbnail URL: {thumbnail_url}")  # Debug print
      return thumbnail_url, link  # Return thumbnail URL and video link
  except Exception as e:
      print(f"Error fetching thumbnail for link: {link} - {e}")
      return None, None

# Recommending Courses based on Skills
# Recommending Courses based on Missing Skills
def course_recommender(extracted_skills, role):
  st.subheader("Courses & Certificates🎓 Recommendations")
  rec_course = []

  # Get required skills for the specified role
  required_skills = role_skills.get(role, [])

  # Identify missing skills
  missing_skills = [skill for skill in required_skills if skill not in extracted_skills]

  # Define a set to hold unique course recommendations
  course_set = set()

  # Append courses based on missing skills, irrespective of role
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

      # Check for software engineering relevant skills
      for key, courses in software_engineering_courses.items():
          if skill in key:
              course_set.update(tuple(course) for course in courses)

  # Convert set back to a list to randomize
  course_list = list(course_set)

  # Limit to a maximum of 10 unique courses
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
    # Convert all skills and resume text to lowercase for uniformity
    resume_text = resume_text.lower()
    skills_list = [skill.lower() for skill in skills_list]

    # Load Spacy model and initialize PhraseMatcher
    nlp = spacy.load("en_core_web_sm")
    matcher = PhraseMatcher(nlp.vocab)

    # Generate patterns for exact and alias matches
    patterns = [nlp(skill) for skill in skills_list]
    matcher.add("Skills", patterns)

    # Remove special characters and clean text
    resume_text = re.sub(r'[^\w\s+]', ' ', resume_text)  # Clean punctuation
    doc = nlp(resume_text)

    # Find matches
    matches = matcher(doc)
    extracted_skills = list(set([doc[start:end].text.strip() for match_id, start, end in matches]))

    # Debugging output to verify extracted skills
    print(f"Extracted skills: {extracted_skills}")
    return extracted_skills


# Function to determine experience level (Fresher, Intermediate, Advanced)
def determine_level(text, skills):
    import re
    from datetime import datetime

    text = text.lower()
    years = 0

    # find explicit phrases first
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

    # fallback: only scan after "experience"
    exp_part = text
    exp_index = text.find('experience')
    if exp_index != -1:
        exp_part = text[exp_index:]

    found_years = list({int(y) for y in re.findall(r'20\d{2}', exp_part)})
    current_year = datetime.now().year

    # try to get graduation year (e.g., latest year in education section)
    grad_years = [int(y) for y in re.findall(r'20\d{2}', text)]
    graduation_year = max(grad_years) if grad_years else current_year

    # remove years ≥ graduation year (likely academic projects or internships)
    work_years = [y for y in found_years if y < graduation_year]

    if len(work_years) >= 2:
        min_year = min(work_years)
        max_year = max(work_years)
        year_span = max_year - min_year
        if year_span >= 2:
            years = year_span

    # final logic
    if years >= 5 or len(skills) > 10:
        return "Advanced"
    elif 2 <= years < 5 or 5 <= len(skills) <= 10:
        return "Intermediate"
    else:
        return "Fresher"



# Function to calculate skill match for a specified role
def match_skills_for_role(extracted_skills, role):
    required_skills = role_skills.get(role, [])

    # Normalize both required skills and extracted skills to ensure matching
    required_skills_normalized = [skill.lower() for skill in required_skills]
    extracted_skills_normalized = [skill.lower() for skill in extracted_skills]

    # Identify matched and missing skills
    matched_skills = [skill for skill in extracted_skills_normalized if skill in required_skills_normalized]
    missing_skills = [skill for skill in required_skills_normalized if skill not in extracted_skills_normalized]

    # Preserve original casing for output
    matched_skills_original = [skill.capitalize() for skill in matched_skills]
    missing_skills_original = [skill.capitalize() for skill in missing_skills]

    # Calculate skill match score
    match_score = (len(matched_skills) / len(required_skills_normalized)) * 100 if required_skills_normalized else 0

    return matched_skills_original, match_score, missing_skills_original


# Display resume tips and interview videos with thumbnails in a two-column layout
def display_videos():
  st.subheader("Resume Building Tips 📋")
  resume_columns = st.columns(2)  # Create two columns for resume tips
  for idx, link in enumerate(resume_videos):
      thumbnail_url, video_url = fetch_yt_thumbnail(link)
      if thumbnail_url:  # Ensure the thumbnail URL is valid
          with resume_columns[idx % 2]:  # Place each video in alternating columns
              st.markdown(
                  f'<a href="{video_url}" target="_blank"><img src="{thumbnail_url}" width="400"></a>',  # Increased width to 400
                  unsafe_allow_html=True
              )
      else:
          st.warning(f"Thumbnail not found for link: {link}")




  st.subheader("Interview Preparation 🎥")
  interview_columns = st.columns(2)  # Create two columns for interview videos
  for idx, link in enumerate(interview_videos):
      thumbnail_url, video_url = fetch_yt_thumbnail(link)
      if thumbnail_url:  # Ensure the thumbnail URL is valid
          with interview_columns[idx % 2]:  # Place each video in alternating columns
              st.markdown(
                  f'<a href="{video_url}" target="_blank"><img src="{thumbnail_url}" width="400"></a>',  # Increased width to 400
                  unsafe_allow_html=True
              )
      else:
          st.warning(f"Thumbnail not found for link: {link}")




def is_resume(text):
  # Simple heuristic: checking for common sections in resumes
  resume_keywords = ["experience", "education", "skills", "certifications", "projects", "summary", "contact"]
  return any(keyword in text.lower() for keyword in resume_keywords)


# Main Streamlit Application
def run():
   st.title("Smart Resume Analyser")
   st.sidebar.markdown("# Choose User")
   activities = ["Normal User", "Admin"]
   choice = st.sidebar.selectbox("Choose among the given options:", activities)


   # MySQL Database Connection
   #connection = pymysql.connect(host='localhost', user='root', password='Carokutty12', database="sra")


   try:
    # ✅ Firestore Setup
    users_ref = db.collection("user_data")

    if choice == 'Normal User':
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        if pdf_file:
            show_pdf(pdf_file)
            pdf_file.seek(0)
            resume_text = pdf_reader(pdf_file)

            if not is_resume(resume_text):
                st.error("Uploaded file does not appear to be a resume. Please upload a valid resume.")
            else:
                st.header("Resume Analysis")
                st.success("Resume successfully read!")
                st.text_area("Resume Text", value=resume_text, height=300)

                basic_info = extract_basic_info(resume_text)

                if basic_info:
                    st.subheader("Basic Info")
                    st.write(f"*Name*: {basic_info['name']}")
                    st.write(f"*Email*: {basic_info['email']}")
                    st.write(f"*Mobile Number*: {basic_info['mobile_number']}")

                    if basic_info['name'] == "N/A":
                        st.warning("Name could not be extracted from the resume.")
                    if basic_info['mobile_number'] == "N/A":
                        st.warning("Mobile number could not be extracted from the resume.")

                    skills_list = [
                        # (your full skill list here)
                    ]

                    relevant_text = extract_relevant_sections(resume_text)
                    extracted_skills = extract_skills(relevant_text if relevant_text else resume_text, skills_list)
                    role = st.selectbox("Select Role for Analysis", list(target_roles_required_skills.keys()))

                    st.write(role_descriptions.get(role, f"No description available for **{role}**."))

                    required_skills = role_skills.get(role, [])
                    matched_skills, match_score, missing_skills = match_skills_for_role(extracted_skills, role)

                    st.subheader("Skills Overview")
                    st.write("Extracted Skills:", ", ".join(extracted_skills) if extracted_skills else "No skills found.")
                    st.write("Matched Skills:", ", ".join(matched_skills))
                    st.write("Missing Skills:", ", ".join(missing_skills))
                    st.write(f"Skill Match Score: {match_score:.2f}%")

                    total_keywords = 20
                    total_structure_criteria = 3
                    resume_score = calculate_resume_score(basic_info, extracted_skills, total_keywords, total_structure_criteria)
                    st.subheader("Resume Score")
                    st.write(f"**{resume_score}**")

                    experience_level = determine_level(resume_text, extracted_skills)
                    st.subheader("Experience Level")
                    st.write(f"Based on the analysis, you are categorized as: **{experience_level}**")

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

                    users_ref.add(user_data)
                    #st.success("✅ Data successfully saved to Firebase Firestore!")
                else:
                    st.error("Unable to extract basic info from resume.")
    elif choice == 'Admin':
        admin_panel(db)

   except Exception as e:
    st.error(f"An error occurred: {e}")


if __name__ == "__main__":
   run()



