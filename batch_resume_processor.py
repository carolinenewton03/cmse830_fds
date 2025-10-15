import os
import pdfplumber
import pandas as pd
import base64
import re
import spacy
from datetime import datetime
from target_roles import role_skills
from MiniProject import (
    extract_basic_info,
    extract_skills,
    extract_relevant_sections,
    determine_level,
    calculate_resume_score,
    is_resume
)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Your master skills list
skills_list = [
    'Python', 'Java', 'SQL', 'Excel', 'Power BI', 'Git', 'HTML', 'CSS', 'JavaScript',
    'React.js', 'OOP', 'APIs', 'Unit Testing', 'Version Control', 'Agile', 'CI/CD',
    'Data Structures', 'Algorithms', 'Communication', 'CRM', 'Problem Solving',
    'Multitasking', 'Salesforce', 'HIPAA', 'ICD-10', 'Medical Terminology',
    'Auditing', 'Anatomy', 'Clinical Documentation', 'Pathophysiology'
]

# Reads text from PDF
def pdf_reader(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text

# Role-matching function
def auto_match_best_role(extracted_skills):
    role_scores = {}
    for role, skills in role_skills.items():
        matched_skills = [s.lower() for s in extracted_skills if s.lower() in [r.lower() for r in skills]]
        score = len(matched_skills) / len(skills) if skills else 0
        role_scores[role] = score
    best_role = max(role_scores, key=role_scores.get)
    return best_role, role_scores[best_role] * 100

# Batch processor
def process_folder(folder_path):
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            try:
                resume_text = pdf_reader(file_path)
                if not is_resume(resume_text):
                    continue

                basic_info = extract_basic_info(resume_text)
                relevant_text = extract_relevant_sections(resume_text)
                extracted_skills = extract_skills(relevant_text if relevant_text else resume_text, skills_list)
                experience_level = determine_level(resume_text, extracted_skills)

                # Initialize best score and role
                best_score = 0
                best_role = None

                # Match best role
                best_role, best_score = auto_match_best_role(extracted_skills)

                total_keywords = 20
                total_structure_criteria = 3
                resume_score = calculate_resume_score(basic_info, extracted_skills, total_keywords, total_structure_criteria)

                results.append({
                    "File Name": file_name,
                    "Name": basic_info['name'],
                    "Email": basic_info['email'],
                    "Mobile": str(basic_info['mobile_number']),  # Always save as string
                    "Best Matched Role": best_role,
                    "Experience Level": experience_level,
                    "Resume Score": resume_score,
                    "Skill Match %": best_score,
                    "Extracted Skills": ", ".join(extracted_skills),
                    "Processed Time": datetime.now().strftime("%d-%m-%Y %H:%M")
                })

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    return pd.DataFrame(results)

# Main block
if __name__ == "__main__":
    resume_folder = "resumes"  # change path if needed
    print(f"\nProcessing resumes in folder: {resume_folder}\n")

    if not os.path.exists(resume_folder):
        print("❌ Folder not found.")
    else:
        df = process_folder(resume_folder)
        if not df.empty:
            output_path = "batch_resume_results.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Processed {len(df)} resumes. Results saved to {output_path}\n")
            print(df.head())
        else:
            print("⚠️ No valid resumes found or all resumes were skipped.\n")
