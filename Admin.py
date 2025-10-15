import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from target_roles import target_roles_required_skills

# Function to create a download link for a DataFrame
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Admin panel function
def admin_panel(cursor):
    st.subheader("Admin Panel")

    # Role selection for filtering resumes
    role_filter = st.selectbox("Select Role to Filter Resumes", list(target_roles_required_skills.keys()))

    # View filtered user data by role
    if role_filter:
        query = "SELECT * FROM user_data WHERE Predicted_Field IS NOT NULL AND Predicted_Field != ''"
        query += " AND LOWER(Predicted_Field) = LOWER(%s)"
        query += " ORDER BY resume_score DESC"

        try:
            cursor.execute(query, (role_filter,))
            user_data = cursor.fetchall()
            if user_data:
                # Create DataFrame
                df = pd.DataFrame(user_data, columns=[
                    "ID", "Name", "Email_ID", "Resume Score", "Timestamp", "Page No",
                    "Predicted Field", "User Level", "Actual Skills",
                    "Recommended Skills", "Recommended Courses", "Extra Column"
                ])

                # Drop unnecessary columns
                df_filtered = df.drop(columns=["Recommended Skills", "Recommended Courses", "Extra Column", "Page No"])

                # Display the filtered DataFrame
                st.dataframe(df_filtered)

                # Provide a download link for filtered data
                download_link = get_table_download_link(df_filtered, f"{role_filter}_filtered_data.csv",
                                                        "Download Filtered Data as CSV")
                st.markdown(download_link, unsafe_allow_html=True)
            else:
                st.warning(f"No resumes found for the selected role: {role_filter}.")
        except Exception as e:
            st.error(f"Error executing query: {e}")

    # View resume score distribution
    view_scores = st.checkbox("View Resume Score Distribution")
    if view_scores:
        cursor.execute("SELECT resume_score FROM user_data")
        scores = [int(row[0]) for row in cursor.fetchall() if row[0].isdigit()]
        if scores:
            fig = px.histogram(scores, nbins=10, title="Resume Score Distribution")
            st.plotly_chart(fig)

    # View user activity over time
    view_activity = st.checkbox("View User Activity Over Time")
    if view_activity:
        try:
            # Fetch timestamps from the database
            cursor.execute("SELECT Timestamp FROM user_data")
            timestamps = [row[0] for row in cursor.fetchall()]

            # Replace '_' with ' ' to ensure proper datetime format
            processed_timestamps = [ts.replace("_", " ") for ts in timestamps]

            # Convert processed timestamps to pandas datetime
            timestamps_dt = [pd.to_datetime(ts) for ts in processed_timestamps]

            if timestamps_dt:
                # Create a DataFrame for user activity
                activity_df = pd.DataFrame(timestamps_dt, columns=["Timestamp"])
                activity_df["Date"] = activity_df["Timestamp"].dt.date

                # Aggregate activity by date
                activity_counts = activity_df["Date"].value_counts().sort_index()

                # Plot user activity over time
                fig = px.line(x=activity_counts.index, y=activity_counts.values,
                              title="User Activity Over Time", labels={"x": "Date", "y": "Activity Count"})
                st.plotly_chart(fig)
            else:
                st.warning("No activity data found.")
        except Exception as e:
            st.error(f"Error processing user activity data: {e}")
