# db_connection.py
import pymysql

def connect_to_db():
    # Establish connection to the MySQL database
    connection = pymysql.connect(
        host='localhost',      # Database host
        user='root',           # Database username
        password='your_password',  # Database password
        db='sra'               # Name of the database
    )
    return connection

def create_table():
    connection = connect_to_db()
    cursor = connection.cursor()

    DB_table_name = 'user_data'
    table_sql = """
    CREATE TABLE IF NOT EXISTS user_data (
        ID INT NOT NULL AUTO_INCREMENT,
        Name VARCHAR(100) NOT NULL,
        Email_ID VARCHAR(50) NOT NULL,
        resume_score VARCHAR(8) NOT NULL,
        matching_score VARCHAR(8) NOT NULL,
        Timestamp VARCHAR(50) NOT NULL,
        Page_no VARCHAR(5) NOT NULL,
        Predicted_Field VARCHAR(25) NOT NULL,
        User_level VARCHAR(30) NOT NULL,
        Actual_skills VARCHAR(300) NOT NULL,
        Recommended_skills VARCHAR(300) NOT NULL,
        Recommended_courses VARCHAR(600) NOT NULL,
        PRIMARY KEY (ID)
    );
"""
    cursor.execute(table_sql)
    connection.commit()
    connection.close()
