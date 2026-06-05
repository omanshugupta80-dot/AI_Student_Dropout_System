import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt
from database.database import get_connection

app = Flask(__name__)


model = joblib.load("model/trained_model.pkl")

#  =========================
#  Dashboard
# =========================
@app.route("/")
def dashboard():

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # Total Students
    cursor.execute("SELECT COUNT(*) AS total FROM students")
    total_students = cursor.fetchone()["total"]

    # High Risk
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
        WHERE risk_level='High Risk'
    """)
    high_risk = cursor.fetchone()["total"]

    # Medium Risk
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
        WHERE risk_level='Medium Risk'
    """)
    medium_risk = cursor.fetchone()["total"]

    # Low Risk
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
        WHERE risk_level='Low Risk'
    """)
    low_risk = cursor.fetchone()["total"]
    
    at_risk = high_risk + medium_risk

    if total_students > 0:
        retention_rate = round(
            ((total_students - at_risk) / total_students) * 100,
            2
        )
    else:
     retention_rate = 0

    # Interventions
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM interventions
    """)
    interventions = cursor.fetchone()["total"]

    # Top Risk Students
    cursor.execute("""
        SELECT
            p.*,
            s.student_name
        FROM predictions p
        JOIN students s
            ON p.student_id = s.student_id
        ORDER BY p.dropout_probability DESC
        LIMIT 10
    """)

    top_students = cursor.fetchall()
    
    labels = ["High Risk", "Medium Risk", "Low Risk"]
    values = [high_risk, medium_risk, low_risk]

    plt.figure(figsize=(5,5))
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    plt.title("Student Risk Distribution")

    plt.savefig("static/risk_chart.png")
    plt.close()

    cursor.close()
    conn.close()

    return render_template(
        "dashboard.html",
        total_students=total_students,
        high_risk=high_risk,
        medium_risk=medium_risk,
        low_risk=low_risk,
        interventions=interventions,
        top_students=top_students,
        retention_rate=retention_rate
    )
    
# ==========================
# ADD STUDENT
# ==========================

@app.route("/add_student", methods=["GET", "POST"])
def add_student():

    if request.method == "POST":

        student_id = request.form["student_id"]
        student_name = request.form["student_name"]
        gender = request.form["gender"]
        age = request.form["age"]

        tuition_fees = request.form["tuition_fees"]
        scholarship = request.form["scholarship"]

        admission_grade = request.form["admission_grade"]

        first_sem = request.form["first_sem"]
        second_sem = request.form["second_sem"]

        conn = get_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO students
        (
            student_id,
            student_name,
            gender,
            age,
            tuition_fees_up_to_date,
            scholarship_holder,
            admission_grade,
            curricular_units_1st_approved,
            curricular_units_2nd_approved
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        values = (
            student_id,
            student_name,
            gender,
            age,
            tuition_fees,
            scholarship,
            admission_grade,
            first_sem,
            second_sem
        )

        cursor.execute(query, values)

        conn.commit()

        cursor.close()
        conn.close()

        return redirect(url_for("students"))

    return render_template("add_student.html")


# ==========================
# VIEW STUDENTS
# ==========================

@app.route("/students")
def students():

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM students")

    students_data = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template(
        "students.html",
        students=students_data
    )

@app.route("/predict/<student_id>")
def predict_student(student_id):

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM students WHERE student_id=%s",
        (student_id,)
    )

    student = cursor.fetchone()

    if not student:
        cursor.close()
        conn.close()
        return "Student not found"

    # Gender Encoding
    gender = 1 if student["gender"] == "Male" else 0

    features = pd.DataFrame([[
        gender,
        student["age"],
        student["tuition_fees_up_to_date"],
        student["scholarship_holder"],
        student["admission_grade"],
        student["curricular_units_1st_approved"],
        student["curricular_units_2nd_approved"]
    ]], columns=[
        "Gender",
        "Age at enrollment",
        "Tuition fees up to date",
        "Scholarship holder",
        "Admission grade",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)"
    ])

    probability = model.predict_proba(features)[0][1] * 100
    probability = round(probability, 2)

    # Risk Factors
    risk_factors = []

    if student["tuition_fees_up_to_date"] == 0:
        risk_factors.append("Pending Tuition Fees")

    if student["curricular_units_1st_approved"] < 4:
        risk_factors.append("Low 1st Semester Performance")

    if student["curricular_units_2nd_approved"] < 4:
        risk_factors.append("Low 2nd Semester Performance")

    if student["scholarship_holder"] == 0:
        risk_factors.append("No Scholarship Support")

    if student["admission_grade"] < 7:
        risk_factors.append("Low Admission Grade")

    if not risk_factors:
        risk_factors.append("No Significant Risk Factors")

    risk_factors_text = ", ".join(risk_factors)

    # Risk Level + Recommendation
    if probability >= 70:
        risk_level = "High Risk"
        recommendation = "Immediate Counseling, Parent Meeting"

    elif probability >= 40:
        risk_level = "Medium Risk"
        recommendation = "Academic Mentoring, Attendance Monitoring"

    else:
        risk_level = "Low Risk"
        recommendation = "Routine Monitoring"

    # Save Prediction
    save_cursor = conn.cursor()

    save_cursor.execute("""
        INSERT INTO predictions
        (
            student_id,
            dropout_probability,
            risk_level,
            risk_factors,
            recommendation
        )
        VALUES (%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            dropout_probability = VALUES(dropout_probability),
            risk_level = VALUES(risk_level),
            risk_factors = VALUES(risk_factors),
            recommendation = VALUES(recommendation)
        """, (
            student_id,
            probability,
            risk_level,
            risk_factors_text,
            recommendation
        ))

    conn.commit()

    save_cursor.close()
    cursor.close()
    conn.close()

    return redirect(url_for("predictions"))

@app.route("/predictions")
def predictions():

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            p.*,
            s.student_name
        FROM predictions p
        JOIN students s
            ON p.student_id = s.student_id
        ORDER BY p.dropout_probability DESC
    """)

    predictions_data = cursor.fetchall()

    cursor.close()
    conn.close()
    

    return render_template(
        "predictions.html",
        predictions=predictions_data
    )
    
    
# add inventory 
@app.route("/add_intervention/<student_id>", methods=["GET", "POST"])
def add_intervention(student_id):

    if request.method == "POST":

        intervention_type = request.form["intervention_type"]
        counselor_name = request.form["counselor_name"]
        remarks = request.form["remarks"]
        outcome = request.form["outcome"]

        conn = get_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO interventions
        (
            student_id,
            intervention_type,
            counselor_name,
            remarks,
            outcome
        )
        VALUES (%s,%s,%s,%s,%s)
        """

        values = (
            student_id,
            intervention_type,
            counselor_name,
            remarks,
            outcome
        )

        cursor.execute(query, values)

        conn.commit()

        cursor.close()
        conn.close()

        return redirect(url_for("interventions"))

    return render_template(
        "add_intervention.html",
        student_id=student_id
    )
    
# inventory 

@app.route("/interventions")
def interventions():

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            i.*,
            s.student_name
        FROM interventions i
        JOIN students s
            ON i.student_id = s.student_id
        ORDER BY intervention_date DESC
    """)

    interventions_data = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template(
        "interventions.html",
        interventions=interventions_data
    )

if __name__ == "__main__":
    app.run(debug=True)