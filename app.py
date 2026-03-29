import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sqlite3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import plotly.graph_objects as go
import io

# ---------------- SESSION STATE ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None   # ← Fixed: no premature variables here

if "form_version" not in st.session_state:
    st.session_state.form_version = 0

# NEW: used to control which record is being edited
if "editing_patient_id" not in st.session_state:
    st.session_state.editing_patient_id = None

# NEW: used to control delete confirmation visibility
if "delete_confirm_id" not in st.session_state:
    st.session_state.delete_confirm_id = None


# ---------------- LOGIN ----------------

def login():
    st.sidebar.title("🩺 Login Portal")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.session_state.current_user = "admin"
            st.session_state.prediction_result = None
            st.rerun()
        elif username == "doctor" and password == "1212":
            st.session_state.logged_in = True
            st.session_state.current_user = "doctor"
            st.session_state.prediction_result = None
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials")


# ---------------- DATABASE ----------------

def init_db():
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        height REAL,
        weight REAL,
        bmi REAL,
        ap_hi INTEGER,
        ap_lo INTEGER,
        cholesterol TEXT,
        gluc TEXT,
        smoke TEXT,
        alco TEXT,
        active TEXT,
        prediction TEXT,
        risk REAL,
        created_by TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()


# ---------------- LOGIN CHECK ----------------

if not st.session_state.logged_in:
    st.set_page_config(page_title="Doctor Login")
    st.title("🏥 Cardiovascular AI Prediction System")
    login()
    st.stop()


# ---------------- MAIN APP ----------------

st.set_page_config(page_title="Heart Prediction", layout="wide")

if st.session_state.current_user == "admin":
    st.sidebar.markdown("**Welcome, System Administrator** 👨‍💻")
else:
    st.sidebar.markdown("**Welcome, Doctor** 🩺")

st.sidebar.success("Logged In")

# ⭐ NEW FEATURE: Model Accuracy Display
st.sidebar.markdown("---")
st.sidebar.info("🤖 Model Accuracy: 87%")
st.sidebar.caption("Algorithm: Random Forest Classifier")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.prediction_result = None
    st.session_state.editing_patient_id = None
    st.session_state.delete_confirm_id = None
    st.rerun()


# ---------------- LOAD MODEL ----------------

model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏥 Cardiovascular Disease Prediction System")

# ⭐ NEW FEATURE: Added About System Tab
tab1, tab2, tab3, tab4 = st.tabs([
    "ℹ️ About System",
    "🩺 New Prediction",
    "📊 Visual Analysis",
    "📋 Patient History"
])


# ======================================================
# TAB 1 : ABOUT SYSTEM
# ======================================================

with tab1:

    st.subheader("AI Cardiovascular Risk Prediction System")

    st.write("""
This system predicts the **risk of cardiovascular disease** using a Machine Learning model.

The application helps doctors evaluate patient health conditions using medical attributes
such as blood pressure, cholesterol, glucose level, lifestyle habits, and BMI.

### Technologies Used

- Python
- Streamlit
- Scikit-Learn
- SQLite Database
- Plotly Visualizations
- ReportLab PDF Generation

### Machine Learning Model

The system uses a **Random Forest Classifier** trained on cardiovascular disease dataset.

The model analyses multiple risk factors including:

- Age
- BMI
- Blood Pressure
- Cholesterol
- Glucose
- Smoking
- Alcohol Consumption
- Physical Activity

### Purpose

The goal of this system is to support **early detection of cardiovascular disease risk**
and help doctors make **data-driven healthcare decisions**.

⚠️ This tool is for **educational and research purposes only**.
    """)


# ======================================================
# TAB 2 : PREDICTION
# ======================================================

with tab2:
    st.subheader("Patient Information")

    key_suffix = f"_{st.session_state.form_version}"

    with st.form("prediction_form"):
        # Use container + columns for perfect vertical alignment
        container = st.container()
        col1, col2 = container.columns(2)

        with col1:
            st.markdown("**Patient Name** *(required)*")
            name = st.text_input("", value="", key=f"name{key_suffix}", label_visibility="collapsed")
            age = st.number_input("Age", min_value=1, max_value=120, value=30, key=f"age{key_suffix}")
            gender = st.selectbox("Gender", ["Female", "Male"], index=0, key=f"gender{key_suffix}")
            height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, key=f"height{key_suffix}")
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, key=f"weight{key_suffix}")
            smoke = st.selectbox("Smoking", ["No", "Yes"], index=0, key=f"smoke{key_suffix}")

        with col2:
            st.markdown("")
            ap_hi = st.number_input("Systolic BP", min_value=50, max_value=300, value=120, key=f"ap_hi{key_suffix}")
            ap_lo = st.number_input("Diastolic BP", min_value=30, max_value=200, value=80, key=f"ap_lo{key_suffix}")
            cholesterol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "High"], index=0, key=f"chol{key_suffix}")
            gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "High"], index=0, key=f"gluc{key_suffix}")
            active = st.selectbox("Physical Activity", ["No", "Yes"], index=0, key=f"active{key_suffix}")
            alco = st.selectbox("Alcohol", ["No", "Yes"], index=0, key=f"alco{key_suffix}")

        predict_btn = st.form_submit_button("Predict Heart Disease Risk", use_container_width=True)

    if predict_btn:
        errors = []
        if not name.strip(): errors.append("Patient Name is required.")
        if age < 1 or age > 120: errors.append("Invalid age.")
        if height <= 0 or weight <= 0: errors.append("Height/Weight must be positive.")
        if ap_hi <= 0 or ap_lo <= 0: errors.append("BP values must be positive.")

        if errors:
            for err in errors: st.error(err)
            st.warning("Please fix errors above.")
        else:
            gender_val = 1 if gender == "Female" else 2
            smoke_val  = 1 if smoke  == "Yes" else 0
            alco_val   = 1 if alco   == "Yes" else 0
            active_val = 1 if active == "Yes" else 0

            chol_map = {"Normal":1, "Above Normal":2, "High":3}
            gluc_map = {"Normal":1, "Above Normal":2, "High":3}

            bmi = weight / ((height / 100) ** 2)

            input_data = np.array([[age*365, gender_val, height, weight, ap_hi, ap_lo,
                                    chol_map[cholesterol], gluc_map[gluc],
                                    smoke_val, alco_val, active_val]])

            input_scaled = scaler.transform(input_data)
            prediction   = model.predict(input_scaled)[0]
            probability  = model.predict_proba(input_scaled)[0][1] * 100
            result       = "High Risk" if prediction == 1 else "Low Risk"

            # ⭐ FIXED: Calculate BMI category HERE (after bmi is defined)
            if bmi < 18.5:
                bmi_category = "Underweight"
            elif bmi < 25:
                bmi_category = "Normal"
            elif bmi < 30:
                bmi_category = "Overweight"
            else:
                bmi_category = "Obese"

            st.session_state.prediction_result = {
                "risk": probability,
                "result": result,
                "bmi": bmi,
                "bmi_category": bmi_category   # ← now correctly included
            }

            st.session_state.prediction_result.update({
                "name": name.strip(),
                "age": age,
                "gender": gender,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": smoke,
                "alco": alco,
                "active": active
            })

            conn = sqlite3.connect("patients.db")
            conn.execute("""
                INSERT INTO patients
                (name,age,gender,height,weight,bmi,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active,prediction,risk,created_by)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (name, age, gender, height, weight, bmi, ap_hi, ap_lo,
                  cholesterol, gluc, smoke, alco, active, result, probability,
                  st.session_state.current_user))
            conn.commit()
            conn.close()

            st.success("Prediction saved!")
            st.session_state.form_version += 1
            st.rerun()

    if st.session_state.prediction_result:
        res = st.session_state.prediction_result
        st.subheader("Prediction Result")
        # ⭐ NEW FEATURE: Display BMI Category
        st.info(f"**BMI**: {res['bmi']:.2f} kg/m² → **{res['bmi_category']}**")
        if res["result"] == "High Risk":
            st.error(f"⚠ High Risk ({res['risk']:.1f}%)")
        else:
            st.success(f"✅ Low Risk ({res['risk']:.1f}%)")

        st.progress(int(res["risk"]))

        if res["risk"] > 70:   st.error("Immediate consultation recommended")
        elif res["risk"] > 40: st.warning("Moderate risk – lifestyle changes advised")
        else:                  st.success("Low cardiovascular risk")

        st.subheader("Heart Risk Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=res["risk"], title={'text': "Risk %"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "white"},
                   'steps': [{'range': [0,40],'color':"green"}, {'range':[40,70],'color':"yellow"}, {'range':[70,100],'color':"red"}]}))
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Generate Patient PDF Report"):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            c = canvas.Canvas(temp_file.name, pagesize=letter)
            c.setFont("Helvetica",16)
            c.drawString(200,750,"Patient Health Report")
            c.setFont("Helvetica",12)
            y=700
            data = [
                f"Patient Name: {res['name']}", f"Age: {res['age']}", f"Gender: {res['gender']}",
                f"Height: {res['height']} cm", f"Weight: {res['weight']} kg", f"BMI: {res['bmi']:.2f}",
                f"Systolic BP: {res['ap_hi']}", f"Diastolic BP: {res['ap_lo']}",
                f"Cholesterol: {res['cholesterol']}", f"Glucose: {res['gluc']}",
                f"Smoking: {res['smoke']}", f"Alcohol: {res['alco']}", f"Activity: {res['active']}",
                "", f"Result: {res['result']}", f"Risk: {res['risk']:.1f}%"
            ]
            for line in data:
                c.drawString(80,y,line)
                y -= 25
            c.save()

            with open(temp_file.name,"rb") as f:
                st.download_button("📄 Download Report", f, "patient_report.pdf", "application/pdf")

        if st.button("Reset Form / New Patient"):
            st.session_state.prediction_result = None
            st.session_state.form_version += 1
            st.rerun()


# ======================================================
# TAB 2 : VISUAL ANALYSIS (unchanged)
# ======================================================

with tab3:
    st.subheader("Dataset Visualisation")

    conn = sqlite3.connect("patients.db")
    if st.session_state.current_user == "admin":
        df = pd.read_sql_query("SELECT * FROM patients", conn)
    else:
        df = pd.read_sql_query("SELECT * FROM patients WHERE created_by = ?", conn, params=(st.session_state.current_user,))
    conn.close()

    if len(df) > 0:
        st.write(f"Showing: **{'All records (Admin)' if st.session_state.current_user == 'admin' else 'Your patients'}**")

        colA, colB = st.columns(2)
        with colA: st.bar_chart(df["prediction"].value_counts(), x_label="Result", y_label="Count")
        with colB: st.bar_chart(df.groupby("prediction")["risk"].mean(), x_label="Result", y_label="Avg Risk %")

        st.markdown("---")
        st.subheader("More Insights")
        colC, colD = st.columns(2)
        with colC:
            if "cholesterol" in df.columns:
                st.bar_chart(df.groupby("cholesterol")["risk"].mean(), x_label="Cholesterol", y_label="Avg Risk %")
        with colD:
            df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,120], labels=['<30','30-45','45-60','60+'])
            st.bar_chart(df.groupby("age_group")["risk"].mean(), x_label="Age Group", y_label="Avg Risk %")
    else:
        st.info("No records yet.")


# ======================================================
# TAB 3 : PATIENT HISTORY – FIXED VERSION
# ======================================================

with tab4:
    st.subheader("Patient Records")

    current_user = st.session_state.current_user
    is_admin = (current_user == "admin")

    search = st.text_input("Search by Name or Result", key="patient_search")

    conn = sqlite3.connect("patients.db")

    base_query = "SELECT * FROM patients" if is_admin else "SELECT * FROM patients WHERE created_by = ?"
    params_base = () if is_admin else (current_user,)

    if search:
        search_param = f"%{search}%"
        if is_admin:
            query = f"{base_query} WHERE name LIKE ? OR prediction LIKE ? ORDER BY id DESC"
            params = (search_param, search_param)
        else:
            query = f"{base_query} AND (name LIKE ? OR prediction LIKE ?) ORDER BY id DESC"
            params = params_base + (search_param, search_param)
    else:
        query = f"{base_query} ORDER BY id DESC"
        params = params_base

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Display table
    display_columns = ["id", "name", "age", "gender", "height", "weight", "prediction", "risk", "created_by"]
    display_df = df[display_columns].rename(columns={
        "id": "ID", "name": "Name", "age": "Age", "gender": "Gender", "height": "Height", "weight": "Weight",
        "prediction": "Result", "risk": "Risk %", "created_by": "Created By"
    })

    st.data_editor(
        display_df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID", disabled=True),
            "Risk %": st.column_config.NumberColumn("Risk %", format="%.1f"),
        }
    )

    # ⭐ NEW FEATURE: Export Patient Data CSV

    st.markdown("---")

    csv = df.to_csv(index=False)

    st.download_button(
        label="📥 Download Patient Data (CSV)",
        data=csv,
        file_name="patient_records.csv",
        mime="text/csv"
    )

    if is_admin and not df.empty:
        st.markdown("---")
        st.subheader("Admin Actions")

        col1, col2 = st.columns([3, 2])

        with col1:
            # EDIT PATIENT
            edit_options = ["— Select patient —"] + [
                f"{row['name']} (ID {row['id']}) – by {row['created_by']}"
                for _, row in df.iterrows()
            ]

            # Use key to force reset when editing_patient_id changes
            edit_selection = st.selectbox(
                "Edit patient record",
                options=edit_options,
                index=0 if st.session_state.editing_patient_id is None else edit_options.index(
                    next((opt for opt in edit_options if str(st.session_state.editing_patient_id) in opt), 0)
                ),
                key="edit_select_key"
            )

            if edit_selection != "— Select patient —":
                selected_edit_id = int(edit_selection.split("ID ")[1].split(")")[0])

                # Store the ID we're editing
                st.session_state.editing_patient_id = selected_edit_id

                patient = df[df["id"] == selected_edit_id].iloc[0]

                with st.form("edit_patient_form", clear_on_submit=False):
                    st.write(f"Editing: **{patient['name']}** (ID {selected_edit_id})")

                    edit_name     = st.text_input("Name", value=patient["name"])
                    edit_age      = st.number_input("Age", 1, 120, int(patient["age"]))
                    edit_gender   = st.selectbox("Gender", ["Female", "Male"], index=0 if patient["gender"]=="Female" else 1)
                    edit_height   = st.number_input("Height (cm)", value=float(patient["height"]))
                    edit_weight   = st.number_input("Weight (kg)", value=float(patient["weight"]))
                    edit_ap_hi    = st.number_input("Systolic BP", value=int(patient["ap_hi"]))
                    edit_ap_lo    = st.number_input("Diastolic BP", value=int(patient["ap_lo"]))
                    edit_chol     = st.selectbox("Cholesterol", ["Normal","Above Normal","High"],
                                                index=["Normal","Above Normal","High"].index(patient["cholesterol"]))
                    edit_gluc     = st.selectbox("Glucose", ["Normal","Above Normal","High"],
                                                index=["Normal","Above Normal","High"].index(patient["gluc"]))
                    edit_smoke    = st.selectbox("Smoking", ["No","Yes"], index=0 if patient["smoke"]=="No" else 1)
                    edit_alco     = st.selectbox("Alcohol", ["No","Yes"], index=0 if patient["alco"]=="No" else 1)
                    edit_active   = st.selectbox("Active", ["No","Yes"], index=0 if patient["active"]=="No" else 1)

                    if st.form_submit_button("💾 Save Changes"):
                        new_bmi = edit_weight / ((edit_height / 100) ** 2)
                        conn = sqlite3.connect("patients.db")
                        conn.execute("""
                            UPDATE patients SET
                                name=?, age=?, gender=?, height=?, weight=?, bmi=?,
                                ap_hi=?, ap_lo=?, cholesterol=?, gluc=?, smoke=?, alco=?, active=?
                            WHERE id = ?
                        """, (edit_name, edit_age, edit_gender, edit_height, edit_weight, new_bmi,
                              edit_ap_hi, edit_ap_lo, edit_chol, edit_gluc, edit_smoke, edit_alco, edit_active,
                              selected_edit_id))
                        conn.commit()
                        conn.close()

                        st.success(f"Patient **{edit_name}** (ID {selected_edit_id}) updated!")

                        # Reset edit mode
                        st.session_state.editing_patient_id = None
                        st.rerun()

        with col2:
            # DELETE PATIENT
            delete_options = ["— Select patient —"] + [
                f"{row['name']} (ID {row['id']}) – by {row['created_by']}"
                for _, row in df.iterrows()
            ]

            delete_selection = st.selectbox("Delete patient record", options=delete_options, key="delete_select_key")

            if delete_selection != "— Select patient —":
                delete_id = int(delete_selection.split("ID ")[1].split(")")[0])
                delete_name = delete_selection.split(" (ID")[0]

                if st.button(f"🗑️ Delete {delete_name}", type="primary"):
                    st.session_state.delete_confirm_id = delete_id
                    st.rerun()

                # Show confirmation only when delete_confirm_id matches
                if st.session_state.delete_confirm_id == delete_id:
                    st.error(f"**PERMANENT DELETE**: {delete_name} (ID {delete_id})", icon="🚨")

                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes – Delete Forever", type="primary"):
                            conn = sqlite3.connect("patients.db")
                            conn.execute("DELETE FROM patients WHERE id = ?", (delete_id,))
                            conn.commit()
                            conn.close()
                            st.session_state.delete_confirm_id = None
                            st.success(f"Deleted: {delete_name} (ID {delete_id})")
                            st.rerun()
                    with col_no:
                        if st.button("No – Cancel"):
                            st.session_state.delete_confirm_id = None
                            st.info("Delete cancelled.")
                            st.rerun()

    # Admin stats
    # ⭐ NEW FEATURE: Dashboard Statistics

    if is_admin:

        st.markdown("---")
        st.subheader("Admin Dashboard Statistics")

        total_patients = len(df)
        high_risk = len(df[df["prediction"] == "High Risk"])
        low_risk = len(df[df["prediction"] == "Low Risk"])

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Patients", total_patients)
        col2.metric("High Risk Cases", high_risk)
        col3.metric("Low Risk Cases", low_risk)

        st.markdown("---")

        if len(df) > 0:
            st.write("Records per doctor:")
            st.write(df["created_by"].value_counts())

