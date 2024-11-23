import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Dashboard für die Zeiterfassung")

# CSV-Datei hochladen
uploaded_file = st.file_uploader("CSV-Datei hochladen", type="csv")

if uploaded_file:
    # CSV-Datei lesen
    data = pd.read_csv(uploaded_file)

    # Konvertieren von "Verbleibende Schätzung" und "Σ Verbleibende Schätzung" in Stunden
    data["Arbeitszeit in Stunden"] = data["Verbleibende Schätzung"] / 3600
    data["Σ Arbeitszeit in Stunden"] = data["Σ Verbleibende Schätzung"] / 3600

    st.subheader("Arbeitszeit in Stunden pro Zugewiesene Person")

    col1, col2 = st.columns(2)

    with col1:
        # Pie Chart 1: Verbleibende Schätzung in Stunden pro Zugewiesene Person
        pie_chart1 = px.pie(data, names="Zugewiesene Person", values="Arbeitszeit in Stunden")
        st.plotly_chart(pie_chart1)

    with col2:
        # Histogramm 1: Verbleibende Schätzung in Stunden pro Zugewiesene Person
        hist1 = px.histogram(data, x="Zugewiesene Person", y="Arbeitszeit in Stunden", nbins=10)
        st.plotly_chart(hist1)

    st.subheader("Arbeitszeit in Stunden pro Epic")

    col3, col4 = st.columns(2)

    # Filter for rows where "Vorgangstyp" is "Task"
    tasks_df = data[data['Vorgangstyp'] == 'Task']

    # Group by "Parent summary" (Epic) and sum the "Σ Verbleibende Schätzung"
    epic_remaining_work = tasks_df.groupby('Parent summary')['Σ Arbeitszeit in Stunden'].sum().reset_index()

    epic_remaining_work.columns = ['Epic', 'Arbeitszeit in Stunden']

    with col3:
        # Pie Chart 2: Σ Verbleibende Schätzung in Stunden pro Parent summary, wenn Vorgangstyp 'Epic'
        pie_chart2 = px.pie(epic_remaining_work, names="Epic", values="Arbeitszeit in Stunden")
        st.plotly_chart(pie_chart2)

    with col4:
        # Histogramm 2: Σ Verbleibende Schätzung in Stunden pro Parent summary (wenn Vorgangstyp 'Epic')
        hist2 = px.histogram(epic_remaining_work, x="Epic", y="Arbeitszeit in Stunden", nbins=10)
        st.plotly_chart(hist2)

    st.subheader("Nicht Zugewiesene Tasks")
    data_nicht_zugewiesen = data[pd.isna(data["Zugewiesene Person"])]
    st.dataframe(data_nicht_zugewiesen)

    st.subheader("Tasks im Status 'In Progress'")
    data_tasks_inprogress = data[data["Status"] == "In Arbeit"]
    st.dataframe(data_tasks_inprogress)
