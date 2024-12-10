import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Dashboard für die Zeiterfassung")

uploaded_file = st.file_uploader("CSV-Datei hochladen", type="csv")

if uploaded_file is not None:
    try:
        # Datei lesen
        data = pd.read_csv(uploaded_file)

        # Datenverarbeitung
        def process_data_arbeitszeit(data):
            #konvertieren der Daten
            data["Arbeitszeit in Stunden"] = data["Verbleibende Schätzung"] / 3600
            data["Σ Arbeitszeit in Stunden"] = data["Σ Verbleibende Schätzung"] / 3600
            data['Arbeitszeit in Stunden'].fillna(0, inplace=True)  # Ersetze fehlende Arbeitszeiten durch 0
            return data

        def process_data_epic(data):
            tasks_df = data[data['Vorgangstyp'] == 'Task']
            epic_remaining_work = tasks_df.groupby('Parent summary')['Σ Arbeitszeit in Stunden'].sum().reset_index()
            epic_remaining_work.columns = ['Epic', 'Arbeitszeit in Stunden']
            return epic_remaining_work

        processed_data_arbeitszeit = process_data_arbeitszeit(data)
        processed_data_epic = process_data_epic(data)

        # Visualisierung
        def visualize_arbeitszeit(data):
            """Visualisiert die Arbeitszeit pro Person."""
            col1, col2 = st.columns(2)
            with col1:
                # Pie Chart 1: Verbleibende Schätzung in Stunden pro Zugewiesene Person
                pie_chart1 = px.pie(data, names="Zugewiesene Person", values="Arbeitszeit in Stunden")
                st.plotly_chart(pie_chart1)
            with col2:
                # Histogramm 1: Verbleibende Schätzung in Stunden pro Zugewiesene Person
                hist1 = px.histogram(data, x="Zugewiesene Person", y="Arbeitszeit in Stunden", nbins=10)
                st.plotly_chart(hist1)

        def visualize_epic(data):
            """Visualisiert die Arbeitszeit pro Epic."""
            col3, col4 = st.columns(2)
            with col3:
                # Pie Chart 2: Σ Verbleibende Schätzung in Stunden pro Parent summary, wenn Vorgangstyp 'Epic'
                pie_chart2 = px.pie(data, names="Epic", values="Arbeitszeit in Stunden")
                st.plotly_chart(pie_chart2)
            with col4:
                # Histogramm 2: Σ Verbleibende Schätzung in Stunden als Balkendiagramm
                hist2 = px.bar(data, x="Epic", y="Arbeitszeit in Stunden")
                st.plotly_chart(hist2)

        # Visualisierungen
        visualize_arbeitszeit(processed_data_arbeitszeit)
        visualize_epic(processed_data_epic)

        # Neue Abschnitte nach unten schieben
        st.subheader("Nicht Zugewiesene Tasks")
        data_nicht_zugewiesen = data[pd.isna(data["Zugewiesene Person"])]
        # Füge eine neue Spalte mit der Umrechnung in Stunden hinzu
        data_nicht_zugewiesen["Verbleibende Schätzung in Stunden"] = data_nicht_zugewiesen[
                                                                         "Σ Verbleibende Schätzung"] / 3600
        # Entferne Spalten mit nur None-Werten und behalte nur die gewünschten Spalten
        gewuenschte_spalten = ["Zusammenfassung", "Vorgangsschlüssel", "Vorgangstyp", "Status", "Zugewiesene Person", "Übergeordnet",
                               "Verbleibende Schätzung in Stunden"]
        data_nicht_zugewiesen = data_nicht_zugewiesen[gewuenschte_spalten]
        st.dataframe(data_nicht_zugewiesen)

        st.subheader("Tasks im Status 'In Progress'")
        data_tasks_inprogress = data[data["Status"] == "In Arbeit"]
        # Füge eine neue Spalte mit der Umrechnung in Stunden hinzu
        data_tasks_inprogress["Verbleibende Schätzung in Stunden"] = data_tasks_inprogress[
                                                                         "Σ Verbleibende Schätzung"] / 3600
        # Entferne Spalten mit nur None-Werten und behalte nur die gewünschten Spalten
        data_tasks_inprogress = data_tasks_inprogress[gewuenschte_spalten]
        st.dataframe(data_tasks_inprogress)

    except Exception as e:
        st.error(f"Ein Fehler ist beim Lesen der Datei aufgetreten: {e}")
else:
    st.warning("Bitte lade eine Datei hoch")