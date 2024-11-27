from xml.dom.minidom import Document

def save_as_svg(prediction, output_path):
    """
    Speichert eine vorhergesagte Label-Maske als SVG-Datei.

    :param prediction: 2D-Array mit Labels (0 = Hintergrund, 1 = Wand, 2 = Tür, 3 = Fenster)
    :param output_path: Pfad zur Ausgabe der SVG-Datei
    """
    doc = Document()

    # Erstelle das root <svg>-Element
    svg = doc.createElement("svg")
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg")
    svg.setAttribute("version", "1.1")
    doc.appendChild(svg)

    h, w = prediction.shape

    # Iteriere durch die Vorhersage-Maske und erstelle Rechtecke für jedes Pixel
    for y in range(h):
        for x in range(w):
            element = prediction[y, x]
            if element == 1:  # Wand
                rect = doc.createElement("rect")
                rect.setAttribute("x", str(x))
                rect.setAttribute("y", str(y))
                rect.setAttribute("width", "1")
                rect.setAttribute("height", "1")
                rect.setAttribute("fill", "black")
                svg.appendChild(rect)
            elif element == 2:  # Tür
                rect = doc.createElement("rect")
                rect.setAttribute("x", str(x))
                rect.setAttribute("y", str(y))
                rect.setAttribute("width", "1")
                rect.setAttribute("height", "1")
                rect.setAttribute("fill", "blue")
                svg.appendChild(rect)
            elif element == 3:  # Fenster
                rect = doc.createElement("rect")
                rect.setAttribute("x", str(x))
                rect.setAttribute("y", str(y))
                rect.setAttribute("width", "1")
                rect.setAttribute("height", "1")
                rect.setAttribute("fill", "green")
                svg.appendChild(rect)

    # Schreibe das Dokument in die Ausgabe-Datei
    with open(output_path, "w") as f:
        f.write(doc.toprettyxml(indent="  "))