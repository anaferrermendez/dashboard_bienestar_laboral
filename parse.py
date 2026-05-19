import json

with open("Proyecto_final_preprocesamiento - GXX.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("nb_output.txt", "w", encoding="utf-8") as f_out:
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            f_out.write(f"--- Cell {i} ---\n")
            f_out.write("".join(cell["source"]) + "\n\n")

