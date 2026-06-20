#!/usr/bin/env python3
"""Führt EDA.ipynb aus und speichert das executed Notebook im gleichen Ordner."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.validator import normalize


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    notebook = project / "notebooks" / "EDA.ipynb"
    output_path = project / "notebooks" / "EDA_executed.ipynb"

    (project / "results").mkdir(exist_ok=True)

    if not notebook.is_file():
        print(f"Notebook fehlt: {notebook}", file=sys.stderr)
        return 1

    print("=== EDA Notebook ausführen ===")
    print(f"Input:  {notebook}")
    print(f"Output: {output_path}")

    with notebook.open(encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    normalize(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    client = NotebookClient(
        nb,
        timeout=-1,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook.parent)}},
    )
    client.execute()

    with output_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Fertig: {output_path}")
    print(f"Plots:  {project / 'results' / 'eda_*.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
