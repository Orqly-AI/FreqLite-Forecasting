"""Assemble a self-contained LaTeX SOURCE zip for Elsevier/KBS submission.

Elsevier compiles the LaTeX itself, so we flatten the project: the paper's
`../results/tables/*.tex` and `../results/figures/*.pdf` references are rewritten
to local `tables/` and `figures/` subfolders, and the pre-built `main.bbl` is
bundled alongside `refs.bib`. Output: dist/FreqLite_LaTeX_source.zip
"""
from __future__ import annotations
import shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "dist" / "kbs_source"
ZIP = ROOT / "dist" / "FreqLite_LaTeX_source.zip"

REWRITES = [("../results/tables/", "tables/"), ("../results/figures/", "figures/")]


def rewrite(text: str) -> str:
    for a, b in REWRITES:
        text = text.replace(a, b)
    return text


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "sections").mkdir(parents=True)
    (OUT / "tables").mkdir()
    (OUT / "figures").mkdir()

    # main.tex (path-rewritten; NO arXiv \pdfoutput line for Elsevier)
    (OUT / "main.tex").write_text(rewrite((PAPER / "main.tex").read_text(encoding="utf-8")),
                                  encoding="utf-8")
    for f in sorted((PAPER / "sections").glob("*.tex")):
        (OUT / "sections" / f.name).write_text(rewrite(f.read_text(encoding="utf-8")),
                                               encoding="utf-8")
    # bibliography: bundle .bib (Elsevier runs bibtex) + pre-built .bbl as a fallback
    shutil.copy2(PAPER / "refs.bib", OUT / "refs.bib")
    if (PAPER / "main.bbl").exists():
        shutil.copy2(PAPER / "main.bbl", OUT / "main.bbl")

    n_t = 0
    for f in sorted((ROOT / "results" / "tables").glob("*.tex")):
        shutil.copy2(f, OUT / "tables" / f.name); n_t += 1
    needed = ["learned_filter.pdf", "arevin_profile.pdf", "accuracy_vs_params.pdf",
              "synthetic_drift.pdf"]
    n_f = 0
    for name in needed:
        src = ROOT / "results" / "figures" / name
        if src.exists():
            shutil.copy2(src, OUT / "figures" / name); n_f += 1

    ZIP.parent.mkdir(exist_ok=True)
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT))

    print(f"assembled {n_t} tables, {n_f} figures, main.tex + "
          f"{len(list((OUT/'sections').glob('*.tex')))} sections + refs.bib + main.bbl")
    print(f"zip -> {ZIP}  ({ZIP.stat().st_size/1024:.0f} KB)")
    with zipfile.ZipFile(ZIP) as z:
        for n in z.namelist():
            print("  ", n)


if __name__ == "__main__":
    main()
