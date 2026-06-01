"""Produce a SINGLE self-contained main.tex for Elsevier per-file upload.

Elsevier's submission system flattens all uploaded files into one directory, so
subfolders (sections/, tables/, figures/) and ../results/ paths fail. This script
inlines every \\input (sections + result tables), inlines the bibliography
(main.bbl), neutralizes the \\IfFileExists guards (always take the present branch),
and flattens figure paths to bare filenames.

Output -> dist/submission/  containing:
  main.tex                      (one self-contained file)
  <figure>.pdf  x4              (flat)
You upload main.tex + the 4 figure PDFs. Nothing else.
"""
from __future__ import annotations
import re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "dist" / "submission"
FIGS = ROOT / "results" / "figures"
TBLS = ROOT / "results" / "tables"


def _match_brace(s, k):
    """s[k] must be '{'. Return index just past the matching '}'. Skips \\{ \\}."""
    depth = 0
    while k < len(s):
        c = s[k]
        if c == "\\":
            k += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return k + 1
        k += 1
    return k


def _resolve_iffileexists(s):
    """Replace each \\IfFileExists{a}{b}{c} with b (the file-present branch)."""
    key = r"\IfFileExists"
    out, i = [], 0
    while True:
        j = s.find(key, i)
        if j == -1:
            out.append(s[i:])
            break
        out.append(s[i:j])
        k = j + len(key)
        args = []
        ok = True
        while len(args) < 3:
            while k < len(s) and s[k] in " \t\r\n%":
                if s[k] == "%":  # skip a comment line
                    nl = s.find("\n", k)
                    k = len(s) if nl == -1 else nl + 1
                else:
                    k += 1
            if k >= len(s) or s[k] != "{":
                ok = False
                break
            end = _match_brace(s, k)
            args.append(s[k + 1:end - 1])
            k = end
        if ok and len(args) == 3:
            out.append(args[1])  # true / file-present branch
            i = k
        else:                    # malformed: leave as-is
            out.append(s[j:k])
            i = k
    return "".join(out)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    text = (PAPER / "main.tex").read_text(encoding="utf-8")

    # 1) inline \input{sections/NAME}
    def sec(m):
        name = m.group(1)
        p = PAPER / "sections" / (name.split("/")[-1] + ".tex")
        return p.read_text(encoding="utf-8")
    text = re.sub(r"\\input\{sections/([^}]+)\}", sec, text)

    # 2) inline \input{../results/tables/NAME.tex} (with or without .tex)
    def tbl(m):
        name = m.group(1)
        if not name.endswith(".tex"):
            name += ".tex"
        p = TBLS / Path(name).name
        return p.read_text(encoding="utf-8") if p.exists() else m.group(0)
    text = re.sub(r"\\input\{\.\./results/tables/([^}]+)\}", tbl, text)

    # 3) flatten figure paths: ../results/figures/NAME -> NAME
    text = text.replace("../results/figures/", "")

    # 4) resolve \IfFileExists{path}{TRUE}{FALSE} -> TRUE branch, by brace
    #    matching (do NOT redefine \IfFileExists globally -- pgf/TikZ uses it
    #    internally). Tables are already inlined and figures already flat, so the
    #    "present" branch is the correct one.
    text = _resolve_iffileexists(text)

    # 5) inline bibliography: replace \bibliographystyle+\bibliography with main.bbl
    bbl = (PAPER / "main.bbl").read_text(encoding="utf-8")
    text = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}",
                  lambda _m: bbl, text)

    (OUT / "main.tex").write_text(text, encoding="utf-8")

    # copy the 4 figures (flat)
    figs = ["learned_filter.pdf", "arevin_profile.pdf", "accuracy_vs_params.pdf",
            "synthetic_drift.pdf"]
    n = 0
    for f in figs:
        if (FIGS / f).exists():
            shutil.copy2(FIGS / f, OUT / f); n += 1

    print(f"wrote {OUT}\\main.tex (self-contained) + {n} figures")
    print("UPLOAD THESE FILES (all flat, no folders):")
    for p in sorted(OUT.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
