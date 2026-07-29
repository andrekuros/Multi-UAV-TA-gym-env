"""Structural sanity check for paper/main.tex (no pdflatex required)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tex_path = ROOT / "paper" / "main.tex"
tex = tex_path.read_text(encoding="utf-8")
code_chars: list[str] = []
i = 0
while i < len(tex):
    if tex[i] == "\\" and i + 1 < len(tex):
        code_chars.append(tex[i])
        code_chars.append(tex[i + 1])
        i += 2
        continue
    if tex[i] == "%":
        while i < len(tex) and tex[i] != "\n":
            i += 1
        continue
    code_chars.append(tex[i])
    i += 1
code = "".join(code_chars)
bal = 0
ok_braces = True
for ch in code:
    if ch == "{":
        bal += 1
    elif ch == "}":
        bal -= 1
        if bal < 0:
            ok_braces = False
            break
print(f"brace_balance={bal} ok={ok_braces and bal == 0}")

labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
missing = sorted(refs - labels)
orphan = sorted(labels - refs)
print(f"labels={len(labels)} refs={len(refs)}")
print("missing_refs:", missing)
print("unreferenced_labels:", orphan)

for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
    rel = m.group(1)
    cands = [
        ROOT / "paper" / rel,
        ROOT / "paper" / f"{rel}.png",
        ROOT / "paper" / f"{rel}.pdf",
    ]
    ok = any(c.exists() for c in cands)
    print(f"fig {rel}: {'OK' if ok else 'MISSING'}")

raise SystemExit(0 if ok_braces and bal == 0 and not missing else 1)
