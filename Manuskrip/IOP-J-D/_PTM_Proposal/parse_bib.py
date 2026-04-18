import re

with open("/Users/birrulwldain/Projects/Thesis-1-Code/Manuskrip/IOP-J-D/_PTM_Proposal/Roadmap/Roadmap.bib", "r", encoding="utf-8") as f:
    text = f.read()

entries = re.split(r"\n\s*@\w+\s*\{", "\n" + text)
entries = [e for e in entries if "=" in e]

out_md = []
out_md.append("# Analisis Ekstraktif Peta Jalan Riset `Roadmap.bib`\n")

outliers = []
group_papers = []

for entry in entries:
    def get_val(matcher):
        m = re.search(r"\b" + matcher + r"\s*=\s*[\{](.*?)[\}]", entry, re.IGNORECASE | re.DOTALL)
        if not m:
            m = re.search(r"\b" + matcher + r"\s*=\s*\"(.*?)\"", entry, re.IGNORECASE | re.DOTALL)
        if not m:
            m = re.search(r"\b" + matcher + r"\s*=\s*(.*?)[,\n]", entry, re.IGNORECASE | re.DOTALL)
        return m.group(1).replace("\n", " ").strip() if m else ""

    author = get_val("author")
    title = get_val("title")
    year = get_val("year")
    if not year.isdigit(): year = "2024"

    is_idris = bool(re.search(r"Idris", author, re.IGNORECASE))
    is_mita = bool(re.search(r"Mitaphonna", author, re.IGNORECASE))
    is_wali = bool(re.search(r"Walidain", author, re.IGNORECASE))

    item = (year, author, title, is_mita, is_wali)
    
    if is_idris:
        group_papers.append(item)
    else:
        outliers.append(item)

out_md.append(f"Dari total **{len(entries)}** publikasi, terdapat **{len(group_papers)}** artikel grup riset dan **{len(outliers)}** artikel outlier.\n")

out_md.append("## 1. Identifikasi Outliers (Non-Grup / Tanpa Idris)\n")
for o in outliers:
    out_md.append(f"- **{o[1]}** ({o[0]}): _{o[2]}_\n")

out_md.append("\n## 2. Peta Jalan & Tematik Grup (Berdasarkan Tahun)\n")

group_papers.sort(key=lambda x: int(x[0]))
last_theme = ""

for p in group_papers:
    y = int(p[0])
    tit = p[2].lower()
    
    theme = ""
    if y < 2014: theme = "Fase 1: Fundamental Plasma (2004-2013)"
    elif y <= 2021: theme = "Fase 2: Geokimia Kealaman & Tsunami (2014-2021)"
    elif y <= 2024:
        if "soil" in tit or "tsunami" in tit or "deposit" in tit: theme = "Fase 3: Eksplorasi Observasional (2022-2024)"
        elif "egg" in tit or "organic" in tit: theme = "Fase 3: Material Eksotis & Organik (2024)"
        else: theme = "Fase 3: Optimasi Instrumentasi (2022-2024)"
    else: theme = "Fase 4: Riset AI Spektral Terkini (2025-2026)"

    if theme != last_theme:
        out_md.append(f"\n### {theme}\n")
        last_theme = theme
        
    tag = ""
    if p[4]: tag = " **[Penelitian Walidain]**"
    elif p[3]: tag = " **[Penelitian Mitaphonna]**"
    
    out_md.append(f"- **{p[1].split('and')[0].strip()} et al. ({p[0]})**: _{p[2]}_{tag}\n")

with open("/Users/birrulwldain/.gemini/antigravity/brain/7edc3355-68a8-4ee6-97ac-4885e48ff7b3/Analisis_Roadbib.md", "w", encoding="utf-8") as f:
    f.writelines(out_md)

