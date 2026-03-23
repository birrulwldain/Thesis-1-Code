import re
import subprocess
import sys

print("Mengeksekusi 24 sampel (harap tunggu ~15 detik)...")
result = subprocess.run(["bash", "./run-blok4-all.sh"], capture_output=True, text=True)
content = result.stdout

if result.returncode != 0:
    print("Error execution script:", result.stderr)

sample_pattern = re.compile(r"Ditemukan Ground Truth historis untuk '(S\d+)'!")
te_pattern = re.compile(r"Suhu \(Te\)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+([\d.]+)%")
ne_pattern = re.compile(r"Densit\(ne\)\s+\|\s+([\d.e\+]+)\s+\|\s+([\d.e\+]+)\s+\|\s+([\d.]+)%")

samples = sample_pattern.findall(content)
te_matches = te_pattern.findall(content)
ne_matches = ne_pattern.findall(content)

print("\n### REKAPITULASI ADU MEKANIK: AI v Saha-Boltzmann (S1 - S24)\n")
print("| Sampel | $T_e$ AI (K) | $T_e$ Saha (K) | Error $T_e$ (%) | $n_e$ AI (cm⁻³) | $n_e$ Saha (cm⁻³) | Error $n_e$ (%) |")
print("|:---:|---:|---:|---:|---:|---:|---:|")

for i in range(len(samples)):
    try:
        s = samples[i]
        te_svr, te_saha, te_err = te_matches[i]
        ne_svr, ne_saha, ne_err = ne_matches[i]
        print(f"| {s} | {te_svr} | {te_saha} | {te_err}% | {ne_svr} | {ne_saha} | {ne_err}% |")
    except IndexError:
        pass
