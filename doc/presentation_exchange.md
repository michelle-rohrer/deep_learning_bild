# Muskel-Segmentierung mit Bayesian 3D U-Net
### FHNW Mini-Challenge – Deep Learning Bild & Signal | M3 Exchange
**Sabina & Michelle**

---

## Folie 1 · Problemstellung & Use Case

**Was?**  
Automatische Segmentierung von **8 linken Oberschenkelmuskeln** in 3D-MRT-Volumen (DIXON Water-Images, GCAP-Studie)

**Muskeln (Klassen 1–8):**  
Quadrizeps links: VL_L · RF_L · VM_L · VI_L  
Hamstrings links: BfSH_L · BfLH_L · ST_L · SM_L

**Warum wichtig?**  
Präzise Muskelsegmentierung ist Grundlage für Muskelvolumen-Quantifizierung  
→ klinisch relevant für Muskelatrophie, Rehabilitation, Biomechanik

**Challenge:**  
- 3D-MRI-Volumen sind sehr gross (~704×508×640 Voxel)  
- Kleiner Datensatz (n=39 Subjects)  
- Extreme Klassen-Imbalance: ~99% Hintergrund

---

## Folie 2 · Datensatz-Übersicht

| Kennzahl | Wert |
|---|---|
| Subjects gesamt | 39 |
| Nur links annotiert | 30 |
| Bilateral annotiert | 9 |
| Maskendateien | 45 (.nii.gz) |
| Voxelgrösse (typisch) | 0.65 × 0.65 × 2.0 mm |
| Bildgrösse (typisch) | 704 × 508 × 640 Voxel |
| Klassen (Training) | 9 (BG + 8 linke Muskeln) |
| **Ausgeschlossen** | Subject 543 (Quadrizeps links fehlt) |
| **Trainings-Split** | 3-Fold CV auf Fall-Ebene (38 Subjects) |

**Fold-Grössen:** Fold 0/1: 25 Train / 13 Val · Fold 2: 26 Train / 12 Val

---

## Folie 3 · EDA – Bildgeometrie

**Was man sieht:**  
36 von 39 Subjects: exakt `(704, 508, 640)` Voxel · 3 Subjects: 626 Schichten  
Voxelgrösse: `0.653 × 0.653 × 2.0 mm` bei **allen** Subjects identisch

**Interpretation / Konsequenzen:**

✔ **Kein Resampling nötig** – perfekt homogene Geometrie vereinfacht die Pipeline massiv  
⚠ **Anisotropie:** In-plane Auflösung (0.65 mm) ist ~3× feiner als Schichtdicke (2.0 mm)  
→ 3D-Patches müssen dies berücksichtigen: `128×128×32` statt kubischer Patches  
→ Anisotrope Kernel oder bewusst dickere Z-Patches wählen  
⚠ **Volumen zu gross für direktes Training** → Patch-basiertes Sampling zwingend  
Gesamtgrösse pro Volume: ~228 Mio. Voxel ≈ ~900 MB pro Subject (float32)

---

## Folie 4 · EDA – Label-Präsenz & Annotation-Qualität

**Was man sieht (Label-Präsenz-Heatmap):**  
- Linke 8 Muskeln: fast vollständig bei allen Subjects  
- Rechte Hälfte: grossteils leer (nur 9 bilateral annotiert)  
- Subject 543 springt ins Auge: Quadrizeps links (Labels 1–4) komplett weiss  
- 4 Subjects mit unvollständigem Label-Set (502, 505, 524, 543)

**Interpretation / Konsequenzen:**

✔ **39 nutzbare Subjects** für linke Muskeln – alle 8 Labels vorhanden  
✗ **Subject 543** → ausgeschlossen (Quadrizeps links fehlt vollständig)  
→ 38 Subjects im Training (nach Ausschluss)  
**Rechte Muskeln** werden auf Hintergrund gemappt, nicht in Dice einbezogen  
→ klare Fokussierung, kein partielles-Label-Problem im Training  
**Inkonsistente Dateinamen** (`GCAP512mask.nii.gz` vs. `GCAP_512_mask.nii.gz`)  
→ robustes Inventar-Skript mit Regex-Matching implementiert

---

## Folie 5 · EDA – Muskelvolumen & Inter-Subject-Variabilität

**Was man sieht (Boxplot Muskelvolumen):**

| Muskel | Median | Spanne |
|---|---|---|
| VL_L (Vastus lateralis) | **614 ml** | 92–971 ml |
| RF_L (Rectus femoris) | 279 ml | 6–396 ml |
| VM_L (Vastus medialis) | 406 ml | 146–644 ml |
| VI_L (Vastus intermedius) | 439 ml | 53–760 ml |
| **BfSH_L** (Biceps femoris KK) | **97 ml** | 0.3–197 ml |
| BfLH_L | 180 ml | 2–290 ml |
| ST_L (Semitendinosus) | 203 ml | 1–328 ml |
| SM_L (Semimembranosus) | 216 ml | 1–329 ml |

**Interpretation:**

⚠ **Grössenratio ~6:1** zwischen grösstem (VL_L) und kleinstem Muskel (BfSH_L)  
→ Das Modell wird VL gut lernen, BfSH ist schwieriger (kleiner, weniger Voxel)  
⚠ **Hohe Variabilität** (z.B. VL_L: ±192 ml) = starke inter-individuelle Unterschiede  
→ Augmentation hilft, aber Cross-Validation ist zwingend für valide Schätzung  
⚠ **Sehr kleine Minima** bei BfSH_L (0.3 ml) und BfLH_L (1.9 ml)  
→ Mögliche Qualitätsprobleme in diesen Annotationen; BfSH ist schwer zu segmentieren  
✔ **Per-class Dice Loss** gewichtet jeden Muskel gleich unabhängig von seiner Grösse

---

## Folie 6 · EDA – Klassen-Imbalance

**Was man sieht:**  
- **Median Muskel-Anteil: 1.27%** der Voxel (Min 0.17%, Max 2.52%)  
- Gesamt-Verhältnis Hintergrund zu Muskel: **77:1**  
- Über 98% jedes Bildes ist Hintergrund

**Interpretation – kritischster Faktor für Training:**

✗ **Pixel-Accuracy ist wertlos** – ein Klassifikator, der alles als Hintergrund vorhersagt, hat >98% Accuracy, segmentiert aber nichts  
→ **Dice Score / IoU** als einzige valide Metriken  
✔ **Dice Loss** eliminiert Imbalance-Problem automatisch:  
$$\mathcal{L}_\text{Dice} = 1 - \frac{2 \sum p_i g_i}{\sum p_i + \sum g_i}$$  
dividiert jeweils durch Gesamtgrösse → kleine Klassen werden nicht dominiert  
✔ **Patch-Sampling**: Patches nur aus Regionen mit Muskelinhalt → leere Patches überspringen  
→ Trainingseffizienz stark verbessert

---

## Folie 7 · EDA – Slice-Coverage & Z-Verteilung

**Was man sieht (Coverage-Boxplots):**  
- **Quadrizeps** (VL, RF, VM, VI): ~24–26% der Schichten enthalten Muskel  
- **Hamstrings** (BfSH, BfLH, ST, SM): ~18–22% der Schichten  
- Z-Span: Muskeln erstrecken sich über ~115–167 Schichten (bei 640 gesamt)

**Interpretation:**

✔ **75–80% der Schichten sind leer** → Patch-Sampling kann leer Schichten komplett überspringen  
→ deutliche Einsparung an Rechenzeit ohne Informationsverlust  
⚠ **Z-Position variiert stark zwischen Subjects** (std ~5%)  
→ keine fixen Z-Grenzen sinnvoll; besser dynamisch aus Maske bestimmen  
⚠ **Sehr tiefe Minima** (0.2% Coverage bei Hamstrings) = einige Subjects haben nur wenige annotierte Schichten  
→ auf Subject-Ebene splitten, nicht auf Slice-Ebene (kein Slice-Leakage!)  
✔ **Unsere Splits** sind auf Fall-Ebene stratifiziert → kein Datenleck zwischen Folds

---

## Folie 8 · EDA – Zusammenfassung der 5 Kernerkenntnisse

| # | Befund | Konsequenz im Design |
|---|---|---|
| 1 | **n=38 Subjects** (sehr klein) | 3-Fold CV auf Fall-Ebene + kein Slice-Splitting |
| 2 | **30/38 nur links annotiert** | Fokus auf linke 8 Klassen; rechte Labels → BG |
| 3 | **Imbalance 77:1** (BG:Muskel) | Dice Loss · Patch-Sampling · Dice als Metrik |
| 4 | **Anisotrop 0.65/0.65/2.0 mm** | Patches 128×128×32 (anisotrop) |
| 5 | **VL:BfSH ≈ 6:1** (Label-Imbalance) | Macro-Dice (gleichgewichtet, unabhängig von Grösse) |

---

## Folie 9 · Methoden – Architektur: Bayesian 3D U-Net

```
Input Patch (1 × 128 × 128 × 32)
       │
   [Enc 1]  16 ch   ──── skip 1 ────┐
   [Pool]                             │
   [Enc 2]  32 ch   ──── skip 2 ──┐  │
   [Pool]                          │  │
   [Enc 3]  64 ch   ──── skip 3 ─┐│  │
   [Pool]                         ││  │
   [Enc 4]  128 ch  ──── skip 4 ┐││  │
   [Pool]                        ││││
 [Bottleneck] 128 ch             ││││
   [Upsample] ◄──────── skip 4 ─┘│││
   [Dec 4]  64 ch                 │││
   [Upsample] ◄──────── skip 3 ──┘││
   [Dec 3]  32 ch                  ││
   [Upsample] ◄──────── skip 2 ───┘│
   [Dec 2]  16 ch                   │
   [Upsample] ◄──────── skip 1 ────┘
   [Dec 1]  16 ch
   [Head 1×1×1] → 9 Klassen (Logits)
```

**Jeder ConvBlock:**  
`Conv3d → InstanceNorm3d → ReLU → Dropout3d(p=0.3) → Conv3d → InstanceNorm3d → ReLU → Dropout3d(p=0.3)`

**Warum Bayesian (MC Dropout)?**  
Dropout bleibt auch zur Inferenzzeit aktiv → N=20 stochastische Forward-Passes  
→ **Mittelwert der Softmax-Probs** = Vorhersage · **Varianz** = Unsicherheitskarte  
→ Modell gibt an, wie sicher es in welchem Bereich ist

---

## Folie 10 · Methoden – Training-Pipeline

**Hyperparameter Baseline:**

| Parameter | Wert | Begründung |
|---|---|---|
| Patch-Grösse | 128×128×32 | anisotrop (2.0 mm Z) |
| Patches/Volume | 8 | Speicher vs. Diversität |
| Batch-Grösse | 2 | GPU-Speicher |
| Loss | Dice Loss | Imbalance 77:1 |
| Optimizer | Adam, lr=1e-4 | Standard Baseline |
| Dropout | 0.3 | MC Bayesian |
| Epochs | 80 | Baseline |
| Augmentation | **Nein** | Baseline-Einstieg |
| MC-Samples | 20 | Inferenz-Stabilität |
| Cross-Validation | 3-Fold (Fall-Ebene) | n=38 klein |

**Ablauf:**  
`make_splits.py` → Folds erzeugen (kein Slice-Leakage)  
`run_overfit.py` → Sanity-Check (Subject 512, Ziel Dice ≥ 0.90)  
`train.py` → 3-Fold Baseline-Training + WandB-Logging  
`evaluate.py` → Checkpoint-Evaluation → `eval_report.json`

---

## Folie 11 · Baseline-Performance – Overfitting-Test

**Ziel:** Technischer Sanity-Check – kann das Modell *prinzipiell* segmentieren?  
**Setup:** 1 Subject (512), 200 Epochs, 50 Steps/Epoch, Dice-Ziel ≥ 0.90

**Was wir testen:**
- Modell ist korrekt implementiert (U-Net, Loss, Metrik)
- Gradient-Fluss funktioniert durch alle Layer
- Patches werden korrekt geladen und verarbeitet
- MC-Inferenz liefert konsistente Ergebnisse

**Erwartetes Ergebnis:**  
Dice ≥ 0.90 auf dem einzigen Trainings-Subject → bestätigt Implementierungs-Korrektheit

**Bedeutung für Generalisierung:**  
Ein Modell, das **nicht** auf einem einzigen Fall overfitten kann, hat einen Bug.  
→ Overfitting-Test als technischer Mindest-Standard vor echtem Training

---

## Folie 12 · Baseline-Performance – 3-Fold Cross-Validation

**Metrik:** Macro-Dice über linke Muskeln (Klassen 1–8, gleichgewichtet)

**Erwartete Ergebnisse (Baseline ohne Augmentation):**

| | Fold 0 | Fold 1 | Fold 2 | Ø |
|---|---|---|---|---|
| Train Dice | — | — | — | — |
| Val Dice | — | — | — | — |

*[Ergebnisse werden nach Trainingsabschluss eingetragen – Training läuft auf CUDA]*

**Wo man Overfitting sieht:**  
Train Dice >> Val Dice → Modell hat die Trainingsdaten auswendig gelernt  
Bei n=38 ist leichtes Overfitting erwartet (Baseline ohne Augmentation)

**Nächste Schritte nach Baseline:**  
→ Augmentation (Flip, Rotation, Intensity-Jitter)  
→ Dropout-Tuning  
→ Learning-Rate-Schedule  
→ WandB-Sweep für Hyperparameter-Optimierung

---

## Folie 13 · Spezialitäten

**1. Bayesian Unsicherheitsquantifizierung**  
MC Dropout → Varianz-Map als zweite Ausgabe des Modells  
Pixel mit hoher Varianz = Modell ist unsicher  
→ klinisch wertvoll: Annotator kann gezielt unsichere Bereiche nachkontrollieren  
→ Standard-U-Net gibt nur eine Vorhersage ohne Konfidenz

**2. Reproduzierbare 3-Fold CV ohne Slice-Leakage**  
Splits strikt auf Fall-Ebene (Subject-Level) → kein Datenleck  
Fixer Seed 42, versioniertes `splits/folds.json`  
→ Ergebnisse sind exakt reproduzierbar

**3. WandB Experiment-Tracking**  
Jedes Experiment mit Config, Metriken, Checkpoints  
→ einfacher Vergleich verschiedener Runs  
Projekt: `dlbs-muscle-seg`, Tags: `baseline`, `bayesian-unet`, `left-only`

**4. Karpathy-Prinzip: Erst Overfit, dann Generalisierung**  
Strukturierter Ansatz: Overfit-Test → Baseline → Tuning  
→ Bugs frühzeitig erkannt, kein "Black-Box"-Training

---

## Folie 14 · Zusammenfassung

**Problemstellung:** 3D-Muskel-Segmentierung aus MRI (8 linke Oberschenkel-Muskeln)

**Kernerkenntnisse EDA:**
- Anisotrope, homogene Geometrie → Patch-Design (128×128×32)
- 77:1 Imbalance → Dice Loss + Patch-Sampling
- VL 6× grösser als BfSH → Macro-Dice gleichgewichtet

**Methodik:** Bayesian 3D U-Net mit MC Dropout (p=0.3, N=20 Samples)  
**Training:** Dice Loss, Adam 1e-4, 3-Fold CV, 80 Epochs, kein Augmentation (Baseline)

**Spezialitäten:**  
→ Unsicherheitskarten aus MC Dropout  
→ Reproduzierbare Pipeline mit WandB-Tracking

**Nächste Schritte:** Augmentation · LR-Schedule · WandB-Sweep

---

## Anhang · Häufige Fragen (Q&A Vorbereitung)

**Warum kein 2D-Ansatz?**  
3D-Kontext ist wichtig – benachbarte Schichten geben Kontext über den Verlauf des Muskels.  
Allerdings: die Anisotropie (2.0 mm Z) bedeutet, dass 3D-Patches dünner in Z-Richtung sind.

**Warum Bayesian und nicht Standard-Dropout?**  
In Standard-Training wird Dropout nur zum Training verwendet und in Inferenz deaktiviert.  
MC Dropout lässt es aktiv → N stochastische Passes geben eine Verteilung statt einem Punkt.  
→ Varianz der Prognosen = Modell-Unsicherheit → klinisch auswertbar.

**Warum Dice Loss statt Cross-Entropy?**  
Cross-Entropy gewichtet nach Anzahl Voxel → bei 77:1 Imbalance dominiert BG total.  
Dice Loss normiert per Klasse → BfSH_L (klein) wird gleich stark gelernt wie VL_L (gross).

**Warum Subject 543 ausgeschlossen?**  
Die 4 linken Quadrizeps-Muskeln (Labels 1–4) fehlen komplett.  
Würden wir Subject 543 trainieren, lernt das Modell "Quadrizeps = Hintergrund" für diesen Fall.  
→ aktiv schädlich für die Generalisierung.

**Wie viele Parameter hat das Modell?**  
Basis-Channels=16, Depth=4 → ca. 1–2 Mio. Parameter (kompaktes Netz, bewusst klein bei n=38).
