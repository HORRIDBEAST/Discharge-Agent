"""
Generate synthetic patient source note PDFs for testing.

Creates two patients:
  Patient 001: Standard case with pending labs
  Patient 002: Complex case with conflicts, med changes, missing data

Uses reportlab-free approach — writes PDFs via fpdf2 if available,
otherwise creates plain text files as fallback for environments
without PDF generation libraries.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


PATIENT_001_DOCS = {
    "admission_note.pdf": """
ADMISSION NOTE

Patient: John Smith
MRN: 001-234-567
Date of Birth: 1955-03-14
Sex: Male
Age: 69

Admission Date: 2024-11-01
Admitting Physician: Dr. A. Johnson

Chief Complaint:
Chest pain and shortness of breath for 2 days.

History of Present Illness:
Mr. John Smith is a 69-year-old male with a known history of type 2 diabetes mellitus,
hypertension, and hyperlipidaemia who presented to the emergency department on 2024-11-01
with a 2-day history of worsening chest pain radiating to the left arm, associated with
shortness of breath and diaphoresis. He denies fever, cough, or lower limb swelling.

Past Medical History:
1. Type 2 diabetes mellitus (on metformin)
2. Hypertension (on lisinopril 10mg daily)
3. Hyperlipidaemia (on atorvastatin 40mg nightly)

Medications on Admission:
1. Metformin 1000mg twice daily
2. Lisinopril 10mg daily
3. Atorvastatin 40mg nightly
4. Aspirin 81mg daily

Allergies: Penicillin (anaphylaxis), Sulfa (rash)

Physical Examination:
BP: 158/94 mmHg  HR: 96 bpm  RR: 18  SpO2: 97% on room air
Mild diaphoresis noted. JVP not elevated. S1S2 heard. No murmurs.

Initial Assessment:
Likely acute coronary syndrome. Rule out NSTEMI.

Initial Plan:
1. ECG - done, showing ST changes in leads II, III, aVF
2. Troponin - sent, pending
3. Chest X-ray - ordered
4. Cardiology consult - requested
5. Continue aspirin, hold metformin pending contrast study
""",

    "progress_note_day2.pdf": """
PROGRESS NOTE — Day 2

Date: 2024-11-02
Patient: John Smith  MRN: 001-234-567
Attending: Dr. A. Johnson

Subjective:
Patient reports improved chest pain. Still mild shortness of breath. Tolerating diet.

Objective:
BP: 142/86  HR: 78  RR: 16  SpO2: 98% on 2L nasal cannula
Afebrile.

Troponin peak: 2.8 ng/mL (elevated) — confirmed NSTEMI.

Assessment and Plan:
1. NSTEMI — confirmed. Patient underwent coronary angiography this morning.
   Findings: 85% stenosis of the right coronary artery (RCA).
   Procedure: Percutaneous coronary intervention (PCI) with drug-eluting stent to RCA.
   Post-procedure: stable, no complications.

2. Type 2 diabetes mellitus — metformin held peri-procedure.
   Blood glucose monitoring ongoing. Will restart metformin post-discharge.

3. Hypertension — lisinopril continued.

4. Medications:
   Added: Clopidogrel 75mg daily (dual antiplatelet post-PCI)
   Added: Bisoprolol 2.5mg daily (heart rate control)
   Changed: Atorvastatin increased to 80mg nightly (high-intensity statin post-ACS)
   Aspirin 81mg daily — continued

Labs pending: Blood culture (sent due to fever spike — preliminary, awaiting sensitivity)
Echocardiogram: ordered, not yet resulted.
""",

    "lab_results.pdf": """
LABORATORY RESULTS

Patient: John Smith  MRN: 001-234-567
Date: 2024-11-02

Haematology:
WBC: 11.2 x10^9/L (H)
Haemoglobin: 13.4 g/dL
Platelets: 198 x10^9/L

Biochemistry:
Sodium: 138 mmol/L
Potassium: 4.1 mmol/L
Creatinine: 94 umol/L
eGFR: 72 mL/min/1.73m2
Glucose: 9.8 mmol/L (H)
HbA1c: 8.2% (H)

Cardiac:
Troponin T: 2.8 ng/mL (H) — PEAK
CK-MB: 45 U/L (H)
BNP: 280 pg/mL (H)

Pending:
- Blood culture: PENDING — sent 2024-11-02, awaiting final results
- Echocardiogram: PENDING — ordered, not yet resulted
""",

    "discharge_medications.pdf": """
DISCHARGE MEDICATION RECORD

Patient: John Smith  MRN: 001-234-567
Discharge Date: 2024-11-05

Discharge Medications:
1. Aspirin 81mg daily — continue (antiplatelet)
2. Clopidogrel 75mg daily — continue for 12 months (dual antiplatelet therapy post-PCI)
3. Bisoprolol 2.5mg daily — continue (beta-blocker post-NSTEMI)
4. Lisinopril 10mg daily — continue (ACE inhibitor, hypertension)
5. Atorvastatin 80mg nightly — high-intensity statin (increased from 40mg)
6. Metformin 1000mg twice daily — restart (was held peri-procedure)

Discharge Condition: Stable

Discharge Date: 2024-11-05

Follow-up Instructions:
1. Cardiology clinic follow-up in 2 weeks (2024-11-19)
2. GP follow-up in 1 week for wound check and blood pressure review
3. Do NOT stop aspirin or clopidogrel without consulting cardiologist
4. Echocardiogram result to be followed up at cardiology clinic
5. Blood culture result to be reviewed by GP — notify patient if any action required
6. Diabetic review with endocrinologist — referral sent
"""
}

PATIENT_002_DOCS = {
    "admission_note.pdf": """
ADMISSION NOTE

Patient: Margaret Chen
MRN: 002-456-789
Date of Birth: 1948-07-22
Sex: Female
Age: 76

Admission Date: 2024-11-03
Admitting Physician: Dr. B. Williams

Chief Complaint:
Confusion and falls at home.

History of Present Illness:
Mrs. Margaret Chen is a 76-year-old female brought in by family due to 3 days of progressive
confusion, reduced oral intake, and two falls at home. No head injury documented by family.
Background of hypertension, atrial fibrillation on warfarin, and mild cognitive impairment.

Past Medical History:
1. Atrial fibrillation — on warfarin
2. Hypertension
3. Mild cognitive impairment
4. Osteoporosis

Medications on Admission:
1. Warfarin 5mg daily (INR target 2.0-3.0)
2. Amlodipine 5mg daily
3. Donepezil 10mg nightly
4. Calcium + Vitamin D supplement daily
5. Ibuprofen 400mg PRN (prescribed by GP for knee pain)

Allergies: No known drug allergies (NKDA)

Assessment:
1. Delirium — likely precipitated by urinary tract infection vs medication effect
2. Atrial fibrillation — rate controlled
3. Supratherapeutic INR: INR 4.8 on admission (TARGET: 2.0-3.0) — warfarin held

Note: Ibuprofen + warfarin combination is a MAJOR drug interaction risk — ibuprofen
discontinued on admission.
""",

    "progress_note_1.pdf": """
PROGRESS NOTE — Day 2

Date: 2024-11-04
Patient: Margaret Chen  MRN: 002-456-789

Subjective: Family reports patient slightly more alert. Eating better.

Objective:
BP: 148/84  HR: 74  Temp: 37.8°C
Urine culture: PENDING — sent yesterday

Assessment:
1. Delirium — improving. Likely UTI as precipitant (dysuria reported by family).
   Commenced empirical trimethoprim 200mg twice daily for UTI pending culture.

2. Supratherapeutic INR — INR now 3.9 (down from 4.8). Warfarin held.
   Will restart at reduced dose when INR in range.

3. Atrial fibrillation — rate controlled, no cardioversion planned.

4. Falls risk — physiotherapy review requested.
""",

    "progress_note_2.pdf": """
PROGRESS NOTE — Day 4

Date: 2024-11-06
Patient: Margaret Chen  MRN: 002-456-789
Author: Dr. C. Patel (Covering)

NOTE: This is a covering physician's note.

Assessment:
1. Primary diagnosis: Community-acquired pneumonia — patient appears to have respiratory
   symptoms. Started on IV amoxicillin-clavulanate.

Note: [This conflicts with Dr. Williams' assessment of UTI as primary diagnosis]

2. INR 2.6 — back in range. Restarting warfarin at 3mg daily (reduced from 5mg).
   Reason: INR was supratherapeutic, dose reduction appropriate.

3. Urine culture: PENDING — still awaiting result.

4. Blood culture: PENDING — sent on Day 2 due to persistent low-grade fever.
""",

    "discharge_medications.pdf": """
DISCHARGE MEDICATION RECORD

Patient: Margaret Chen  MRN: 002-456-789
Discharge Date: 2024-11-08

Discharge Medications:
1. Warfarin 3mg daily (REDUCED from 5mg — dose adjusted due to supratherapeutic INR)
2. Amlodipine 5mg daily — continue
3. Donepezil 10mg nightly — continue
4. Calcium + Vitamin D supplement daily — continue
5. Trimethoprim 200mg twice daily — complete 7-day course (for UTI)

NOTE: Ibuprofen DISCONTINUED — do not restart. Drug interaction with warfarin.

Discharge Condition: Improved

Follow-up Instructions:
1. INR check with GP in 3 days (2024-11-11) — warfarin dose may need further adjustment
2. Urine culture result to be reviewed by GP — action if sensitivity changes antibiotic
3. Blood culture result pending — review with GP
4. Cardiology follow-up for atrial fibrillation in 4 weeks
5. Falls assessment at home — community OT referral sent
6. Cognitive review — memory clinic referral submitted

Discharge Date: 2024-11-08
"""
}


def write_text_as_pdf(content: str, filepath: Path) -> None:
    """
    Write content as a PDF using PyMuPDF (fitz).
    Falls back to .txt if PDF generation fails.
    """
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        # Insert text with a standard font
        font_size = 10
        margin = 50
        y = margin
        for line in content.strip().split("\n"):
            if y > page.rect.height - margin:
                page = doc.new_page()
                y = margin
            page.insert_text((margin, y), line, fontsize=font_size)
            y += font_size + 2
        doc.save(str(filepath))
        doc.close()
        print(f"  Created: {filepath}")
    except Exception as e:
        # Fallback: write as txt (parser will OCR-stub it)
        txt_path = filepath.with_suffix(".txt")
        txt_path.write_text(content.strip())
        print(f"  Created (txt fallback): {txt_path} — PDF creation failed: {e}")


def generate_all():
    base = Path(__file__).parent.parent / "sample_data"

    print("Generating Patient 001 — NSTEMI with pending labs...")
    p1_dir = base / "patient_001"
    p1_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in PATIENT_001_DOCS.items():
        write_text_as_pdf(content, p1_dir / filename)

    print("\nGenerating Patient 002 — Delirium with conflict and missing data...")
    p2_dir = base / "patient_002"
    p2_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in PATIENT_002_DOCS.items():
        write_text_as_pdf(content, p2_dir / filename)

    print("\nDone. Sample data written to:", base)


if __name__ == "__main__":
    generate_all()
