# eTBF AI Diagnostics System - Patent Examples

This folder contains real-world demonstration examples for the **Enhanced Total Biological Fingerprint (eTBF) AI Diagnostics System**.

## Purpose

These examples serve as **proof of work** for patent examiners to understand how the system processes multimodal medical data and generates diagnostic outputs.

## How to Use This

1. **Browse the case folders** to see example inputs and outputs.
2. **Run the demo notebook** (`notebooks/demo.ipynb`) to interact with the system.
3. **Review the pre-generated results** in each case's `outputs/` folder.

## Cases Included

| Case | Description | Modalities | Expected Diagnosis |
|------|-------------|------------|-------------------|
| **Case 1** | Aneurysmal Bone Cyst | X-ray, CT, Laboratory | Aneurysmal Bone Cyst |
| **Case 2** | Pulmonary Nodule | CT, Laboratory | Pulmonary Nodule |
| **Case 3** | Multiple Myeloma | Electrophoresis, X-ray, Laboratory | Multiple Myeloma |
| **Case 4** | Acute Gouty Arthritis | X-ray, Ultrasound, Microbiology, Laboratory | Acute Gout |
| **Case 5** | Papillary Thyroid Carcinoma | Ultrasound, Histopathology, Laboratory | Papillary Thyroid Carcinoma |
| **Case 6** | Solitary Fibrous Tumor | MRI, Histopathology, Laboratory | Solitary Fibrous Tumor |

---

## Modalities Demonstrated

| Modality | Cases | Description |
|----------|-------|-------------|
| Radiology (X-ray) | 1, 3, 4 | Bone lesions, joint effusion |
| Radiology (CT) | 1, 2 | Soft tissue and lung nodules |
| Radiology (MRI) | 6 | Soft tissue mass characterization |
| Ultrasound | 4, 5 | Joint and thyroid evaluation |
| Electrophoresis | 3 | Monoclonal protein detection |
| Histopathology | 5, 6 | Microscopy reports with IHC |
| Microbiology | 4 | Joint fluid analysis, crystals |
| Laboratory | All | CBC, BMP, thyroid, inflammatory markers |

---

## Data Sources

All example data is sourced from **free, open-access medical databases**:

- **Radiopaedia.org** - Radiology cases with multimodal imaging
- **Open-i (NIH)** - Biomedical images and case reports

Each case includes an `ATTRIBUTION.txt` file with proper source citation.

---

## License

All examples are provided under CC-NC-BY-SA 3.0 license terms as per Radiopaedia requirements.

---

## Running the Demo

### Prerequisites

1. Python 3.10+
2. eTBF server running on `http://localhost:8000`
3. Jupyter Notebook installed

### Steps

```bash
# 1. Install Jupyter if not already installed
pip install notebook

# 2. Start the eTBF server
cd ~/Desktop/etbf-ai-diagnostics
venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 3. In a new terminal, launch the notebook
jupyter notebook notebooks/demo.ipynb
