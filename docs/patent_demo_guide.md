# eTBF AI Diagnostics System - Patent Demonstration Guide

## Overview

This document provides patent examiners with a comprehensive guide to understanding and replicating the **Enhanced Total Biological Fingerprint (eTBF) AI Diagnostics System** demonstration.

---

## What the System Does

The eTBF system is a multimodal AI platform that:

1. **Ingests** 5 types of biological data:
   - Radiology (MRI, CT, X-ray)
   - Ultrasound
   - Electrophoresis (protein/DNA/lipoprotein gels)
   - Microscopy (Whole Slide Images)
   - Laboratory (CBC, BMP, CMP, ABG)

2. **Processes** each modality through specialized AI encoders

3. **Retrieves** similar historical cases from a vector database

4. **Fuses** all information to generate a diagnosis

5. **Explains** the diagnosis with visual heatmaps, feature importance, and reasoning traces

---

## How to View the Demo

### Option 1: Browse the GitHub Repository

1. Go to: https://github.com/Tank0823/etbf-ai-diagnostics
2. Navigate to the `examples/` folder
3. Each case folder contains:
   - `README.md` - Case description
   - `inputs/` - Sample images and lab data
   - `outputs/` - Pre-generated diagnosis results

### Option 2: Run the Interactive Notebook

1. Clone the repository
2. Start the eTBF server
3. Open `notebooks/demo.ipynb` in Jupyter
4. Run all cells to see live diagnoses

### Option 3: Use the API Directly

```bash
curl -X POST "http://localhost:8000/api/demo/diagnose?patient_id=John%20Doe"
