# Speech Understanding — Programming Assignment 2
**Name:** Vyankatesh Deshpande
**Roll No:** [B23CS1079]

---

## 📋 Overview

This repository implements an end-to-end pipeline for processing Hinglish (code-switched Hindi–English) lectures:

```
original_segment.wav
    ├── [Task 1.1] Frame-Level LID → lid_labels.json, lid_model.pt
    ├── [Task 1.2] Constrained Decoding (Whisper + N-gram) → transcript_constrained.json
    ├── [Task 1.3] DeepFilterNet3 Denoising → denoised_segment.wav
    │
    ├── [Task 2.1] IPA Conversion (epitran + g2p_en) → transcript_ipa.json
    ├── [Task 2.2] NLLB-200 → Maithili Translation → translated_maithili.json
    │
    ├── [Task 3.1] ECAPA-TDNN Speaker Embedding → speaker_embedding.npy
    ├── [Task 3.2] pyin F0 + FastDTW Prosody Warping → warped_f0.npy, warped_energy.npy
    ├── [Task 3.3] MMS-TTS Synthesis → output_LRL_flat.wav, output_LRL_prosody.wav
    │
    ├── [Task 4.1] LFCC Anti-Spoofing Classifier → antispoof_model.pt (EER: 0.0%)
    └── [Task 4.2] FGSM Adversarial Attack → adversarial_results.json
```

---

## 📁 Repository Structure

```
Assignment2/
│   ├── task1_1.ipynb         # Frame-level LID (Wav2Vec2 + MLP)
│   ├── task1_2.ipynb         # Constrained decoding (Whisper-large-v3)
│   ├── task1_3.py          # Two-stage denoising (DeepFilterNet3)
│   ├── task2_1.ipynb         # IPA mapping (epitran + g2p_en)
│   ├── task2_2.ipynb               # Neural MT to Maithili (NLLB-200)
│   ├── task3_1.ipynb               # Speaker embedding (ECAPA-TDNN)
│   ├── task3_2.ipynb               # Prosody warping (pyin + FastDTW)
│   ├── task3_3.ipynb               # TTS synthesis (MMS-TTS)
│   ├── task4_1.ipynb               # Anti-spoofing classifier (LFCC-MLP)
│   ├── task4_2.ipynb               # FGSM adversarial attack
│   │
│   ├── output/                     # Generated outputs and artifacts
│   │   ├── lid_model.pt                # Trained LID head weights
│   │   ├── lid_labels.json             # Frame-level LID output
│   │   ├── transcript_constrained.json # Whisper-large-v3 + N-gram transcript
│   │   ├── transcript_ipa.json         # IPA-converted transcript
│   │   ├── translated_maithili.json    # NLLB-200 Maithili translation
│   │   ├── translated_maithili.txt     # Plain text translation
│   │   ├── speaker_embedding.npy       # ECAPA-TDNN x-vector (192-dim)
│   │   ├── warped_f0.npy               # DTW-warped F0 contour
│   │   ├── warped_energy.npy           # DTW-warped energy contour
│   │   ├── antispoof_model.pt          # Trained anti-spoofing MLP weights
│   │   ├── antispoof_results.json      # EER: 0.0%
│   │   ├── adversarial_results.json    # FGSM attack results
│   │   ├── output_LRL_flat.wav         # Maithili TTS (no prosody)
│   │   ├── output_LRL_prosody.wav      # Maithili TTS (with prosody warp)
│   │   ├── output_LRL_cloned.wav       # Final cloned output
│   │   ├── denoise_comparison.png      # Denoising before/after
│   │   ├── prosody_comparison.png      # F0/energy contour comparison
│   │   ├── antispoof_det.png           # DET curve
|   |   ├── parallel_corpus_maithili.json   # 500-word EN/HI → Maithili corpus
│   │   └── adversarial_analysis.png    # FGSM attack analysis
│
├── report.pdf                      # IEEE two-column LaTeX report
├── Implementation_note.pdf         # 1-page implementation notes
└── README.md                       
```

**Audio Manifest:**
- `original_segment.wav` — source 10-minute lecture snippet
- `student_voice_ref.wav` — 60s student voice reference recording
- `output_LRL_cloned.wav` — final ~9-minute Maithili synthesized lecture

---

### Execution

Run notebooks, uploading the outputs from each step:

| Step | Notebook | Input Files | Output Files |
|------|----------|-------------|--------------|
| 1 | `task1_1.ipynb` | `original_segment.wav` | `lid_model.pt`, `lid_labels.json` |
| 2 | `task1_2.ipynb` | `original_segment.wav` | `transcript_constrained.json` |
| 3 | `task1_3_denoise.py` | `original_segment.wav` | `denoised_segment.wav` |
| 4 | `task2_1.ipynb` | `transcript_constrained.json`, `lid_labels.json` | `transcript_ipa.json` |
| 5 | `task2_2.ipynb` | `transcript_constrained.json`, `lid_labels.json`, `parallel_corpus_maithili.json` | `translated_maithili.json` |
| 6 | `task3_1.ipynb` | `Student_voice_ref.wav` | `speaker_embedding.npy` |
| 7 | `task3_3.ipynb` | `translated_maithili.json` | `output_LRL_flat.wav` |
| 8 | `task3_2.ipynb` | `original_segment.wav`, `output_LRL_flat.wav` | `warped_f0.npy`, `warped_energy.npy`, `output_LRL_prosody.wav` |
| 9 | `task4_1.ipynb` | `Student_voice_ref.wav`, `original_segment.wav`, `output_LRL_cloned.wav` | `antispoof_model.pt`, `antispoof_results.json` |
| 10 | `task4_2.ipynb` | `original_segment.wav`, `lid_model.pt` | `adversarial_results.json` |

---

## 📊 Key Results

### Task 1.1 — LID
- **F1-Score: 0.999** (macro, binarized English/Hindi — val set epoch 10)
- Frame resolution: 20ms (Wav2Vec2 output stride)
- Switch latency: ≤ 20ms (one frame)
- 53 language segments detected (post median-filter + transition penalty smoothing)

### Task 1.2 — Constrained Decoding
- Whisper-large-v3 + trigram N-gram logit bias (λ=2.5)
- N-gram LM: vocab=265, unique contexts=1,180
- 24 transcript segments produced

### Task 1.3 — Denoising
- DeepFilterNet3 + Spectral Gating two-stage pipeline
- SNR improvement: 0.06 dB (input was already relatively clean)

### Task 3.2/3.3 — Voice Cloning Ablation
| Config | MCD (dB) |
|--------|----------|
| Flat TTS (no prosody) | 9.2 |
| + Prosody Warping | **7.1** ✅ |

### Task 4.1 — Anti-Spoofing
- **EER: 0.0%** (perfect bona fide / spoof separation)
- Decision threshold: 0.9951
- Features: LFCC + Δ + ΔΔ (240-dim), 3-layer BN-MLP
- Data: 659 bona fide + 539 spoof segments (train=958, test=240)
- Test split: 122 bona fide + 118 spoof samples
- Note: 0% EER reflects large acoustic domain mismatch between classroom recording and clean TTS output

### Task 4.2 — Adversarial Robustness
- Attack: FGSM on Wav2Vec2 LID system
- Result: **Could not flip in range ε ∈ [0.0001, 0.05]**
- Analysis: Wav2Vec2 Transformer's global self-attention is highly robust to single-step FGSM; perturbation at ε=0.05 reduces SNR to -8.1 dB (louder than speech) but still fails to flip prediction
- This demonstrates strong adversarial robustness of the Transformer-based LID system

---

## 📚 References

1. Baevski et al., "wav2vec 2.0", NeurIPS 2020
2. Costa-jussà et al., "NLLB-200", arXiv:2207.04672, 2022
3. Desplanques et al., "ECAPA-TDNN", Interspeech 2020
4. Pratap et al., "MMS", arXiv:2305.13516, 2023
5. Goodfellow et al., "FGSM", ICLR 2015
6. Salvador & Chan, "FastDTW", Intelligent Data Analysis 2007
7. Mauch & Dixon, "pYIN", ICASSP 2014

---
