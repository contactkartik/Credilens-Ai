---
title: CrediLens AI
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 80
pinned: false
---

# CrediLens AI

AI-powered information credibility detection. Paste text or upload a
PDF/DOCX/image and the fine-tuned RoBERTa model classifies it as
credible or not, with confidence scores.

- **Frontend + backend** served by nginx in a single container
- **Model**: fine-tuned `roberta-base` baked into the image
- **Storage**: SQLite prediction history on HF persistent storage
