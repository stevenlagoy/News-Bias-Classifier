# Article-Level Political Partisanship Classification

Classifier for bias in news articles, determining partisan leaning for individual documents.

CS 567: Machine Learning

Fall 2026

## Team

- Steven LaGoy

- Afeef Seyed

## Background and Motivation

Existing tools for characterizing news bias (AllSides, Ad Fontes Media, Media Bias/Fact Check) operate primarily at the publisher level, even though a single outlet's articles can vary substantially in partisan framing by topic and author. Article-level bias detection has direct applications in media literacy tooling and journalism studies, and it connects to an established NLP literature (Baly et al., EMNLP 2020) that supplies both labeled data and a released baseline to compare against, giving the project a rigorous evaluation target rather than an ungrounded from-scratch effort.

## Problem / Question

Can we predict the political leaning (left / center / right) of an individual news article from its text, in a way that captures specific ideological framing rather than memorizing outlet-specific writing style? We will evaluate this distinction by comparing performance on a random train/test split against a publisher-disjoint split. The core scope is the partisanship axis; the model architecture will be designed so that other axes (e.g., factual reliability, economic vs. social framing) could be added later without redesigning the system, though training and evaluating those additional axes is out of scope for this project.

## Datasets

- **Primary:** Baly et al., Article-Bias-Prediction (github.com/ramybaly/Article-Bias-Prediction)
    - ~37,500 AllSides-sourced articles
    - Labeled left/center/right, both random and publisher-disjoint (“media-based”) splits
    - Directly supports the outlet-leakage comparison above. 
- **Secondary (generalization check):** Qbias
    - ~21,700 AllSides headline-roundup articles
    - Nov. 2022, out-of-distribution/temporal test set
- **Related resource, not core to this project:** ramybaly/News-Media-Reliability
    - Companion source-level reliability/bias dataset
    - Natural future axis under the extensible architecture
    - Not part of this project's training or evaluation data

## Initial Approach

- **Baseline:** TF-IDF features with a linear classifier (logistic regression / linear SVM), evaluated on the publisher-disjoint split.
- **Advanced model:** a fine-tuned transformer (e.g., RoBERTa-base) on article text.
- **Design constraint:** a classification head structured to support per-axis outputs, so a future axis could be added without retraining the base encoder.
- **Stretch goal, time permitting:** conditional generation of article text targeted at a specified partisan lean, using the same ground-truth labels.

## Initial Project Plan

- **Weeks 1–3:** Dataset acquisition, exploratory analysis, related-work review.
- **Weeks 4–6:** Preprocessing pipeline (boilerplate/outlet-identity stripping, split verification); baseline model.
- **Weeks 7–9:** Transformer fine-tuning on the publisher-disjoint split; baseline vs. advanced comparison.
- **Weeks 10–11:** Error analysis (topic-stratified), out-of-distribution check on Qbias.
- **Weeks 12–14:** Stretch goal — generation prototype, if core results are on track.
- **Weeks 15–16:** Final report and presentation preparation.

*This plan is tentative and may be adjusted as the project progresses.*
