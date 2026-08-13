# Extractive Question Answering — RoBERTa Fine-tuned on SQuAD v1.1

An end-to-end NLP pipeline that fine-tunes transformer models for extractive
question answering, runs a controlled comparison across three architectures,
and serves the best model through an interactive web app.

**Live demo:** run locally with `streamlit run app.py`
**Model on Hugging Face:** [yoeel/roberta-qa-squad](https://huggingface.co/yoeel/roberta-qa-squad)


## Results

Three transformer architectures were fine-tuned and compared under
**identical conditions** — same data split, same hyperparameters, same
seed — on a fixed 10,000-example subset, to select the best-performing
model before committing to full-scale training.

| Model | Parameters | Exact Match | F1 Score |
|---|---|---|---|
| ALBERT-base-v2 | 11.1M | 72.22% | 83.37% |
| BERT-base-uncased | 108.9M | 67.54% | 78.70% |
| **RoBERTa-base** | 124.1M | **74.74%** | **84.63%** |

**RoBERTa-base was selected** as the production model and retrained on the
full SQuAD v1.1 dataset (~87K examples):

| Metric | Score |
|---|---|
| Exact Match | 72.38% |
| F1 Score | **85.26%** |
| Validation examples | 8,654 |

*Note: ALBERT reaches within ~1.3 F1 points of RoBERTa using 11x fewer
parameters — a notable efficiency result on its own.*

## Why a pipeline, not just notebooks

This project started as three separate notebooks (one per model). Copy-pasted
preprocessing logic silently drifted between them — inconsistent
deduplication, a mismatched learning rate — which made the model comparison
unreliable without it being obvious from any single notebook. The project was
refactored into a single, config-driven pipeline: one training script, driven
by per-model YAML configs, sharing the same data/preprocessing/evaluation
code. This removes that entire class of bug structurally, not just for this
comparison but for any future one.

## Project structure

├── README.md
├── requirements.txt
├── configs/ # one YAML per training run
│ ├── albert.yaml
│ ├── bert.yaml
│ └── roberta.yaml # production config (full data)
├── data/ # gitignored; see setup below
├── src/
│ ├── utils/
│ │ ├── data_utils.py # load / dedup / split (shared, identical
│ │ │ logic across all models)
│ │ ├── metrics.py # Exact Match / F1
│ │ └── preprocessing.py # tokenization + answer-span decoding
│ ├── models/
│ │ ├── train.py # single script trains any model via --config
│ │ └── predict.py # load a saved model, answer questions
│ └── eval/
│ └── evaluate.py # EM/F1 + error analysis on a saved model
├── saved_models/ # gitignored; trained checkpoints land here
└── app.py # Streamlit demo UI

## Setup

**1. Get the data.** This project uses SQuAD v1.1 via the Kaggle dataset
[`akashdesarda/squad-v11`](https://www.kaggle.com/datasets/akashdesarda/squad-v11):

\`\`\`python
import kagglehub
path = kagglehub.dataset_download("akashdesarda/squad-v11")
\`\`\`
Copy the resulting CSV into `data/SQuAD-v1.1.csv`.

**2. Install dependencies:**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

**3. Train** (GPU required — a free Colab or Kaggle GPU is sufficient):
\`\`\`bash
python -m src.models.train --config configs/roberta.yaml
\`\`\`

**4. Evaluate:**
\`\`\`bash
python -m src.eval.evaluate --model_dir saved_models/roberta_qa_full_best
\`\`\`

**5. Run the demo app:**
\`\`\`bash
streamlit run app.py
\`\`\`
By default the app loads the model from the Hugging Face Hub
(`yoeel/roberta-qa-squad`), so no local training is required just to try it.

## Technical details

- **Base checkpoint:** `FacebookAI/roberta-base`
- **Preprocessing:** overlapping chunks for long contexts (`max_length=384`,
  `doc_stride=128`), with answer spans mapped to token positions
- **Training:** 3 epochs, batch size 8, learning rate 2e-5, linear warmup
- **Split:** context-grouped train/valid/test (80/10/10) to prevent the same
  passage appearing in both train and validation
- **Checkpointing:** best model selected by validation loss; automatically
  pushed to the Hugging Face Hub after every epoch during training, so a
  disconnected training session never loses more than one epoch of progress

## Limitations

- The model is purely **extractive** — it returns a span from the given
  passage and cannot answer questions the passage doesn't cover, nor
  synthesize information across multiple passages.
- Trained and evaluated on SQuAD v1.1, which does not include unanswerable
  questions (unlike SQuAD 2.0) — the model has not been trained to recognize
  when no answer exists in the passage.
