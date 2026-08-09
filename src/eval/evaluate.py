import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.models.predict import QAModel
from src.utils.metrics import exact_match_score, f1_score


def evaluate_qa_dataset(data: pd.DataFrame, qa_model: QAModel, model_name: str):
    rows = []
    em_scores, f1_scores = [], []

    for _, example in tqdm(data.iterrows(), total=len(data), desc=f"Evaluating {model_name}"):
        pred = qa_model.answer(
            question=example["question"],
            context=example["context"],
        )
        em = exact_match_score(pred["answer"], example["answer"])
        f1 = f1_score(pred["answer"], example["answer"])
        em_scores.append(em)
        f1_scores.append(f1)

        rows.append({
            "question": example["question"],
            "true_answer": example["answer"],
            "predicted_answer": pred["answer"],
            "exact_match": em,
            "f1": f1,
        })

    results = {
        "model": model_name,
        "number_of_examples": len(data),
        "exact_match": 100 * sum(em_scores) / len(em_scores),
        "f1": 100 * sum(f1_scores) / len(f1_scores),
    }
    return results, pd.DataFrame(rows)


def main(model_dir: str):
    model_dir = Path(model_dir)

    manifest = json.loads((model_dir / "run_manifest.json").read_text())
    model_name = manifest["model_name"]

    qa_model = QAModel(str(model_dir))

    valid_df = pd.read_csv(model_dir / "valid_split.csv")

    results, predictions_df = evaluate_qa_dataset(valid_df, qa_model, model_name)

    print(json.dumps(results, indent=2))

    predictions_df.to_csv(model_dir / "validation_predictions.csv", index=False)

    errors_df = (
        predictions_df[predictions_df["exact_match"] == 0]
        .sort_values("f1")
        .reset_index(drop=True)
    )
    errors_df.to_csv(model_dir / "validation_errors.csv", index=False)
    print(f"\n{len(errors_df)} non-exact predictions saved to validation_errors.csv")

    with open(model_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    args = parser.parse_args()
    main(args.model_dir)