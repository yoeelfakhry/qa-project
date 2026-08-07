"""
Data loading, cleaning, and splitting for the QA project.
"""
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd


def load_raw_data(csv_path: str, n_rows=None) -> pd.DataFrame:
    """Load the raw SQuAD-style CSV.

    n_rows=None loads the FULL file. Pass a number (e.g. 10_000)
    only when you deliberately want a smaller subset (like our
    model-comparison phase).
    """
    if n_rows is None:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, nrows=n_rows)
    return df.copy()


def validate_answer_spans(data: pd.DataFrame) -> pd.DataFrame:
    """Sanity check: answer_start/answer_end must actually point at
    the answer text inside the context. Adds diagnostic columns.
    """
    data = data.reset_index(drop=True)
    data["answer_start"] = data["answer_start"].astype(int)
    data["answer_end"] = data["answer_end"].astype(int)

    data["extracted_answer"] = data.apply(
        lambda row: row["context"][row["answer_start"]:row["answer_end"]],
        axis=1,
    )
    data["span_match"] = data["extracted_answer"] == data["answer"]

    n_wrong = (~data["span_match"]).sum()
    if n_wrong > 0:
        print(f"WARNING: {n_wrong} rows have misaligned answer spans.")


    return data

def check_leakage(train_data: pd.DataFrame, valid_data: pd.DataFrame) -> int:
    """Confirm no (question, context) pair appears in both splits."""
    train_pairs = set(zip(train_data["question"], train_data["context"]))
    valid_pairs = set(zip(valid_data["question"], valid_data["context"]))
    overlap = len(train_pairs & valid_pairs)
    print(f"Train/valid question-context overlap: {overlap}")
    return overlap


def load_and_prepare(csv_path: str, seed: int, n_rows=None):
    """One-call convenience wrapper: raw CSV -> clean, split, validated
    train/valid/test dataframes."""
    data = load_raw_data(csv_path, n_rows=n_rows)
    data = validate_answer_spans(data)
    data = deduplicate(data)
    train_data, valid_data, test_data = split_data(data, seed=seed)
    check_leakage(train_data, valid_data)
    return train_data, valid_data, test_data


DUPLICATE_SUBSET_COLUMNS = [
    "context",
    "question",
    "answer",
    "answer_start",
    "answer_end",
]


def deduplicate(data: pd.DataFrame) -> pd.DataFrame:
    rows_before = len(data)
    data = data.drop_duplicates(
        subset=DUPLICATE_SUBSET_COLUMNS
    ).reset_index(drop=True)
    rows_after = len(data)

    print(f"Rows before: {rows_before}")
    print(f"Rows after:  {rows_after}")
    print(f"Removed:     {rows_before - rows_after}")

    return data


def split_data(data: pd.DataFrame, seed: int, test_size: float = 0.20):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, temp_idx = next(splitter.split(data, groups=data["context"]))

    train_data = data.iloc[train_idx].reset_index(drop=True)
    temp_data = data.iloc[temp_idx].reset_index(drop=True)

    valid_test_splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    valid_idx, test_idx = next(valid_test_splitter.split(temp_data, groups=temp_data["context"]))

    valid_data = temp_data.iloc[valid_idx].reset_index(drop=True)
    test_data = temp_data.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {len(train_data)}  Valid: {len(valid_data)}  Test: {len(test_data)}")
    return train_data, valid_data, test_data


if __name__ == "__main__":
    train_df, valid_df, test_df = load_and_prepare( "data/SQuAD-v1.1.csv", seed=42, n_rows=1000)
    print(train_df.head())