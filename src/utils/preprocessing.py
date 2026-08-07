import numpy as np


def make_preprocess_fn(tokenizer, max_length: int, doc_stride: int):
    def preprocess_qa_examples(examples):
        """
        Convert raw QA examples into tokenized model features.

        The function:
        1. Cleans the questions.
        2. Tokenizes question-context pairs.
        3. Splits long contexts into overlapping chunks.
        4. Maps character-level answer positions to token positions.
        5. Uses the CLS token for chunks that do not contain the answer.
        """

        # 1. Remove leading and trailing spaces from questions
        questions = [
            str(question).strip()
            for question in examples["question"]
        ]

        # 2. Tokenize questions and contexts
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=CONFIG["max_length"],
            truncation="only_second",
            stride=CONFIG["doc_stride"],
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        # 3. Save tokenizer metadata before returning model inputs
        offset_mapping = inputs.pop("offset_mapping")

        sample_mapping = inputs.pop(
            "overflow_to_sample_mapping"
        )

        start_positions = []
        end_positions = []

        # 4. Process every generated feature/chunk
        for feature_index, offsets in enumerate(offset_mapping):

            input_ids = inputs["input_ids"][feature_index]

            # CLS is used for chunks that do not contain the answer
            cls_index = input_ids.index(
                tokenizer.cls_token_id
            )

            # None = special/padding
            # 0 = question
            # 1 = context
            sequence_ids = inputs.sequence_ids(
                feature_index
            )

            # Find the original example that produced this chunk
            sample_index = sample_mapping[feature_index]

            # Get the answer character positions
            start_char = int(
                examples["answer_start"][sample_index]
            )

            end_char = int(
                examples["answer_end"][sample_index]
            )

            # 5. Locate the first context token
            context_start = 0

            while sequence_ids[context_start] != 1:
                context_start += 1

            # Locate the final context token
            context_end = len(input_ids) - 1

            while sequence_ids[context_end] != 1:
                context_end -= 1

            # 6. Check whether the complete answer is inside this chunk
            answer_not_inside_chunk = (
                offsets[context_start][0] > start_char
                or offsets[context_end][1] < end_char
            )

            if answer_not_inside_chunk:
                start_positions.append(cls_index)
                end_positions.append(cls_index)

            else:
                # 7. Convert answer start character to start token
                start_token = context_start

                while (
                    start_token <= context_end
                    and offsets[start_token][0] <= start_char
                ):
                    start_token += 1

                start_positions.append(start_token - 1)

                # 8. Convert answer end character to end token
                end_token = context_end

                while (
                    end_token >= context_start
                    and offsets[end_token][1] >= end_char
                ):
                    end_token -= 1

                end_positions.append(end_token + 1)

        # 9. Add the labels required by the QA model
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions

        return inputs

    return preprocess_qa_examples


def predict_qa_answer(
    question: str,
    context: str,
    model,
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
    n_best: int = 20,
    max_answer_length: int = 30,
):
    import torch

    model.eval()
    question = str(question).strip()
    context = str(context)

    encoded = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    offset_mapping = encoded.pop("offset_mapping")
    encoded.pop("overflow_to_sample_mapping")

    device = next(model.parameters()).device
    inputs_on_device = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**inputs_on_device)

    start_logits = outputs.start_logits.cpu().numpy()
    end_logits = outputs.end_logits.cpu().numpy()

    best_answer = ""
    best_score = -float("inf")
    best_chunk = 0

    # هنا بالظبط: بندور جوه كل chunk (مش أول واحد بس)
    for chunk_idx in range(start_logits.shape[0]):
        offsets = offset_mapping[chunk_idx].numpy()
        sequence_ids = encoded.sequence_ids(chunk_idx)

        start_indexes = np.argsort(start_logits[chunk_idx])[-n_best:][::-1]
        end_indexes = np.argsort(end_logits[chunk_idx])[-n_best:][::-1]

        for start_index in start_indexes:
            for end_index in end_indexes:
                if (
                    start_index >= len(sequence_ids)
                    or end_index >= len(sequence_ids)
                    or sequence_ids[start_index] != 1
                    or sequence_ids[end_index] != 1
                ):
                    continue
                if end_index < start_index or (end_index - start_index + 1) > max_answer_length:
                    continue

                score = start_logits[chunk_idx][start_index] + end_logits[chunk_idx][end_index]
                # بنحدّث best_answer بس لو النتيجة دي أحسن من كل الـ chunks اللي فاتت
                if score > best_score:
                    start_char = offsets[start_index][0]
                    end_char = offsets[end_index][1]
                    best_answer = context[start_char:end_char]
                    best_score = score
                    best_chunk = chunk_idx

    return {
        "answer": best_answer,
        "score": float(best_score),
        "number_of_chunks": start_logits.shape[0],
        "chunk_index": best_chunk,
    }