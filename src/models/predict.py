from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from src.utils.preprocessing import predict_qa_answer


class QAModel:
    def __init__(self, model_dir: str, max_length=384, doc_stride=128,
                 n_best=20, max_answer_length=30):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.model.eval()

        self.max_length = max_length
        self.doc_stride = doc_stride
        self.n_best = n_best
        self.max_answer_length = max_answer_length

    def answer(self, question: str, context: str) -> dict:
        return predict_qa_answer(
            question=question,
            context=context,
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            doc_stride=self.doc_stride,
            n_best=self.n_best,
            max_answer_length=self.max_answer_length,)

