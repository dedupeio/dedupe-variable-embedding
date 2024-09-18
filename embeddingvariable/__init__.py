from typing import Iterable

import dedupe.predicates
import sentence_transformers
from dedupe.variables.string import BaseStringType


class Embedding(BaseStringType):
    type = "Embedding"

    _index_predicates = [
        dedupe.predicates.TfidfNGramCanopyPredicate,
        dedupe.predicates.TfidfNGramSearchPredicate,
        dedupe.predicates.TfidfTextCanopyPredicate,
        dedupe.predicates.TfidfTextSearchPredicate,
    ]

    def __init__(
        self,
        field: str,
        corpus: Iterable[str],
        model: sentence_transformers.SentenceTransformer | None = None,
        **kwargs
    ):

        super().__init__(field, **kwargs)

        if model is None:
            model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")

        self.embeddings = {
            string: embedding.reshape(1, -1)
            for string, embedding in zip(
                corpus, model.encode(corpus, show_progress_bar=True)
            )
        }

        self.cosine = (
            sentence_transformers.SimilarityFunction.to_similarity_pairwise_fn("cosine")
        )

    def comparator(self, field_a, field_b) -> float:

        if field_a is None or field_b is None:
            return None

        return self.cosine(self.embeddings[field_a], self.embeddings[field_b])[0].item()
