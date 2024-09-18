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
        model_id: str = "all-mpnet-base-v2",
        trust_remote_code: bool = False,
        config_kwargs: dict | None = None,
        **kwargs
    ):

        super().__init__(field, **kwargs)

        model = sentence_transformers.SentenceTransformer(
            model_id, trust_remote_code=trust_remote_code, config_kwargs=config_kwargs
        )

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
