from typing import Iterable

import dedupe.predicates
import numpy
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

        self.embeddings = model.encode(corpus, show_progress_bar=True)
        self.str_to_i = {string: i for i, string in enumerate(corpus)}

        self.cosine = (
            sentence_transformers.SimilarityFunction.to_similarity_pairwise_fn("cosine")
        )

    def comparator(self, field_a, field_b) -> float:

        if field_a is None or field_b is None:
            return None

        result = self.cosine(
            self.embeddings[self.str_to_i[field_a]][None, :],
            self.embeddings[self.str_to_i[field_b]][None, :],
        )[0].item()

        return result
