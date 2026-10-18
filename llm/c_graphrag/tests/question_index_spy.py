"""Spy on faiss.read_index for the question index (question_embeddings_ivf.faiss,
~4.15 GB read fully into RAM): counts reads per process and fails on the
second one, so an integration session can never hold two copies."""

QUESTION_INDEX_NAME = "question_embeddings_ivf.faiss"


class QuestionIndexReadSpy:
    def __init__(self, faiss_module):
        self.faiss = faiss_module
        self.original = faiss_module.read_index
        self.reads: list[str] = []

    def __call__(self, path, *args, **kwargs):
        if str(path).endswith(QUESTION_INDEX_NAME):
            if self.reads:
                raise AssertionError(f"question index read a second time in this process: {path} "
                                     f"(first: {self.reads[0]}) -- use the session fixture `question_faiss`")
            self.reads.append(str(path))
        return self.original(path, *args, **kwargs)

    def install(self):
        self.faiss.read_index = self
        return self

    def uninstall(self):
        self.faiss.read_index = self.original
