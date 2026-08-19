import bm25s

class IndexSearch:
    def __init__(self,retriever:bm25s.BM25=None):
        if retriever is None:
            self.retriever = bm25s.BM25()
        else:
            self.retriever = retriever
    @classmethod
    def load(cls,path) -> "IndexSearch":
        retriever = bm25s.BM25.load(path)
        return cls(retriever)
    def search(self,query:str,k:int=5):
        query_token = bm25s.tokenize([query],stopwords="en",show_progress=False)
        result = self.retriever.retrieve(query_token,k=k,show_progress=False)
        return result
    def index(self,documents):
        corpus_token = bm25s.tokenize(documents)
        self.retriever.index(corpus_token)
    def save(self,path):
        self.retriever.save(path)