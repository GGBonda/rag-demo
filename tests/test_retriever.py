import json
from unittest import TestCase

from realtime_response.retriever import Retriever


def test_search() -> None:
    results = Retriever().search(
        query="产品实例id的含义",
        top_k=5,
        similarity_threshold=0.1
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    test_search()
