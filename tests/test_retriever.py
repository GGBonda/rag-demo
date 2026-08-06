import json
from dataclasses import asdict

from realtime_response.retriever import Retriever


def test_search() -> None:
    results = Retriever().search(query="区域店项目的背景是什么")

    print("="*600)
    print([hit.id for hit in results])

if __name__ == "__main__":
    test_search()
