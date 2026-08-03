import unittest
import warnings
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from store import (
    MARKDOWN_DOCUMENT_COLLECTION,
    MarkdownDocumentStore,
    PARAGRAPH_CHUNK_COLLECTION,
    PARAGRAPH_DENSE_VECTOR,
    PARAGRAPH_TEXT_SPARSE_VECTOR,
    ParagraphChunkStore,
    RETRIEVE_CHUNK_COLLECTION,
    QdrantStore,
    RetrieveChunkStore,
)


class QdrantStoreTest(unittest.TestCase):
    @staticmethod
    def _create_store(client: QdrantClient) -> QdrantStore:
        store = object.__new__(QdrantStore)
        store.vector_size = 3
        store.client = client
        return store

    def setUp(self) -> None:
        client = QdrantClient(":memory:")
        self.store = self._create_store(client)
        self.markdown_store = MarkdownDocumentStore(self.store)
        self.retrieve_store = RetrieveChunkStore(self.store)
        self.paragraph_store = ParagraphChunkStore(self.store)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.store.initialize_collections()

    def test_initialize_collections_with_expected_vector_configs(self) -> None:
        markdown_config = self.store.client.get_collection(
            MARKDOWN_DOCUMENT_COLLECTION
        ).config.params.vectors
        retrieve_config = self.store.client.get_collection(
            RETRIEVE_CHUNK_COLLECTION
        ).config.params.vectors
        paragraph_config = self.store.client.get_collection(
            PARAGRAPH_CHUNK_COLLECTION
        ).config.params.vectors
        paragraph_sparse_config = self.store.client.get_collection(
            PARAGRAPH_CHUNK_COLLECTION
        ).config.params.sparse_vectors

        self.assertEqual(markdown_config, {})
        self.assertEqual(retrieve_config, {})
        self.assertEqual(paragraph_config[PARAGRAPH_DENSE_VECTOR].size, 3)
        self.assertEqual(
            paragraph_config[PARAGRAPH_DENSE_VECTOR].distance,
            models.Distance.COSINE,
        )
        self.assertEqual(
            set(paragraph_sparse_config),
            {PARAGRAPH_TEXT_SPARSE_VECTOR},
        )
        self.assertTrue(
            all(
                config.modifier == models.Modifier.IDF
                for config in paragraph_sparse_config.values()
            )
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.store.initialize_collections()

    def test_non_vector_collection_supports_exact_field_matching(self) -> None:
        self.store.client.upsert(
            collection_name=MARKDOWN_DOCUMENT_COLLECTION,
            points=[
                models.PointStruct(
                    id=1,
                    vector={},
                    payload={"file_name": "guide.pdf", "md5": "abc"},
                ),
                models.PointStruct(
                    id=2,
                    vector={},
                    payload={"file_name": "other.pdf", "md5": "def"},
                ),
            ],
        )

        records = self.markdown_store.query(
            {"file_name": "guide.pdf", "md5": "abc"},
        )

        self.assertEqual([record.id for record in records], [1])

    def test_paragraph_collection_uses_cosine_similarity(self) -> None:
        self.store.client.upsert(
            collection_name=PARAGRAPH_CHUNK_COLLECTION,
            points=[
                models.PointStruct(
                    id=1,
                    vector={PARAGRAPH_DENSE_VECTOR: [1.0, 0.0, 0.0]},
                ),
                models.PointStruct(
                    id=2,
                    vector={PARAGRAPH_DENSE_VECTOR: [0.0, 1.0, 0.0]},
                ),
            ],
        )

        results = self.paragraph_store.query([0.9, 0.1, 0.0], limit=2)

        self.assertEqual([result.id for result in results], [1, 2])

    def test_markdown_document_batch_insert_query_and_clear(self) -> None:
        documents = [
            SimpleNamespace(
                id=1,
                file_name="guide.pdf",
                author="Alice",
                original_file_url="https://example.com/guide.pdf",
                created_at=datetime(2026, 8, 3, 12, 0, 0),
                business_team_id=7,
                markdown_text="# Guide",
                md5="abc",
            ),
            SimpleNamespace(
                id=2,
                file_name="other.pdf",
                author="Bob",
                original_file_url="https://example.com/other.pdf",
                created_at=None,
                business_team_id=8,
                markdown_text="# Other",
                md5="def",
            ),
        ]

        inserted = self.markdown_store.batch_insert(documents)
        records = self.markdown_store.query(
            {"business_team_id": 7, "md5": "abc"}
        )

        self.assertEqual(inserted, 2)
        self.assertEqual([record.id for record in records], [1])
        self.assertEqual(records[0].payload["created_at"], "2026-08-03T12:00:00")

        self.markdown_store.clear()
        self.assertEqual(
            self.store.client.count(MARKDOWN_DOCUMENT_COLLECTION).count,
            0,
        )
        self.assertTrue(
            self.store.client.collection_exists(MARKDOWN_DOCUMENT_COLLECTION)
        )

    def test_retrieve_chunk_batch_insert_query_and_clear(self) -> None:
        chunks = [
            SimpleNamespace(id=11, doc_id=1, title_path="A > B", text="first"),
            SimpleNamespace(id=12, doc_id=2, title_path="C", text="second"),
        ]

        inserted = self.retrieve_store.batch_insert(chunks)
        records = self.retrieve_store.query({"doc_id": 1})

        self.assertEqual(inserted, 2)
        self.assertEqual([record.id for record in records], [11])
        self.assertEqual(records[0].payload["text"], "first")

        self.retrieve_store.clear()
        self.assertEqual(
            self.store.client.count(RETRIEVE_CHUNK_COLLECTION).count,
            0,
        )

    def test_paragraph_chunk_batch_insert_query_and_clear(self) -> None:
        chunks = [
            SimpleNamespace(
                id=111,
                retrieve_id=11,
                text="first paragraph",
                type="text",
                ai_desc_text="first description",
                embedding_vector=[1.0, 0.0, 0.0],
            ),
            SimpleNamespace(
                id=112,
                retrieve_id=11,
                text="second paragraph",
                type="text",
                ai_desc_text="",
                embedding_vector=[0.0, 1.0, 0.0],
            ),
        ]

        with patch.object(self.store.client, "upsert") as upsert:
            inserted = self.paragraph_store.batch_insert(chunks)

        points = upsert.call_args.kwargs["points"]

        self.assertEqual(inserted, 2)
        self.assertEqual(
            points[0].vector[PARAGRAPH_DENSE_VECTOR],
            [1.0, 0.0, 0.0],
        )
        self.assertIsInstance(
            points[0].vector[PARAGRAPH_TEXT_SPARSE_VECTOR],
            models.Document,
        )
        self.assertEqual(
            points[0].vector[PARAGRAPH_TEXT_SPARSE_VECTOR].text,
            "first description",
        )
        self.assertEqual(
            points[1].vector[PARAGRAPH_TEXT_SPARSE_VECTOR].text,
            "second paragraph",
        )

        self.store.client.upsert(
            collection_name=PARAGRAPH_CHUNK_COLLECTION,
            points=[
                models.PointStruct(
                    id=111,
                    vector={PARAGRAPH_DENSE_VECTOR: [1.0, 0.0, 0.0]},
                    payload={"text": "first paragraph"},
                ),
                models.PointStruct(
                    id=112,
                    vector={PARAGRAPH_DENSE_VECTOR: [0.0, 1.0, 0.0]},
                    payload={"text": "second paragraph"},
                ),
            ],
        )
        results = self.paragraph_store.query([0.9, 0.1, 0.0], limit=2)

        self.assertEqual([result.id for result in results], [111, 112])
        self.assertEqual(results[0].payload["text"], "first paragraph")

        self.paragraph_store.clear()
        self.assertEqual(
            self.store.client.count(PARAGRAPH_CHUNK_COLLECTION).count,
            0,
        )

    def test_paragraph_sparse_query_uses_text_named_vector(self) -> None:
        response = SimpleNamespace(points=[])
        with patch.object(
            self.store.client,
            "query_points",
            return_value=response,
        ) as query_points:
            self.paragraph_store.query_text_sparse("安装部署")
            text_call = query_points.call_args.kwargs

        self.assertEqual(text_call["using"], PARAGRAPH_TEXT_SPARSE_VECTOR)
        self.assertEqual(text_call["query"].model, "qdrant/bm25")
        self.assertEqual(text_call["query"].text, "安装部署")
        self.assertEqual(text_call["query"].options.language, "chinese")

    def test_empty_batch_is_a_no_op(self) -> None:
        self.assertEqual(self.markdown_store.batch_insert([]), 0)
        self.assertEqual(self.retrieve_store.batch_insert([]), 0)
        self.assertEqual(self.paragraph_store.batch_insert([]), 0)

    def test_collection_operations_do_not_initialize_collections(self) -> None:
        store = self._create_store(QdrantClient(":memory:"))
        markdown_store = MarkdownDocumentStore(store)
        retrieve_store = RetrieveChunkStore(store)
        paragraph_store = ParagraphChunkStore(store)

        with self.assertRaisesRegex(ValueError, "not found"):
            markdown_store.query({"md5": "abc"})
        with self.assertRaisesRegex(ValueError, "not found"):
            retrieve_store.query({"doc_id": 1})
        with self.assertRaisesRegex(ValueError, "not found"):
            paragraph_store.query([1.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "not found"):
            markdown_store.clear()


if __name__ == "__main__":
    unittest.main()
