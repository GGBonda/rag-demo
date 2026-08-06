import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config import config
from offline_processing.ai_description_generator import AIDescriptionGenerator
from offline_processing.chunker import RetrieveChunk


class GenerateRetrieveDescTest(unittest.TestCase):
    def test_generates_each_description_and_question_as_index_chunk(self) -> None:
        response_content = json.dumps(
            {
                "descriptions": ["简介一", "简介二"],
                "questions": ["问题一？", "问题二？", "问题三？"],
            },
            ensure_ascii=False,
        )
        generator = object.__new__(AIDescriptionGenerator)
        generator.text_model = "text-model"
        generator.vision_model = "vision-model"
        generator._text_client = Mock()
        generator._text_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response_content),
                )
            ]
        )
        generator._vision_client = Mock()

        with (
            patch.object(config.chunk, "index_chunk_description_count", 2),
            patch.object(config.chunk, "index_chunk_question_count", 3),
        ):
            chunks = generator.generate_retrieve_desc(
                [RetrieveChunk(id=123, title_path="标题", text="正文")],
                start_index=10,
            )

        self.assertEqual(len(chunks), 5)
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["简介一", "简介二", "问题一？", "问题二？", "问题三？"],
        )
        self.assertEqual([chunk.retrieve_id for chunk in chunks], [123] * 5)
        self.assertEqual(
            [chunk.id for chunk in chunks],
            [123011, 123012, 123013, 123014, 123015],
        )
        call_kwargs = generator._text_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["response_format"], {"type": "json_object"})

    def test_parses_json_from_markdown_code_block(self) -> None:
        descriptions, questions = AIDescriptionGenerator._parse_index_content(
            '```json\n{"descriptions":["简介"],"questions":["问题？"]}\n```',
            1,
            1,
        )

        self.assertEqual(descriptions, ["简介"])
        self.assertEqual(questions, ["问题？"])

    def test_parses_json_with_surrounding_explanation(self) -> None:
        descriptions, questions = AIDescriptionGenerator._parse_index_content(
            '生成结果如下：{"descriptions":["简介"],"questions":["问题？"]}请查收。',
            1,
            1,
        )

        self.assertEqual(descriptions, ["简介"])
        self.assertEqual(questions, ["问题？"])

    def test_rejects_malformed_json(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "不是有效的 JSON 对象"):
            AIDescriptionGenerator._parse_index_content(
                "{'descriptions':['简介'],'questions':['问题？']}",
                1,
                1,
            )


if __name__ == "__main__":
    unittest.main()
