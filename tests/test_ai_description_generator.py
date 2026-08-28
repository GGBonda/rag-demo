import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from offline_processing.ai_description_generator import AIDescriptionGenerator


class BuildMultimodalUserContentTest(unittest.TestCase):

    def test_preserves_mixed_text_and_image_order(self) -> None:
        user_content = (
            "第一张图片：![小狗](https://example.com/dog.jpeg)"
            "第二张图片：![图表](data:image/png;base64,iVBORw0KGgoAAAANS)"
            "这些图片里有什么内容？"
        )

        result = AIDescriptionGenerator.build_multimodal_user_content(user_content)

        self.assertEqual(
            result,
            [
                {"type": "text", "text": "第一张图片："},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/dog.jpeg"},
                },
                {"type": "text", "text": "第二张图片："},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANS"
                    },
                },
                {"type": "text", "text": "这些图片里有什么内容？"},
            ],
        )

    def test_ignores_whitespace_between_adjacent_images(self) -> None:
        user_content = (
            "![图片一](https://example.com/one.png)\n"
            "![图片二](https://example.com/two.png)\n"
            "这些图片里有什么内容？"
        )

        result = AIDescriptionGenerator.build_multimodal_user_content(user_content)

        self.assertEqual(
            result,
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/one.png"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/two.png"},
                },
                {"type": "text", "text": "这些图片里有什么内容？"},
            ],
        )

    def test_returns_text_content_when_there_is_no_image(self) -> None:
        result = AIDescriptionGenerator.build_multimodal_user_content("纯文本内容")

        self.assertEqual(result, [{"type": "text", "text": "纯文本内容"}])


if __name__ == "__main__":
    unittest.main()
