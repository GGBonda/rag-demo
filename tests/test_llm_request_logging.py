import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from config import config
from llm import request_code_description, request_cross_encoder_rerank


class LLMRequestLoggingTest(unittest.TestCase):

    @patch("llm.llm_request.time.perf_counter", side_effect=[10.0, 11.25])
    @patch("llm.llm_request.requests.post")
    def test_chat_request_logs_in_llm_request(self, post, _perf_counter) -> None:
        post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {"content": "模型回复"},
                    "finish_reason": "stop",
                }
            ]
        }

        with (
            patch.object(config.llm, "openai_api_key", "text-key"),
            patch.object(config.llm, "openai_base_url", "https://text.test/v1"),
            patch.object(config.llm, "openai_pro_model", "text-model"),
            redirect_stdout(io.StringIO()) as output,
        ):
            request_code_description("代码内容")

        self.assertIn(
            "[大模型请求] 请求开始: task=代码描述, model=text-model",
            output.getvalue(),
        )
        self.assertIn(
            "[大模型请求] 请求结束: task=代码描述, model=text-model, 耗时=1.25秒",
            output.getvalue(),
        )

    @patch("llm.llm_request.time.perf_counter", side_effect=[20.0, 20.5])
    @patch("llm.llm_request.requests.post")
    def test_rerank_request_logs_in_llm_request(self, post, _perf_counter) -> None:
        post.return_value.json.return_value = {"output": {"results": []}}

        with (
            patch.object(
                config.cross_encoder,
                "base_url",
                "https://rerank.test/api",
            ),
            patch.object(config.cross_encoder, "api_key", "rerank-key"),
            patch.object(config.cross_encoder, "model", "rerank-model"),
            redirect_stdout(io.StringIO()) as output,
        ):
            request_cross_encoder_rerank(
                query="问题",
                documents=[{"text": "文档"}],
            )

        self.assertIn(
            "[Cross-Encoder 精排] 请求开始: model=rerank-model, documents=1",
            output.getvalue(),
        )
        self.assertIn(
            "[Cross-Encoder 精排] 请求结束: model=rerank-model, 耗时=0.50秒",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
