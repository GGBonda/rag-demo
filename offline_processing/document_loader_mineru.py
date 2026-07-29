"""
RAG 知识库 - MinerU 文档加载器模块
使用 MinerU 云 API 精准解析 PDF，并返回 Markdown
"""

import base64
import hashlib
import mimetypes
import posixpath
import re
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from config import config


@dataclass
class MarkdownDocument:
    """PDF 解析后的 Markdown 文档"""
    """主键 ID"""
    id: int | None = None
    """文件名"""
    file_name: str = ""
    """作者"""
    author: str = ""
    """原文件 URL"""
    original_file_url: str = ""
    """文档创建时间"""
    created_at: datetime | None = None
    """所属业务团队 ID"""
    business_team_id: int | None = None
    """Markdown 文本"""
    markdown_text: str = ""
    """原文件内容的 MD5 摘要"""
    md5: str = ""


class MinerUDocumentLoader:
    """使用 MinerU 云 API 精准解析单个 PDF，并返回 Markdown"""

    SUPPORTED_EXTENSIONS = {
        ".pdf": "application/pdf",
    }

    _FILE_URLS_API = "https://mineru.net/api/v4/file-urls/batch"
    _BATCH_RESULT_API = "https://mineru.net/api/v4/extract-results/batch"
    _POLL_INTERVAL_SECONDS = 3
    _POLL_TIMEOUT_SECONDS = 30 * 60
    _REQUEST_TIMEOUT_SECONDS = 5 * 60
    _MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def __init__(
        self,
        file_path: str | None = None,
        method: str = "auto",
        output_dir: str | None = None,
        lang: str = "ch",
        start_page: int | None = None,
        end_page: int | None = None,
    ):
        """
        Args:
            file_path: PDF 文件路径
            method: 解析方式；"ocr" 强制 OCR，"auto" 和 "txt" 不强制 OCR
            output_dir: 保留用于兼容旧调用，云 API 模式下不使用
            lang: 文档语言代码，默认 "ch"（中文）
            start_page: 起始页码（1-indexed，含），None 表示从第一页开始
            end_page: 结束页码（1-indexed，含），None 表示到最后一页
        """
        self.file_path = Path(file_path) if file_path else None
        self.method = method
        self.lang = lang
        self.start_page = start_page
        self.end_page = end_page
        self._output_dir = output_dir
        self._temp_dir: tempfile.TemporaryDirectory | None = None

        if method not in ("auto", "txt", "ocr"):
            raise ValueError(f"不支持的解析方式: {method}，可选值: auto, txt, ocr")

    @property
    def output_dir(self) -> str:
        """获取输出目录（懒初始化）"""
        if self._output_dir is None:
            if self._temp_dir is None:
                self._temp_dir = tempfile.TemporaryDirectory(prefix="mineru_")
            self._output_dir = self._temp_dir.name
        return self._output_dir

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def load(self, file_path: str | None = None) -> MarkdownDocument:
        """通过 MinerU 精准解析 API 加载 PDF，返回 Markdown 文档"""
        source = file_path or self.file_path
        if source is None:
            raise ValueError("未指定 PDF 文件路径")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {path.suffix.lower()}")

        token = config.mineru.api_token.strip()
        if not token:
            raise ValueError("未配置 MINERU_API_TOKEN 环境变量")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        file_options = {
            "name": path.name,
            "is_ocr": self.method == "ocr",
        }
        page_ranges = self._build_page_ranges()
        if page_ranges:
            file_options["page_ranges"] = page_ranges

        create_response = requests.post(
            self._FILE_URLS_API,
            headers=headers,
            json={
                "files": [file_options],
                "model_version": "vlm",
                "language": self.lang,
            },
            timeout=self._REQUEST_TIMEOUT_SECONDS,
        )
        create_data = self._get_response_data(create_response, "申请文件上传地址")
        batch_id = create_data.get("batch_id")
        file_urls = create_data.get("file_urls") or []
        if not batch_id or not file_urls:
            raise RuntimeError("MinerU 未返回 batch_id 或文件上传地址")

        with path.open("rb") as pdf_file:
            upload_response = requests.put(
                file_urls[0],
                data=pdf_file,
                timeout=self._REQUEST_TIMEOUT_SECONDS,
            )
        upload_response.raise_for_status()

        result_url = f"{self._BATCH_RESULT_API}/{batch_id}"
        deadline = time.monotonic() + self._POLL_TIMEOUT_SECONDS
        while True:
            result_response = requests.get(
                result_url,
                headers=headers,
                timeout=self._REQUEST_TIMEOUT_SECONDS,
            )
            result_data = self._get_response_data(result_response, "查询解析结果")
            extract_results = result_data.get("extract_result") or []
            extract_result = extract_results[0] if extract_results else {}
            state = extract_result.get("state")

            if state == "done":
                full_zip_url = extract_result.get("full_zip_url")
                if not full_zip_url:
                    raise RuntimeError("MinerU 解析完成，但未返回结果下载地址")
                break
            if state == "failed":
                error_message = extract_result.get("err_msg") or "未知错误"
                raise RuntimeError(f"MinerU 解析失败: {error_message}")
            if time.monotonic() >= deadline:
                raise TimeoutError("等待 MinerU 解析结果超时")

            time.sleep(self._POLL_INTERVAL_SECONDS)

        download_response = requests.get(
            full_zip_url,
            stream=True,
            timeout=self._REQUEST_TIMEOUT_SECONDS,
        )
        download_response.raise_for_status()
        with tempfile.TemporaryFile() as zip_file:
            for chunk in download_response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    zip_file.write(chunk)
            zip_file.seek(0)
            with zipfile.ZipFile(zip_file) as result_zip:
                markdown_path = next(
                    (
                        name
                        for name in result_zip.namelist()
                        if Path(name).name == "full.md"
                    ),
                    None,
                )
                if markdown_path is None:
                    raise FileNotFoundError("MinerU 解析结果中未找到 full.md")
                markdown_text = result_zip.read(markdown_path).decode("utf-8")
                markdown_text = self._replace_zip_images_with_base64(
                    markdown_text,
                    markdown_path,
                    result_zip,
                )

        return MarkdownDocument(
            file_name=path.name,
            markdown_text=markdown_text,
            md5=self._calculate_md5(path),
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_md5(path: Path) -> str:
        """计算文件内容的 MD5 摘要"""
        digest = hashlib.md5()
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _replace_zip_images_with_base64(
        self,
        markdown_text: str,
        markdown_path: str,
        result_zip: zipfile.ZipFile,
    ) -> str:
        """将 Markdown 中引用的 ZIP 内图片替换为 Base64 data URI"""
        zip_entries = {
            posixpath.normpath(name): name
            for name in result_zip.namelist()
            if not name.endswith("/")
        }
        markdown_dir = posixpath.dirname(markdown_path)
        encoded_images: dict[str, str] = {}

        def replace_image(match: re.Match) -> str:
            alt_text = match.group(1)
            raw_target = match.group(2).strip()
            image_target, title_suffix = self._split_image_target(raw_target)
            parsed_target = urlparse(image_target)
            if parsed_target.scheme or image_target.startswith("//"):
                return match.group(0)

            image_path = unquote(parsed_target.path).replace("\\", "/")
            if not image_path:
                return match.group(0)

            candidates = (
                posixpath.normpath(
                    posixpath.join(markdown_dir, image_path.lstrip("/"))
                ),
                posixpath.normpath(image_path.lstrip("/")),
            )
            zip_image_path = next(
                (
                    zip_entries[candidate]
                    for candidate in candidates
                    if candidate in zip_entries
                ),
                None,
            )
            if zip_image_path is None:
                return match.group(0)

            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith("image/"):
                return match.group(0)

            data_uri = encoded_images.get(zip_image_path)
            if data_uri is None:
                encoded = base64.b64encode(
                    result_zip.read(zip_image_path)
                ).decode("ascii")
                data_uri = f"data:{mime_type};base64,{encoded}"
                encoded_images[zip_image_path] = data_uri
            return f"![{alt_text}]({data_uri}{title_suffix})"

        return self._MARKDOWN_IMAGE_RE.sub(replace_image, markdown_text)

    @staticmethod
    def _split_image_target(raw_target: str) -> tuple[str, str]:
        """拆分 Markdown 图片地址和可选标题"""
        if raw_target.startswith("<"):
            closing_bracket = raw_target.find(">")
            if closing_bracket != -1:
                return (
                    raw_target[1:closing_bracket],
                    raw_target[closing_bracket + 1:],
                )

        parts = raw_target.split(maxsplit=1)
        image_target = parts[0]
        title_suffix = f" {parts[1]}" if len(parts) == 2 else ""
        return image_target, title_suffix

    def _build_page_ranges(self) -> str | None:
        """将本地页码参数转换为 MinerU API 的 page_ranges 格式"""
        if self.start_page is None and self.end_page is None:
            return None

        start_page = self.start_page or 1
        end_page = self.end_page if self.end_page is not None else -1
        if start_page < 1 or (end_page != -1 and end_page < start_page):
            raise ValueError("页码范围无效")
        return f"{start_page}-{end_page}"

    @staticmethod
    def _get_response_data(response: requests.Response, action: str) -> dict:
        """校验 MinerU API 响应并返回 data 字段"""
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"MinerU {action}失败: {payload.get('msg', '未知错误')}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"MinerU {action}响应缺少 data 字段")
        return data
