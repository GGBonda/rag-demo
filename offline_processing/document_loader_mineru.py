"""
RAG 知识库 - MinerU 文档加载器模块
使用 MinerU pipeline 后端加载 PDF 并解析为 Markdown 格式
"""

import os
import tempfile
from pathlib import Path

from .document_loader_unstructured import ParsedDocument


class MinerUDocumentLoader:
    """使用 MinerU pipeline 引擎的文档加载器，加载单个 PDF 并解析为 Markdown 格式"""

    SUPPORTED_EXTENSIONS = {
        ".pdf": "application/pdf",
    }

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
            method: 解析方式 - "auto"（自动选择）、"txt"（文本模式）、"ocr"（OCR 模式）
            output_dir: 中间文件输出目录，默认使用临时目录
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

    def load(self, file_path: str | None = None) -> ParsedDocument:
        """加载指定 PDF 文件，返回 ParsedDocument"""
        path = Path(file_path or self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        markdown_text = self._load_single_file(path)
        return ParsedDocument(
            markdown_text=markdown_text,
            file_name=path.name,
            file_path=str(path.resolve()),
        )

    # ------------------------------------------------------------------
    # 文档解析核心
    # ------------------------------------------------------------------

    def _load_single_file(self, file_path: Path) -> str:
        """加载单个 PDF 文件，返回解析后的 Markdown 文本"""
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {suffix}")

        from mineru.cli.common import do_parse
        from mineru.cli.common import read_fn
        from mineru.utils.enum_class import MakeMode

        # 读取 PDF 原始字节
        pdf_bytes = read_fn(file_path)

        # 每个文件使用独立的输出子目录
        file_stem = file_path.stem
        file_output_dir = os.path.join(self.output_dir, file_stem)
        os.makedirs(file_output_dir, exist_ok=True)

        # 调用 mineru pipeline 解析，只输出 markdown
        do_parse(
            output_dir=file_output_dir,
            pdf_file_names=[file_stem],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[self.lang],
            backend="pipeline",
            parse_method=self.method,
            start_page_id=self.start_page,
            end_page_id=self.end_page,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
            f_make_md_mode=MakeMode.MM_MD,
        )

        # 读取生成的 markdown 文件
        md_path = os.path.join(
            file_output_dir, file_stem, self.method, f"{file_stem}.md"
        )
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"解析后的 Markdown 文件未生成: {md_path}")

        with open(md_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        return markdown_text
