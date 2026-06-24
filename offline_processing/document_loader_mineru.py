"""
RAG 知识库 - MinerU 文档加载器模块
使用 MinerU pipeline 后端加载 PDF 并解析为 Markdown 格式
"""

import os
import tempfile
from pathlib import Path
from typing import List

from .document_loader_unstructured import ParsedDocument


class MinerUDocumentLoader:
    """使用 MinerU pipeline 引擎的文档加载器，加载 PDF 并解析为 Markdown 格式"""

    SUPPORTED_EXTENSIONS = {
        ".pdf": "application/pdf",
    }

    def __init__(
        self,
        input_dir: str = "./documents",
        method: str = "auto",
        output_dir: str | None = None,
        lang: str = "ch",
    ):
        """
        Args:
            input_dir: 文档目录路径
            method: 解析方式 - "auto"（自动选择）、"txt"（文本模式）、"ocr"（OCR 模式）
            output_dir: 中间文件输出目录，默认使用临时目录
            lang: 文档语言代码，默认 "ch"（中文）
        """
        self.input_dir = Path(input_dir)
        self.method = method
        self.lang = lang
        self._output_dir = output_dir
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        os.makedirs(self.input_dir, exist_ok=True)

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

    def _get_supported_files(self) -> List[Path]:
        """获取目录下所有支持的文件"""
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.input_dir.glob(f"**/*{ext}"))
        return sorted(files)

    def load_all(self) -> List[ParsedDocument]:
        """加载目录下所有支持的文档，返回 ParsedDocument 列表"""
        results: List[ParsedDocument] = []
        files = self._get_supported_files()

        if not files:
            print(f"警告: 目录 '{self.input_dir}' 下未找到支持的文档文件")
            print(f"支持的格式: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}")
            return results

        print(f"找到 {len(files)} 个文件待加载...")

        for file_path in files:
            try:
                markdown_text = self._load_single_file(file_path)
                parsed = ParsedDocument(
                    markdown_text=markdown_text,
                    file_name=file_path.name,
                    file_path=str(file_path.resolve()),
                )
                results.append(parsed)
                print(f"  ✓ 已加载: {file_path.name} ({len(markdown_text)} 字符)")
            except Exception as e:
                print(f"  ✗ 加载失败: {file_path.name} - {e}")

        print(f"\n总共加载 {len(results)} 个文件")
        return results

    def load_file(self, file_path: str) -> ParsedDocument:
        """加载单个指定文件，返回 ParsedDocument"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
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
