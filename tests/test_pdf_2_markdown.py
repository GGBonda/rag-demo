import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from offline_processing.document_loader_mineru import MinerUDocumentLoader


PDF_DIR = PROJECT_ROOT / "doc_pdf"
MARKDOWN_DIR = PROJECT_ROOT / "doc_markdown"


def convert_pdfs_to_markdown() -> None:
    """逐个解析 doc_pdf 下的 PDF，并将 Markdown 写入 doc_markdown。"""
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(
        path
        for path in PDF_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_paths:
        print(f"未在 {PDF_DIR} 中找到 PDF 文件")
        return

    loader = MinerUDocumentLoader()
    for pdf_path in pdf_paths:
        print(f"正在解析: {pdf_path.name}")
        document = loader.load(str(pdf_path))

        markdown_path = MARKDOWN_DIR / f"{pdf_path.stem}.md"
        markdown_path.write_text(document.markdown_text, encoding="utf-8")
        print(f"已生成: {markdown_path}")


if __name__ == "__main__":
    convert_pdfs_to_markdown()
