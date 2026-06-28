import ssl

# Workaround: 解决 macOS 上 NLTK 数据下载时的 SSL 证书验证问题
# 错误: [nltk_data] error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from offline_processing.document_loader_manager import DocumentLoaderManager
from offline_processing.chunker import Chunker


if __name__ == '__main__':
    manager = DocumentLoaderManager(backend="mineru", file_path="/Users/shengjunhui/Downloads/rag_test_pdf/阿里开发规范.pdf", start_page=3)

    doc = manager.load()

    chunker = Chunker(chunk_size=1000, chunk_overlap=0)

    chunks = chunker.chunk_markdown(
        doc.markdown_text,
        file_name=doc.file_name,
        file_path=doc.file_path,
    )

    for i, chunk in enumerate(chunks):
        print(f"=========================================================={i}, node length: {len(chunk.text)}")
        print(chunk.text)
