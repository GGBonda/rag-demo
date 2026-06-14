import ssl

# Workaround: 解决 macOS 上 NLTK 数据下载时的 SSL 证书验证问题
# 错误: [nltk_data] error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from offline_processing.document_loader import DocumentLoader
from offline_processing.chunker import Chunker

loader = DocumentLoader(
    input_dir='/Users/shengjunhui/Downloads/rag_test_pdf',
)

parsedDocuments = loader.load_all()

chunker = Chunker(chunk_size=10000, chunk_overlap=0)

chunks = chunker.chunk_elements(parsedDocuments[0])

for i, chunk in enumerate(chunks):
    print(f"=========================================================={i}, node length: {len(chunk.text)}")
    print(chunk.text)
