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

loader = DocumentLoader(input_dir='/Users/shengjunhui/Downloads/rag_test_pdf')
chunker = Chunker(
    chunk_size=1000,
    chunk_overlap=0,
    strategy='chapter',
)

documents = loader.load_all()

nodes = chunker.split(documents)

for i, node in enumerate(nodes):
    print(f"=========================================================={i}")
    print(node.text)
