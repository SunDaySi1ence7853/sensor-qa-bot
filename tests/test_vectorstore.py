"""
测试 src/vectorstore.py

覆盖：
1. 知识库目录不存在时报错
2. 知识库为空时报错
3. 正常加载文档并切分
4. 遇到编码错误文件：跳过而不崩溃
5. 加载不存在的向量库时报错
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import vectorstore as vs_module


@pytest.fixture
def fake_knowledge_dir(tmp_path, valid_env, monkeypatch):
    """
    创建一个临时知识库目录，替换模块级的 KNOWLEDGE_DIR。
    """
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    monkeypatch.setattr(vs_module, "KNOWLEDGE_DIR", kdir)
    return kdir


class TestLoadDocuments:
    def test_missing_knowledge_dir_raises(self, valid_env, tmp_path, monkeypatch):
        missing = tmp_path / "does_not_exist"
        monkeypatch.setattr(vs_module, "KNOWLEDGE_DIR", missing)
        with pytest.raises(RuntimeError, match="知识库目录不存在"):
            vs_module._load_documents()

    def test_empty_knowledge_dir_raises(self, fake_knowledge_dir):
        with pytest.raises(RuntimeError, match="知识库为空"):
            vs_module._load_documents()

    def test_loads_valid_txt_and_md_files(self, fake_knowledge_dir):
        (fake_knowledge_dir / "sensor1.txt").write_text(
            "DHT22 是数字温湿度传感器。测量范围 -40 到 80 摄氏度。",
            encoding="utf-8",
        )
        (fake_knowledge_dir / "sensor2.md").write_text(
            "# BMP280\n\nBMP280 是气压传感器，精度 ±1 hPa。",
            encoding="utf-8",
        )

        docs = vs_module._load_documents()
        assert len(docs) >= 2
        all_content = " ".join(d.page_content for d in docs)
        assert "DHT22" in all_content
        assert "BMP280" in all_content

    def test_skips_gbk_encoded_file(self, fake_knowledge_dir, monkeypatch, caplog):
        import logging

        # 正常 UTF-8 文件
        (fake_knowledge_dir / "good.txt").write_text("正常内容", encoding="utf-8")

        # GBK 编码文件（会读取失败）
        (fake_knowledge_dir / "bad_gbk.txt").write_bytes("你好".encode("gbk"))

        with caplog.at_level(logging.WARNING, logger="src.vectorstore"):
            docs = vs_module._load_documents()

        # 断言 warning 里提到了 bad_gbk.txt
        assert any("bad_gbk.txt" in rec.message for rec in caplog.records)
        # 且正常文件被加载了
        assert len(docs) > 0

    def test_all_files_broken_raises(self, fake_knowledge_dir):
        (fake_knowledge_dir / "broken1.txt").write_bytes("中文".encode("gbk"))
        (fake_knowledge_dir / "broken2.txt").write_bytes("更多中文".encode("gbk"))

        with pytest.raises(RuntimeError, match="所有文件都无法读取"):
            vs_module._load_documents()

    def test_ignores_non_text_files(self, fake_knowledge_dir):
        (fake_knowledge_dir / "keep.txt").write_text("内容", encoding="utf-8")
        (fake_knowledge_dir / "ignore.pdf").write_bytes(b"%PDF-fake")
        (fake_knowledge_dir / "ignore.jpg").write_bytes(b"\xff\xd8\xff")

        docs = vs_module._load_documents()
        assert len(docs) >= 1
        all_content = " ".join(d.page_content for d in docs)
        assert "内容" in all_content


class TestLoadVectorstore:
    def test_missing_index_file_raises(self, valid_env, tmp_path, monkeypatch):
        empty_dir = tmp_path / "vectorstore_empty"
        empty_dir.mkdir()
        monkeypatch.setattr(vs_module, "PROJECT_ROOT", tmp_path)
        valid_env.setenv("VECTORSTORE_DIR", "vectorstore_empty")

        with pytest.raises(RuntimeError, match="向量库不存在"):
            vs_module.load_vectorstore()

    def test_corrupt_index_raises_friendly_error(self, valid_env, tmp_path, monkeypatch):
        """索引文件存在但损坏时应给出清晰指引。"""
        vdir = tmp_path / "vectorstore_corrupt"
        vdir.mkdir()
        (vdir / "index.faiss").write_bytes(b"not a real faiss index")
        (vdir / "index.pkl").write_bytes(b"garbage")

        monkeypatch.setattr(vs_module, "PROJECT_ROOT", tmp_path)
        valid_env.setenv("VECTORSTORE_DIR", "vectorstore_corrupt")

        # mock 掉 embeddings 以避免真的下载模型
        monkeypatch.setattr(vs_module, "get_embeddings", lambda: MagicMock())

        with pytest.raises(RuntimeError, match="向量库加载失败"):
            vs_module.load_vectorstore()