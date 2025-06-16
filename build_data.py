import json
import re

import numpy as np
import faiss
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import time


class VocabIndexer:
    """
    词汇索引构建器 - 负责将词汇表构建成FAISS索引
    """

    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化词汇索引构建器

        Args:
            model_name: SentenceTransformer模型名称
        """
        print(f"正在加载模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"模型加载完成，嵌入维度: {self.embedding_dim}")

    def load_vocab(self, vocab_path):
        """
        加载词汇表

        Args:
            vocab_path: 词汇表文件路径

        Returns:
            词汇表(set)
        """
        with open(vocab_path, 'rb') as f:
            return pickle.load(f)\

    def build_embedding(self, vocab, batch_size=1000, index_type='L2'):
        """
        为词汇构建FAISS索引

        Args:
            vocab: 词汇集合
            batch_size: 批处理大小
            index_type: 索引类型，可选 'L2'(欧氏距离) 或 'IP'(内积，余弦相似度)

        Returns:
            (index, word_list): FAISS索引和对应的词汇列表
        """
        vocab_list = list(vocab)
        word_to_embedding_map = {}

        # 创建FAISS索引
        if index_type == 'L2':
            # 使用欧氏距离的索引
            index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            # 使用内积的索引（适用于余弦相似度）
            index = faiss.IndexFlatIP(self.embedding_dim)

        start_time = time.time()
        for i in tqdm(range(0, len(vocab_list), batch_size), desc="构建索引"):
            batch_words = vocab_list[i:i + batch_size]

            # 计算词嵌入
            embeddings = self.model.encode(batch_words, show_progress_bar=False)

            # 【新增】填充词语到嵌入的映射字典
            for word, embedding in zip(batch_words, embeddings):
                word_to_embedding_map[word] = embedding.astype('float32')

            # 如果使用内积索引，需要归一化向量
            if index_type == 'IP':
                faiss.normalize_L2(embeddings)

            # 添加到索引
            index.add(np.array(embeddings).astype('float32'))

            # 每10个批次显示进度
            if (i // batch_size) % 10 == 0 and i > 0:
                elapsed = time.time() - start_time
                words_per_sec = i / elapsed
                remaining = (len(vocab_list) - i) / words_per_sec if words_per_sec > 0 else 0
                print(f"已处理 {i}/{len(vocab_list)} 词汇 "
                      f"({i / len(vocab_list) * 100:.2f}%), "
                      f"预计剩余时间: {remaining / 60:.2f}分钟")

        return index, word_to_embedding_map

    def save_faiss(self, index, faiss_path):
        """
        保存FAISS索引和词汇列表
        """
        faiss.write_index(index, faiss_path)

    def save_map(self, word_map, embedding_map_path):
        with open(embedding_map_path, 'wb') as f:
            pickle.dump(word_map, f)

def build_faiss(vocab_path, faiss_path, embedding_map_path):
    # 初始化索引构建器
    indexer = VocabIndexer(model_name='shibing624/text2vec-base-chinese')

    # 加载词汇表
    vocab = indexer.load_vocab(vocab_path)
    print(f"词汇表加载完成，包含{len(vocab)}个词汇")

    # 构建索引
    print("开始构建FAISS索引...")
    index, word_map = indexer.build_embedding(vocab, batch_size=1000, index_type='IP')

    # 保存索引
    indexer.save_faiss(index, faiss_path)
    indexer.save_map(word_map, embedding_map_path)

    print("索引构建和保存完成！")

def build_synonyms(input_path, output_path):
    """
    正确转换包含Unicode转义序列的近义词库
    输入文件必须是原始Unicode转义格式（如："\u54c0\u6bc1"）
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 此处会自动解码一次

    try:
        # 将数据写入Python文件
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"转换完成！数据已保存到 {output_path}")
        return True
    except Exception as e:
        print(f"转换过程中出错: {e}")
        return False


def serialize_vocab(input_path: str, output_path: str) -> None:
    """
    清洗词汇表并序列化为Python set

    Args:
        input_path: 原始词汇表文件路径（txt格式，每行一个词）
        output_path: 输出序列化文件路径（.pkl格式）
    """
    chinese_pattern = re.compile(r'^[\u4e00-\u9fa5]{2,2}$')  # 匹配1-4个中文字符
    vocab = list()

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if chinese_pattern.fullmatch(word):
                vocab.append(word)

    # 序列化保存
    with open(output_path, 'wb') as f:
        pickle.dump(vocab, f)

    print(f"清洗完成！共保留{len(vocab)}个有效词，已保存到 {output_path}")


if __name__ == "__main__":
    word_list_path = './data/vocal_list.pkl'
    faiss_path = './data/vocal_list.faiss'
    embedding_map_path = './data/word_embedding_map.pkl'

    # 1. 构造词汇表
    serialize_vocab(
        input_path="data/tight/tight.txt",
        output_path=word_list_path
    )

    # 2. 构造faiss数据库
    build_faiss(word_list_path, faiss_path, embedding_map_path)

    # # 3. 构造同义词表
    # build_synonyms(
    #     input_path='./data/synonyms_expanded_narrow.json',
    #     output_path='./data/synonyms.pkl'
    # )

