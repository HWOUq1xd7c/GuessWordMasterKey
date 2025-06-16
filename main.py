import heapq
import faiss
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import random

from api import GuessWordAPI


# 判题器类保持不变
class LocalJudge:
    def __init__(self, target_word, model, vocab):
        self.target_word = target_word
        self.model = model
        self.vocab = vocab
        if target_word not in vocab:
            raise ValueError("Target word not in vocabulary")
        self.target_embedding = model.encode([target_word], convert_to_tensor=True)

    def query(self, word):
        """供搜索器调用的相似度计算接口"""
        if word == self.target_word:
            return 1.0
        if word not in self.vocab:
            return 0.0
        query_embedding = self.model.encode([word], convert_to_tensor=True)
        return util.cos_sim(self.target_embedding, query_embedding).item()

class OnlineJudge:
    def __init__(self):
        try:
            self.api = GuessWordAPI(0.5, 0.1, True)
        except Exception as e:
            print(f"Warning: Could not initialize OnlineJudge API: {e}")
            self.api = None

    def query(self, word):
        if self.api is None:
            print("Error: API not initialized. Cannot query.")
            return 0.0

        result = self.api.guess_word(word)
        if result['correct']:
            return 1.0
        elif result['similarity'] is None:
            return 0.0
        return result['similarity']


class Searcher:
    """词语搜索器类，负责执行搜索策略"""

    def __init__(self, faiss_index, word_list, embedding_map, max_queries=100):
        """
        初始化搜索器
        """
        self.faiss_index = faiss_index
        self.word_list = word_list
        self.embedding_map = embedding_map
        self.word_set = set(word_list)
        self.max_queries = max_queries
        self.run_initial_exploration = True # 控制是否执行初始探索

        # 维护搜索状态
        self.queried = {}  # 已查询词及其相似度 {word: similarity}

        # 取前K个词中的随机一个进行新的搜索
        self.search_window_size = 5
        # faiss搜索返回数量
        self.faiss_search_size = 50

        # 存储 (相似度, 嵌入向量)，使用最小堆以便高效获取最高和最低相似度项
        self.top_k_heap = []

    def _update_state(self, word, similarity):
        """更新搜索状态，包括记录已查询词和维护最大堆。"""
        if word not in self.queried:
            self.queried[word] = similarity
            # 只有相似度大于0的词才值得放入堆中作为未来的种子
            if similarity > 0:
                heapq.heappush(self.top_k_heap, (-similarity, word))

    def _find_unqueried_neighbors(self, seed_word):
        """使用FAISS查找一个种子词的未被查询过的近邻。"""
        # 编码种子词并进行L2归一化
        # vector = self.model.encode([seed_word])
        vector = self.embedding_map[seed_word]
        vector = vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(vector)

        # 扩大搜索范围以过滤已查询的词
        search_k = self.faiss_search_size + len(self.queried)
        distances, indices = self.faiss_index.search(vector, search_k)

        # 过滤掉无效索引和已查询过的词
        candidates = [
            self.word_list[i] for i in indices[0]
            if i >= 0 and self.word_list[i] not in self.queried
        ]
        return candidates[:self.faiss_search_size]  # 返回指定数量的有效候选
        # try:
        #     # 编码种子词并进行L2归一化
        #     # vector = self.model.encode([seed_word])
        #     vector = self.embedding_map[seed_word]
        #     vector = vector.astype('float32').reshape(1, -1)
        #     faiss.normalize_L2(vector)
        #
        #     # 扩大搜索范围以过滤已查询的词
        #     search_k = self.faiss_search_size + len(self.queried)
        #     distances, indices = self.faiss_index.search(vector, search_k)
        #
        #     # 过滤掉无效索引和已查询过的词
        #     candidates = [
        #         self.word_list[i] for i in indices[0]
        #         if i >= 0 and self.word_list[i] not in self.queried
        #     ]
        #     return candidates[:self.faiss_search_size]  # 返回指定数量的有效候选
        # except Exception as e:
        #     print(f"FAISS搜索期间发生错误: {e}")
        #     return []

    def _get_next_candidate(self):
        """根据策略生成下一个要查询的候选词。"""
        # 1. 如果堆为空（初始阶段或所有种子都耗尽），则从整个词库中随机选一个
        if not self.top_k_heap:
            # 从未查询过的词中随机选择一个
            unqueried_words = list(self.word_set - set(self.queried.keys()))
            if not unqueried_words: return None  # 所有词都查完了
            return random.choice(unqueried_words)

        # 2. 从堆中获取Top-K相似度的词作为种子池
        top_k_seeds = heapq.nsmallest(self.search_window_size, self.top_k_heap)

        # 3. 随机选择一个种子并寻找其近邻
        random.shuffle(top_k_seeds)  # 打乱种子以增加随机性
        for neg_sim, seed_word in top_k_seeds:
            neighbors = self._find_unqueried_neighbors(seed_word)
            if neighbors:
                # 策略：选择最近的那个未被查询过的邻居
                return random.choice(neighbors[:3])

        # 4. 如果Top-K种子的所有近邻都已被查询，则回退到随机选择策略
        unqueried_words = list(self.word_set - set(self.queried.keys()))
        if not unqueried_words: return None
        return random.choice(unqueried_words)

    def run(self, judge):
        """执行完整的搜索流程。"""
        progress_bar = tqdm(total=self.max_queries, desc="搜索进度", unit="次查询")
        best_word, best_similarity = "", 0.0

        for i in range(self.max_queries):
            # 获取下一个候选词
            candidate = self._get_next_candidate()
            if candidate is None:
                print("所有词汇已查询完毕。")
                break

            # 查询
            similarity = judge.query(candidate)

            # 更新状态
            self._update_state(candidate, similarity)

            # 更新最佳结果
            if similarity > best_similarity:
                best_similarity = similarity
                best_word = candidate

            # 更新进度条显示
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"最佳: '{best_word}' ({best_similarity:.4f}) | 当前: '{candidate}' ({similarity:.4f})")

            # 检查是否成功
            if similarity == 1.0:
                print(f"\n🎉 成功！在第 {i + 1} 次查询时找到目标词！")
                progress_bar.total = i + 1  # 将进度条总数设为当前次数
                progress_bar.refresh()
                break

        progress_bar.close()
        self._show_final_result()

    def _show_final_result(self):
        """显示最终的搜索结果统计。"""
        if not self.queried:
            print("\n搜索结束，无任何查询记录。")
            return

        # 从记录中找到最终的最佳匹配
        final_best_word = max(self.queried, key=self.queried.get)
        final_best_similarity = self.queried[final_best_word]

        print(f"\n🔍 搜索完成。")
        print(f"最终结果: 最佳匹配词 '{final_best_word}' (相似度: {final_best_similarity:.4f})")
        print(f"总查询次数: {len(self.queried)}")

        # 显示Top-10结果
        sorted_items = sorted(self.queried.items(), key=lambda x: x[1], reverse=True)
        print("\n📊 相似度TOP 10记录:")
        for i, (word, sim) in enumerate(sorted_items[:10], 1):
            print(f"  {i:2d}. {word:<15s} | 相似度: {sim:.4f}")


# 主程序
if __name__ == "__main__":
    # 加载FAISS索引和词汇表
    try:
        print("正在加载 FAISS 索引和词汇表...")
        import os
        data_path = "data/tight"
        faiss_path = os.path.join(data_path, "vocal_list.faiss")
        vocab_list_path = os.path.join(data_path, "vocal_list.pkl")
        vocab_embedding_path = os.path.join(data_path, "word_embedding_map.pkl")

        if not os.path.exists(faiss_path):
            print(f"Error: {faiss_path} not found.")
            print("请确保 'data' 文件夹存在且包含预计算的 FAISS 索引文件。您可能需要先运行一个数据准备脚本来生成这些文件。退出。")
            exit()
        if not os.path.exists(vocab_list_path):
            print(f"Error: {vocab_list_path} not found.")
            print("请确保 'data' 文件夹存在且包含词汇列表文件。退出。")
            exit()

        faiss_index = faiss.read_index(faiss_path)
        with open(vocab_list_path, "rb") as f:
            word_list = pickle.load(f)

        with open(vocab_embedding_path, "rb") as f:
            embedding_map = pickle.load(f)

    except Exception as e:
        print(f"Error loading data files: {e}")
        print("请检查文件路径和文件格式。退出。")
        exit()


    # 初始化判题器（设置目标词）
    use_local_judge = False
    if use_local_judge:
        # 共享资源加载
        try:
            print("正在加载 SentenceTransformer 模型...")
            # model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
            model_name = 'shibing624/text2vec-base-chinese'
            try:
                model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Error loading local model '{model_name}': {e}")
                print(f"Attempting to download and load model '{model_name}'...")
                try:
                    model = SentenceTransformer(model_name)  # Re-attempt download
                    print("Model downloaded and loaded successfully.")
                except Exception as download_e:
                    print(f"Error downloading/loading model '{model_name}': {download_e}")
                    print("请检查网络连接或模型名称。退出。")
                    exit()

            print("模型加载完成。")
        except Exception as e:
            print(f"Unexpected error during model loading: {e}")
            print("请检查 SentenceTransformer 库安装 (pip install sentence-transformers)。退出。")
            exit()

        judge = LocalJudge("刑警", model, word_list) # 使用本地判题器调试
    else:
        judge = OnlineJudge() # 使用在线判题器

    # 初始化搜索器
    # max_queries: 总查询次数上限
    # run_initial_exploration: 是否执行初始探索
    searcher = Searcher(
        faiss_index=faiss_index,
        word_list=word_list,
        embedding_map=embedding_map,
        max_queries=500, # 可以根据需要调整总查询次数
    )

    # 执行搜索
    searcher.run(judge)