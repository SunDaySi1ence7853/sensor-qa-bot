"""
检索质量评估脚本 (任务四)

用法：
  python scripts/eval_retrieval.py

功能：
  1. 读取 eval/retrieval_questions.json 评测集
  2. 调用向量库 similarity_search 检索 top-k 文档
  3. 对比期望文档，输出总命中率、分类命中率、未命中清单
  4. 统计平均检索耗时
"""
import json
import time
from pathlib import Path
from collections import defaultdict

# 将项目根目录加入 sys.path，确保能导入 src 模块
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.vectorstore import load_vectorstore
from src.logging_config import get_logger

logger = get_logger(__name__)

# 评估参数
TOP_K = 5  # 检索 top-k 文档，可根据实际情况调整
EVAL_FILE = PROJECT_ROOT / "eval" / "retrieval_questions.json"


def evaluate():
    if not EVAL_FILE.exists():
        logger.error(f"评测集文件不存在: {EVAL_FILE}")
        return

    # 1. 加载评测集
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    logger.info(f"成功加载评测集，共 {len(eval_data)} 条问题。开始检索评估 (top_k={TOP_K})...")

    # 2. 加载向量库
    try:
        vectorstore = load_vectorstore()
    except Exception as e:
        logger.error(f"向量库加载失败，请先运行 build 命令。错误: {e}")
        return

    # 3. 开始评估
    total_questions = len(eval_data)
    hit_count = 0
    total_time = 0.0
    
    # 分类统计
    category_stats = defaultdict(lambda: {"total": 0, "hit": 0})
    missed_questions = []

    for i, item in enumerate(eval_data, 1):
        question = item.get("question")
        expected_sources = item.get("expectedsources", [])
        category = item.get("category", "未分类")

        category_stats[category]["total"] += 1

        start_time = time.time()
        
        # 执行检索 (只检索，不生成)
        try:
            docs = vectorstore.similarity_search(question, k=TOP_K)
        except Exception as e:
            logger.error(f"第 {i} 题检索失败: {question}，错误: {e}")
            missed_questions.append({
                "question": question,
                "expected": expected_sources,
                "got": [],
                "reason": f"检索异常: {e}"
            })
            continue

        end_time = time.time()
        search_time = end_time - start_time
        total_time += search_time

        # 提取检索到的文件名
        # 你的 vectorstore 中 source 是绝对路径，这里取 Path().name 转为文件名比对
        retrieved_sources = [Path(doc.metadata.get("source", "")).name for doc in docs]

        # 判断是否命中 (期望的文档是否在检索到的 top-k 中)
        is_hit = any(exp in retrieved_sources for exp in expected_sources)

        if is_hit:
            hit_count += 1
            category_stats[category]["hit"] += 1
            logger.info(f"[{i}/{total_questions}] ✅ 命中 | 耗时: {search_time:.2f}s | 问题: {question[:30]}...")
        else:
            logger.warning(f"[{i}/{total_questions}] ❌ 未命中 | 耗时: {search_time:.2f}s | 问题: {question[:30]}...")
            missed_questions.append({
                "question": question,
                "expected": expected_sources,
                "got": retrieved_sources
            })

    # 4. 输出评估报告
    print("\n" + "="*50)
    print("🔥 检索质量评估报告 🔥")
    print("="*50)
    
    overall_hit_rate = (hit_count / total_questions) * 100 if total_questions > 0 else 0
    avg_time = (total_time / total_questions) if total_questions > 0 else 0

    print(f"\n总问题数: {total_questions}")
    print(f"命中数: {hit_count}")
    print(f"总命中率: {overall_hit_rate:.2f}%")
    print(f"平均检索耗时: {avg_time:.3f}s (不含大模型生成时间)")

    print("\n--- 分类命中率 ---")
    for cat, stats in category_stats.items():
        cat_rate = (stats["hit"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"[{cat}] {stats['hit']}/{stats['total']} = {cat_rate:.2f}%")

    print("\n--- 未命中清单 ---")
    if missed_questions:
        for miss in missed_questions:
            print(f"❓ 问题: {miss['question']}")
            print(f"   期望来源: {miss['expected']}")
            print(f"   实际检索到: {miss['got']}")
            if 'reason' in miss:
                print(f"   失败原因: {miss['reason']}")
    else:
        print("无未命中问题，全部命中！")

    print("="*50 + "\n")

    # 保存报告到文件，方便后续截图交差
    report_path = PROJECT_ROOT / "eval" / "retrieval_eval_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        # 简单地将控制台输出重定向写入文件 (偷懒但有效)
        import io
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        # 重新打印一次
        print(f"总命中率: {overall_hit_rate:.2f}%")
        print(f"平均耗时: {avg_time:.3f}s")
        for cat, stats in category_stats.items():
            print(f"[{cat}] {stats['hit']}/{stats['total']}")
        for miss in missed_questions:
            print(f"未命中: {miss['question']} | 期望: {miss['expected']} | 实际: {miss['got']}")
        sys.stdout = old_stdout
        f.write(buffer.getvalue())

    logger.info(f"评估报告已保存至: {report_path}")


if __name__ == "__main__":
    evaluate()