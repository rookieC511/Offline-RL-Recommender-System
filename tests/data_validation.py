import json
import os
from dateutil.parser import parse
import numpy as np
import csv
from collections import defaultdict
import random

# --- 配置数据目录 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 假设此脚本与数据文件夹在同一级或其父级
PROJECT_DIR = os.path.join(BASE_DIR, "project/")
ENTRY_DIR = os.path.join(BASE_DIR, "entry/")
WORKER_QUALITY_FILE = os.path.join(BASE_DIR, "worker_quality.csv")
PROJECT_LIST_FILE = os.path.join(BASE_DIR, "project_list.csv")


# --- 数据加载函数 (与你训练脚本中的类似，但为了独立验证，重新实现) ---
def load_project_list():
    project_ids_and_required = {}
    if not os.path.exists(PROJECT_LIST_FILE):
        print(f"错误: 未找到 project_list.csv at {PROJECT_LIST_FILE}")
        return project_ids_and_required
    try:
        with open(PROJECT_LIST_FILE, "r", encoding='utf-8') as file:
            lines = file.readlines()
            for line_idx, line in enumerate(lines):
                if line_idx == 0: continue  # 跳过表头
                parts = line.strip('\n').split(',')
                try:
                    project_ids_and_required[int(parts[0])] = int(parts[1])
                except (IndexError, ValueError) as e:
                    print(f"警告: 解析 project_list.csv 第 {line_idx + 1} 行失败: '{line.strip()}' - {e}")
    except Exception as e:
        print(f"读取 project_list.csv 出错: {e}")
    return project_ids_and_required


def load_project_info(project_list_data_map):
    project_data_map = {}
    industry_map_local = {}
    industry_counter_local = 0
    if not os.path.exists(PROJECT_DIR):
        print(f"错误: 项目目录 {PROJECT_DIR} 未找到。")
        return project_data_map, industry_map_local

    project_files = [f for f in os.listdir(PROJECT_DIR) if f.startswith("project_") and f.endswith(".txt")]
    print(f"找到 {len(project_files)} 个项目文件。")

    loaded_count = 0
    for project_filename in project_files:
        if project_filename == ".DS_Store": continue
        try:
            project_id_str = project_filename.replace("project_", "").replace(".txt", "")
            project_id = int(project_id_str)

            if project_id not in project_list_data_map:
                # print(f"信息: 项目 {project_id} 不在 project_list.csv 中，跳过。")
                continue

            with open(os.path.join(PROJECT_DIR, project_filename), "r", encoding='utf-8') as file:
                text = json.load(file)

            project_data_map[project_id] = {
                "id": project_id,
                "sub_category": int(text.get("sub_category", -1)),
                "category": int(text.get("category", -1)),
                "start_date_dt": parse(text.get("start_date", "1970-01-01T00:00:00Z")),  # 提供默认值
                "deadline_dt": parse(text.get("deadline", "1970-01-01T00:00:00Z")),
                "total_awards": float(text.get("total_awards", 0.0)),
                "status": text.get("status", "unknown").lower(),
                "required_answers": project_list_data_map.get(project_id, 1),
                "json_entry_count": int(text.get("entry_count", 0))  # 项目文件中的 entry_count
            }
            industry_str = text.get("industry", "unknown_industry")
            if industry_str not in industry_map_local:
                industry_map_local[industry_str] = industry_counter_local
                industry_counter_local += 1
            project_data_map[project_id]["industry_id"] = industry_map_local[industry_str]
            loaded_count += 1
        except Exception as e:
            print(f"警告: 处理项目文件 {project_filename} 出错: {e}")
    print(f"成功加载 {loaded_count} 个项目信息。")
    return project_data_map, industry_map_local


def load_entry_info(project_ids_available):  # 只加载在project_info中存在的项目的entry
    entry_data_map = {}
    if not os.path.exists(ENTRY_DIR):
        print(f"错误: Entry目录 {ENTRY_DIR} 未找到。")
        return entry_data_map

    print(f"开始加载 {len(project_ids_available)} 个项目的Entry数据...")
    loaded_entry_projects = 0
    total_entries_loaded = 0

    for project_id in project_ids_available:
        entry_data_map[project_id] = {}
        page_k = 0
        project_has_entries = False
        while True:
            entry_filename = os.path.join(ENTRY_DIR, f"entry_{project_id}_{page_k}.txt")
            if not os.path.exists(entry_filename):
                if page_k == 0 and not project_has_entries:  # 如果第一页就不存在
                    # print(f"信息: 项目 {project_id} 没有找到 entry_{project_id}_0.txt 文件。")
                    pass
                break

            project_has_entries = True
            try:
                with open(entry_filename, "r", encoding='utf-8') as efile:
                    entry_text_data = json.load(efile)

                current_page_entries = 0
                for item in entry_text_data.get("results", []):
                    entry_number = int(item["entry_number"])
                    score = 0  # 默认分数
                    if item.get("revisions") and isinstance(item["revisions"], list) and len(item["revisions"]) > 0:
                        score = item["revisions"][0].get("score", 0)

                    entry_data_map[project_id][entry_number] = {
                        "entry_created_at_dt": parse(item.get("entry_created_at", "1970-01-01T00:00:00Z")),
                        "worker_id": int(item["author"]),
                        "withdrawn": item.get("withdrawn", False),
                        "award_value": item.get("award_value"),  # 保留原始值 (可能是None, 数字或字符串)
                        "score": score,  # 添加score字段
                        "winner": item.get("winner", False)  # 添加winner字段
                    }
                    current_page_entries += 1
                    total_entries_loaded += 1
                # print(f"  项目 {project_id} 页 {page_k//24}: 加载 {current_page_entries} 条 entries。")
            except Exception as e_entry:
                print(f"警告: 读取/解析条目文件 {entry_filename} 出错: {e_entry}")
            page_k += 24
        if project_has_entries:
            loaded_entry_projects += 1
    print(f"为 {loaded_entry_projects} 个项目加载了 Entry 数据，总计 {total_entries_loaded} 条 Entries。")
    return entry_data_map


# --- 主要验证和分析逻辑 ---
if __name__ == "__main__":
    print("开始数据验证和分析脚本...")

    # 1. 加载数据
    project_list = load_project_list()
    if not project_list:
        print("错误: project_list.csv 加载失败或为空，无法继续。")
        sys.exit(1)

    projects, industries = load_project_info(project_list)
    if not projects:
        print("错误: 项目信息加载失败或为空，无法继续。")
        sys.exit(1)

    entries = load_entry_info(projects.keys())  # 只加载在projects中存在的项目的entry
    if not entries:
        print("警告: Entry信息加载失败或为空。部分后续分析可能受影响。")

    print("\n--- 数据加载摘要 ---")
    print(f"项目列表中的项目数: {len(project_list)}")
    print(f"成功加载的项目信息数: {len(projects)}")
    print(f"识别出的行业种类数: {len(industries)}")
    print(f"加载了Entry信息的项目数: {len(entries)}")
    total_entry_count_from_dict = sum(len(v) for v in entries.values())
    print(f"从文件加载的总Entry条目数: {total_entry_count_from_dict}")

    # 2. award_value 和 score 的统计分析
    all_award_values_numeric = []
    all_scores = []
    entries_with_award_field = 0
    entries_with_positive_award = 0
    entries_marked_winner = 0
    entries_winner_with_award = 0
    entries_winner_without_award = 0

    if entries:
        for project_id, entries_in_project in entries.items():
            for entry_number, entry_data in entries_in_project.items():
                if entry_data.get("award_value") is not None:
                    entries_with_award_field += 1
                    try:
                        award_val = float(entry_data["award_value"])
                        if award_val > 0:
                            all_award_values_numeric.append(award_val)
                            entries_with_positive_award += 1
                            if entry_data.get("winner") is True:
                                entries_winner_with_award += 1
                        elif entry_data.get("winner") is True:  # award_value是0或负数但标记为winner
                            entries_winner_without_award += 1
                    except (ValueError, TypeError):
                        # print(f"警告: 项目 {project_id}, Entry {entry_number} 的 award_value ('{entry_data.get('award_value')}') 不是有效的数字。")
                        if entry_data.get("winner") is True:
                            entries_winner_without_award += 1
                elif entry_data.get("winner") is True:  # award_value是None但标记为winner
                    entries_winner_without_award += 1

                if entry_data.get("winner") is True:
                    entries_marked_winner += 1

                all_scores.append(entry_data.get("score", 0))  # 假设缺失score为0

    print("\n--- 'award_value' 字段统计 ---")
    print(f"包含 'award_value' 字段的Entry数量: {entries_with_award_field}")
    print(f"包含有效正向奖励 (award_value > 0) 的Entry数量: {entries_with_positive_award}")
    if all_award_values_numeric:
        awards_arr = np.array(all_award_values_numeric)
        print(
            f"  正向奖励统计: Min={np.min(awards_arr):.2f}, Max={np.max(awards_arr):.2f}, Mean={np.mean(awards_arr):.2f}, Median={np.median(awards_arr):.2f}")
        print(f"  正向奖励75百分位: {np.percentile(awards_arr, 75):.2f}, 95百分位: {np.percentile(awards_arr, 95):.2f}")
    else:
        print("  未找到数值大于0的 award_value。")

    print("\n--- 'winner' 字段统计 ---")
    print(f"标记为 'winner: true' 的Entry数量: {entries_marked_winner}")
    print(f"  其中 'winner: true' 且 'award_value > 0' 的数量: {entries_winner_with_award}")
    print(f"  其中 'winner: true' 但 'award_value' 不是正数或为None的数量: {entries_winner_without_award}")

    print("\n--- 'score' 字段统计 (所有 Revisions 中的第一个 Score) ---")
    if all_scores:
        scores_arr = np.array(all_scores)
        print(f"评分样本数量 (包括0分): {len(scores_arr)}")
        print(
            f"  评分统计: Min={np.min(scores_arr):.2f}, Max={np.max(scores_arr):.2f}, Mean={np.mean(scores_arr):.2f}, Median={np.median(scores_arr):.2f}")
        non_zero_scores = scores_arr[scores_arr > 0]
        if len(non_zero_scores) > 0:
            print(f"  非零评分样本数量: {len(non_zero_scores)}")
            print(
                f"  非零评分统计: Min={np.min(non_zero_scores):.2f}, Max={np.max(non_zero_scores):.2f}, Mean={np.mean(non_zero_scores):.2f}, Median={np.median(non_zero_scores):.2f}")
            print(
                f"  非零评分75百分位: {np.percentile(non_zero_scores, 75):.2f}, 95百分位: {np.percentile(non_zero_scores, 95):.2f}")
            # 查看高分段的分布
            for i in range(int(np.max(non_zero_scores)), 0, -1):
                count = np.sum(non_zero_scores == i)
                if count > 0: print(
                    f"    得分为 {i} 的数量: {count} (占非零评分的 {count / len(non_zero_scores) * 100:.2f}%)")

        else:
            print("  未找到非零评分。")
    else:
        print("  未收集到评分数据。")

    # 3. 检查项目信息中的 total_awards 与 entry 中 award_value 的关系
    print("\n--- 项目总奖励与Entry获奖金额对比抽样 ---")
    awarded_project_count = 0
    if projects and entries and all_award_values_numeric:  # 确保有数据可供分析
        for pid in random.sample(list(projects.keys()), min(10, len(projects))):  # 抽样10个项目
            p_info = projects.get(pid)
            p_total_awards = p_info.get("total_awards", 0.0)
            entries_for_this_project = entries.get(pid, {})
            actual_sum_of_awards_in_entries = 0
            num_winners_in_entries = 0
            has_award_in_entry = False

            for _, e_data in entries_for_this_project.items():
                award_val = e_data.get("award_value")
                if award_val is not None:
                    try:
                        val = float(award_val)
                        if val > 0:
                            actual_sum_of_awards_in_entries += val
                            num_winners_in_entries += 1
                            has_award_in_entry = True
                    except (ValueError, TypeError):
                        pass

            if has_award_in_entry:  # 只打印那些在entry中有明确奖励的项目
                awarded_project_count += 1
                print(f"项目ID: {pid}")
                print(f"  project_info 中的 total_awards: {p_total_awards}")
                print(f"  entry_info 中该项目下所有正 award_value 之和: {actual_sum_of_awards_in_entries}")
                print(f"  entry_info 中该项目下获奖的 entry 数量: {num_winners_in_entries}")
                print(f"  project_list.csv 中该项目 required_answers: {project_list.get(pid)}")
                if p_total_awards > 0 and num_winners_in_entries > 0 and p_total_awards != actual_sum_of_awards_in_entries:
                    print(f"  注意: 项目总奖励与Entry中奖励之和不完全匹配!")
                print("-" * 20)
        print(f"抽样检查了 {awarded_project_count} 个在Entry中有明确奖励的项目。")

    print("\n脚本执行完毕。请仔细检查上面的统计输出。")