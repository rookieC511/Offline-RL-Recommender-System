import json
import time
import datetime
import math
import csv
import copy
import sys
import os
from dateutil.parser import parse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, IterableDataset
from collections import namedtuple # 确保这行 import 存在

# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- 常量和配置 ---
ACTION_SIZE = 1
##GAMMA = 0.99
LEARNING_RATE =3e-5
BUFFER_SIZE = int(5e4)
BATCH_SIZE = 64
##TAU = 1e-3
##UPDATE_EVERY = 4
REWARD_SCALING_FACTOR = 10  # 定义一个奖励缩放因子
##TARGET_UPDATE_EVERY = 100
# --- DPO专属配置 ---
DPO_BETA = 0.3  # DPO的beta超参数，控制策略与隐式参考策略的偏离程度。0.1是常用初始值。
NUM_REJECTED_SAMPLES = 3 # <-- 新增：为每个chosen样本匹配3个rejected样本
CATEGORY_EMBED_DIM = 5
SUB_CATEGORY_EMBED_DIM = 8
INDUSTRY_EMBED_DIM = 5
WORKER_ID_EMBED_DIM = 8
# <--- 关键修改: 为新特征更新常量 ---
##CQL_ALPHA = 0.15  # CQL损失的权重。这是一个需要调优的超参数，可以从0.5, 1.0, 2.0, 5.0等开始尝试。
NUM_NUMERIC_WORKER_FEATURES = 5
NUM_NUMERIC_BASE_PROJECT_FEATURES = 4
NUM_NUMERIC_INTERACTION_FEATURES = 4 # 新增的“工人类别历史表现”特征
NUM_NUMERIC_CONTEXT_FEATURES = 1
TOTAL_NUMERIC_FEATURES = NUM_NUMERIC_WORKER_FEATURES + NUM_NUMERIC_BASE_PROJECT_FEATURES + NUM_NUMERIC_INTERACTION_FEATURES + NUM_NUMERIC_CONTEXT_FEATURES


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备: {device}")

all_begin_time_dt = parse("2018-01-01T0:0:0Z")
feature_min_max = {}
REWARD_SCALE_REFERENCE = 20.0  # 先给一个默认值，会被后续分析覆盖

# --- 数据加载 ---
worker_quality_map = {}
project_list_data = {}
project_info = {}
entry_info = {}
industry_map = {}
industry_counter = 0
try:
    with open("worker_quality.csv", "r", encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for line in csvreader:
            try:
                worker_id, quality_str = line[0], line[1]
                quality = float(quality_str)
                if quality > 0.0: worker_quality_map[int(worker_id)] = quality / 100.0
            except (ValueError, IndexError):
                pass  # 静默处理格式错误行
except FileNotFoundError:
    print("警告: 未找到 worker_quality.csv。")
try:
    with open("project_list.csv", "r", encoding='utf-8') as file:
        project_list_lines = file.readlines()
        for line in project_list_lines[1:]:
            parts = line.strip('\n').split(',')
            try:
                project_list_data[int(parts[0])] = int(parts[1])
            except (IndexError, ValueError):
                pass
except FileNotFoundError:
    print("警告: 未找到 project_list.csv。")
project_dir, entry_dir = "project/", "entry/"
project_files = [f for f in os.listdir(project_dir) if
                 f.startswith("project_") and f.endswith(".txt")] if os.path.exists(project_dir) else []
for project_filename in project_files:
    if project_filename == ".DS_Store": continue
    try:
        project_id = int(project_filename.replace("project_", "").replace(".txt", ""))
        if project_id not in project_list_data: continue
        with open(os.path.join(project_dir, project_filename), "r", encoding='utf-8') as file:
            text = json.load(file)
        project_info[project_id] = {"id": project_id, "sub_category": int(text.get("sub_category", -1)),
                                    "category": int(text.get("category", -1)),
                                    "start_date_dt": parse(text["start_date"]), "deadline_dt": parse(text["deadline"]),
                                    "total_awards": float(text.get("total_awards", 0.0)),
                                    "status": text.get("status", "unknown"),
                                    "required_answers": project_list_data.get(project_id, 1)}
        industry_str = text.get("industry", "unknown_industry");
        if industry_str not in industry_map: industry_map[industry_str] = industry_counter; industry_counter += 1
        project_info[project_id]["industry_id"] = industry_map[industry_str]
        entry_info[project_id] = {}
        page_k = 0
        while True:
            entry_filename = os.path.join(entry_dir, f"entry_{project_id}_{page_k}.txt")
            if not os.path.exists(entry_filename): break
            try:
                with open(entry_filename, "r", encoding='utf-8') as efile:
                    entry_text_data = json.load(efile)
                for item in entry_text_data.get("results", []):                 score_val = 0 # 默认分数
                if item.get("revisions") and isinstance(item["revisions"], list) and len(item["revisions"]) > 0:
                    # 确保 revisions[0] 是一个字典并且包含 "score"
                    if isinstance(item["revisions"][0], dict):
                        score_val = item["revisions"][0].get("score", 0) # 如果没有score键，默认为0

                entry_info[project_id][int(item["entry_number"])] = {
                    "entry_created_at_dt": parse(item.get("entry_created_at", "1970-01-01T00:00:00Z")), # 添加默认值以防万一
                    "worker_id": int(item["author"]),
                    "withdrawn": item.get("withdrawn", False),
                    "award_value": item.get("award_value"),
                    "score": score_val, # <--- 新增或确保正确赋值
                    "winner": item.get("winner", False) # <--- 同时确保winner也被加载
                }
            except Exception:
                pass  # 静默处理
            page_k += 24
    except Exception:
        pass  # 静默处理
# 示例性的数据加载结束
print(
    f"数据读取完成。项目数: {len(project_info)}, Entry项目数: {len(entry_info)}, 工人质量记录数: {len(worker_quality_map)}, 行业数: {len(industry_map if industry_map else [])}")

# --- 历史奖励统计分析 ---
# ... (粘贴历史奖励统计分析代码，它可以修改全局的 REWARD_SCALE_REFERENCE) ...
print("\n--- 开始历史奖励 (award_value > 0) 统计分析 ---")
all_historical_awards = []  # ... (其余统计代码如前一个回复所示) ...
# ... (确保 REWARD_SCALE_REFERENCE 在这里被正确赋值) ...
if entry_info:
    for project_id, entries_in_project in entry_info.items():
        for entry_number, entry_data in entries_in_project.items():
            award_val_raw = entry_data.get("award_value")
            if award_val_raw is not None:
                try:
                    award_val_float = float(award_val_raw)
                    if award_val_float > 0: all_historical_awards.append(award_val_float)
                except (ValueError, TypeError):
                    pass
if all_historical_awards:
    awards_array = np.array(all_historical_awards)
    REWARD_SCALE_REFERENCE = np.median(awards_array)
    if REWARD_SCALE_REFERENCE <= 0: REWARD_SCALE_REFERENCE = np.mean(awards_array) if np.mean(
        awards_array) > 0 else 20.0
    print(f"建议用于奖励缩放的参考值 (REWARD_SCALE_REFERENCE): {REWARD_SCALE_REFERENCE:.2f}")
else:
    print("数据集中未找到有效的正向历史奖励。使用默认缩放参考。")
    REWARD_SCALE_REFERENCE = 20.0  # 保持默认
print("--- 历史奖励统计分析结束 ---\n")
# --- 在数据加载后，进行新的预计算 ---
print("\n--- 开始构建丰富的工人全局及交互特征 ---")

# 用于存储全局画像
worker_global_stats = {} 
# worker_global_stats[worker_id] = {'total_score': float, 'count': int, 'wins': int, 'categories': set()}

# 用于存储交互特征 (升级版)
worker_cat_performance = {} 
# worker_cat_performance[(worker_id, cat_id)] = {'total_score': float, 'count': int, 'wins': int}

if entry_info:
    for proj_id, entries in entry_info.items():
        proj_details = project_info.get(proj_id)
        if not proj_details: continue
        
        proj_cat = proj_details.get("category")
        if proj_cat is None: continue

        for _, entry_data in entries.items():
            worker_id = entry_data["worker_id"]
            score = entry_data.get("score", 0)
            is_winner = entry_data.get("winner", False)

            # --- 更新交互特征 ---
            key = (worker_id, proj_cat)
            if key not in worker_cat_performance:
                worker_cat_performance[key] = {'total_score': 0.0, 'count': 0, 'wins': 0}
            
            worker_cat_performance[key]['total_score'] += score
            worker_cat_performance[key]['count'] += 1
            if is_winner:
                worker_cat_performance[key]['wins'] += 1

            # --- 更新全局画像 ---
            if worker_id not in worker_global_stats:
                worker_global_stats[worker_id] = {'total_score': 0.0, 'count': 0, 'wins': 0, 'categories': set()}

            worker_global_stats[worker_id]['total_score'] += score
            worker_global_stats[worker_id]['count'] += 1
            worker_global_stats[worker_id]['categories'].add(proj_cat)
            if is_winner:
                worker_global_stats[worker_id]['wins'] += 1

print(f"完成。共记录了 {len(worker_global_stats)} 位工人的全局画像和 {len(worker_cat_performance)} 条交互表现。")
# --- 动态确定分类特征的基数 ---
# ... (粘贴动态确定基数代码，确保 NUM_..._EMBED_SIZE 被正确赋值) ...
max_cat_id, max_sub_cat_id, max_worker_id, max_industry_id = 0, 0, 0, 0
if project_info: max_cat_id = max(
    p["category"] for p in project_info.values() if "category" in p); max_sub_cat_id = max(
    p["sub_category"] for p in project_info.values() if "sub_category" in p); max_industry_id = max(
    p["industry_id"] for p in project_info.values() if "industry_id" in p)
if entry_info: all_workers_set = set(
    ed["worker_id"] for entries in entry_info.values() for ed in entries.values()); max_worker_id = max(
    all_workers_set) if all_workers_set else 0
NUM_CATEGORIES_EMBED_SIZE = max_cat_id + 1;
NUM_SUB_CATEGORIES_EMBED_SIZE = max_sub_cat_id + 1;
NUM_INDUSTRIES_EMBED_SIZE = max_industry_id + 1;
NUM_WORKERS_EMBED_SIZE = max_worker_id + 1
print(
    f"嵌入层大小: Cat={NUM_CATEGORIES_EMBED_SIZE}, SubCat={NUM_SUB_CATEGORIES_EMBED_SIZE}, Ind={NUM_INDUSTRIES_EMBED_SIZE}, Worker={NUM_WORKERS_EMBED_SIZE}")

# --- 特征归一化：预计算 Min/Max ---
# ... (粘贴 precompute_feature_min_max 和 min_max_scale 函数定义) ...
def get_available_projects(current_time_dt, project_info_map, entry_info_map):  # 确保此函数已定义
    available = []
    for pid, p_data in project_info_map.items():
        if p_data.get("start_date_dt") <= current_time_dt and p_data.get(
                "deadline_dt") > current_time_dt and p_data.get("status", "open").lower() != "completed":
            accepted_count = 0
            if pid in entry_info_map:
                for _, e_data in entry_info_map[pid].items():
                    if not e_data.get("withdrawn", False) and e_data.get(
                        "entry_created_at_dt") <= current_time_dt: accepted_count += 1
            if accepted_count < p_data.get("required_answers", 1): available.append(pid)
    return available


def precompute_feature_min_max(arrival_events, proj_info, ent_info, wq_map):  # precompute 函数定义
    global feature_min_max;
    print("开始预计算特征 Min/Max...")
    data_to_scale = {name: [] for name in
                     ["worker_quality", "time_until_deadline_sec", "task_age_sec", "project_duration_sec",
                      "reward_per_slot", "current_time_val"]}
    for event in arrival_events:
        worker_id, current_time = event['worker_id'], event['arrival_time_dt']
        data_to_scale["worker_quality"].append(wq_map.get(worker_id, 0.0))
        data_to_scale["current_time_val"].append((current_time - all_begin_time_dt).total_seconds())
        available_p_ids = get_available_projects(current_time, proj_info, ent_info)
        for p_id in available_p_ids:
            p_data = proj_info.get(p_id)
            if p_data:
                data_to_scale["time_until_deadline_sec"].append((p_data["deadline_dt"] - current_time).total_seconds())
                data_to_scale["task_age_sec"].append((current_time - p_data["start_date_dt"]).total_seconds())
                dur_sec = (p_data["deadline_dt"] - p_data["start_date_dt"]).total_seconds();
                data_to_scale["project_duration_sec"].append(max(0, dur_sec))
                rps = (p_data["total_awards"] / p_data["required_answers"]) if p_data.get("required_answers",
                                                                                          0) > 0 and p_data.get(
                    "total_awards", 0) > 0 else 0
                data_to_scale["reward_per_slot"].append(rps)
    for name, values in data_to_scale.items():
        if values:
            min_v, max_v = np.min(values), np.max(values)
            if min_v == max_v: max_v = min_v + 1e-6 if min_v == 0 else min_v * 1.01;
            if min_v == max_v: min_v, max_v = 0.0, 1.0
            feature_min_max[name] = (min_v, max_v)
            # print(f"特征 '{name}': Min={min_v:.2f}, Max={max_v:.2f}") # 可以注释掉
        else:
            feature_min_max[name] = (0, 1)
    print("特征 Min/Max 预计算完成。")


def min_max_scale(value, feature_name):  # min_max_scale 函数定义
    min_val, max_val = feature_min_max.get(feature_name, (0, 1))
    if max_val == min_val: return 0.5
    return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)  # 添加clip确保在0-1


# --- 特征工程函数 ---
# ... (粘贴 get_worker_features_simplified, get_project_features_simplified, get_context_features_simplified 函数定义) ...
def get_worker_features_simplified(worker_id, wq_map):  # get_worker_features_simplified 定义
    quality_raw = wq_map.get(worker_id, 0.0)
    quality_scaled = min_max_scale(quality_raw, "worker_quality")
    # 为了调试归一化，可以暂时在这里打印
    #if random.random() < 0.001: # 随机打印一小部分，避免输出过多
         #print(f"DEBUG_WQ: raw_quality={quality_raw:.2f}, scaled_quality={quality_scaled:.4f}")
    return worker_id, np.array([quality_scaled])


def get_project_features_simplified(project_id, proj_data_map, current_time):  # get_project_features_simplified 定义
    p_data = proj_data_map.get(project_id)
    if not p_data: return 0, 0, 0, None
    cat_id, sub_cat_id, ind_id = max(0, p_data.get("category", 0)), max(0, p_data.get("sub_category", 0)), max(0,
                                                                                                               p_data.get(
                                                                                                                   "industry_id",
                                                                                                                   0))
    time_until_deadline_raw = (p_data["deadline_dt"] - current_time).total_seconds()
    task_age_raw = (current_time - p_data["start_date_dt"]).total_seconds()
    project_duration_raw = max(0, (p_data["deadline_dt"] - p_data["start_date_dt"]).total_seconds())
    reward_per_slot_raw = (p_data["total_awards"] / p_data["required_answers"]) if p_data.get("required_answers",
                                                                                              0) > 0 and p_data.get(
        "total_awards", 0) > 0 else 0
    numeric_project_f = np.array(
        [min_max_scale(time_until_deadline_raw, "time_until_deadline_sec"), min_max_scale(task_age_raw, "task_age_sec"),
         min_max_scale(project_duration_raw, "project_duration_sec"),
         min_max_scale(reward_per_slot_raw, "reward_per_slot")])

    # 为了调试归一化
    #if random.random() < 0.001:
         #print(f"DEBUG_PF_RAW: ProjID:{project_id} T_Deadline={time_until_deadline_raw:.0f}, T_Age={task_age_raw:.0f}, T_Dur={project_duration_raw:.0f}, RPS={reward_per_slot_raw:.2f}")
         #print(f"DEBUG_PF_SCA: ProjID:{project_id} Scaled_Feats={np.array2string(numeric_project_f, formatter={'float_kind':lambda x: '%.4f' % x})}")
    return cat_id, sub_cat_id, ind_id, numeric_project_f


def get_context_features_simplified(current_time):
    current_time_raw = (current_time - all_begin_time_dt).total_seconds()
    current_time_scaled_value = min_max_scale(current_time_raw, "current_time_val")  # 先计算缩放后的单个值

    # 为了调试归一化 (可选，如果需要打印)
    #if random.random() < 0.001: # 或者你可以移除这个随机条件，在训练循环外控制打印频率
         #print(f"DEBUG_CF: raw_time={current_time_raw:.0f}, scaled_time={current_time_scaled_value:.4f}")

    return np.array([current_time_scaled_value])  # 将缩放后的单个值放入Numpy数组中返回
# --- 定义新的状态生成函数 ---

def get_new_final_state_tuple(worker_id, project_id, current_time, 
                              proj_info_map, wq_map, global_stats, cat_perf_map):
    
    # --- 1. 获取项目基础特征 (不变) ---
    cat_id, sub_cat_id, ind_id, numeric_project_f_base = get_project_features_simplified(project_id, proj_info_map, current_time)
    if numeric_project_f_base is None: return None

    # --- 2. 获取工人全局画像特征 ---
    #    (这里需要进行归一化，可以预计算min/max，或使用估计值)
    worker_stats = global_stats.get(worker_id, {})
    
    # 原始质量分 (来自worker_quality.csv)
    worker_quality_scaled = min_max_scale(wq_map.get(worker_id, 0.0), "worker_quality") 
    
    # 全局总参与次数 (简单归一化，比如除以50)
    global_participation_count = worker_stats.get('count', 0)
    global_participation_scaled = np.clip(global_participation_count / 50.0, 0, 1)

    # 全局平均分 (0-5分 -> 0-1)
    global_avg_score = (worker_stats.get('total_score', 0) / global_participation_count) if global_participation_count > 0 else 0
    global_avg_score_scaled = global_avg_score / 5.0
    
    # 全局胜率
    global_win_rate = (worker_stats.get('wins', 0) / global_participation_count) if global_participation_count > 0 else 0
    
    # 参与类别的多样性 (简单归一化，比如除以10)
    category_diversity = len(worker_stats.get('categories', set()))
    category_diversity_scaled = np.clip(category_diversity / 10.0, 0, 1)

    numeric_worker_global_f = np.array([
        worker_quality_scaled, 
        global_participation_scaled, 
        global_avg_score_scaled, 
        global_win_rate, 
        category_diversity_scaled
    ])

    # --- 3. 获取工人-类别交互特征 ---
    key = (worker_id, cat_id)
    cat_stats = cat_perf_map.get(key, {})
    
    cat_participation_count = cat_stats.get('count', 0)
    
    # 在该类别下的平均分
    cat_avg_score = (cat_stats.get('total_score', 0) / cat_participation_count) if cat_participation_count > 0 else 0
    cat_avg_score_scaled = cat_avg_score / 5.0
    
    # 在该类别下的参与次数
    cat_participation_scaled = np.clip(cat_participation_count / 10.0, 0, 1) # 除以10归一化
    
    # 在该类别下的胜率
    cat_win_rate = (cat_stats.get('wins', 0) / cat_participation_count) if cat_participation_count > 0 else 0
    
    # 是否是新手 (重要信号)
    is_new_to_category = 1.0 if cat_participation_count == 0 else 0.0

    numeric_interaction_f = np.array([
        cat_avg_score_scaled, 
        cat_participation_scaled, 
        cat_win_rate,
        is_new_to_category
    ])

    # --- 4. 获取上下文特征 (不变) ---
    numeric_context_f = get_context_features_simplified(current_time)

    # --- 5. 拼接所有数值特征 ---
    final_numeric_features = np.concatenate([
        numeric_worker_global_f,
        numeric_project_f_base,
        numeric_interaction_f,
        numeric_context_f
    ])
    
    # 注意：返回值不再包含 worker_id_embed
    return (max(0, cat_id), max(0, sub_cat_id), max(0, ind_id), final_numeric_features)
# 替换掉或者注释掉原来的Experience namedtuple
PreferencePair = namedtuple("PreferencePair", 
                            field_names=["chosen_state_tuple", "rejected_state_tuple"])
# --- QNetwork, ReplayBuffer, DQNAgentWithEmbeddings 类定义 ---
# ... (粘贴你已验证无误的这三个类的完整定义) ...
# (确保 ReplayBuffer.sample 和 DQNAgentWithEmbeddings.act/learn/step 正确处理特征元组)
# (此处粘贴上一回复中的 PolicyNetwork, ReplayBuffer, DQNAgentWithEmbeddings 类定义)
# 在 get_new_final_state_tuple 函数之后添加

def create_preference_dataset_v4(events, proj_info, ent_info, wq_map, global_stats, cat_perf_map):
    """
    【V4最终版】
    放宽 "Chosen" 的标准，以生成足够多的高质量训练数据。
    """
    print(f"开始为 {len(events)} 个事件创建偏好数据集 (V4版本)...")
    preference_data = []
    
    for event in events:
        current_worker_id = event['worker_id']
        current_time_dt = event['arrival_time_dt']
        chosen_project_id = event['project_id']
        chosen_entry_num = event['entry_number']
        
        # --- 核心优化点：放宽 "Chosen" 的标准 ---
        try:
            entry_details = ent_info[chosen_project_id][chosen_entry_num]
            is_winner = entry_details.get("winner", False)
            score = entry_details.get("score", 0)
            
            # 处理 award_value 可能为 None 的情况
            award_val_raw = entry_details.get("award_value")
            has_award = False
            if award_val_raw is not None:
                try:
                    has_award = float(award_val_raw) > 0
                except (ValueError, TypeError):
                    pass # 如果转换失败，则认为没有奖励

        except KeyError:
            continue

        # 使用新的、更宽松的条件
        if not (is_winner or score >= 4 or has_award):
            continue # 如果不是获胜者、高分者或获奖者，则跳过

        # 寻找"Rejected"样本 (逻辑不变)
        available_project_ids = get_available_projects(current_time_dt, proj_info, ent_info)
        rejected_candidates = [pid for pid in available_project_ids if pid != chosen_project_id]
        
        if not rejected_candidates:
            continue

        num_to_sample = min(NUM_REJECTED_SAMPLES, len(rejected_candidates))
        if num_to_sample == 0:
            continue

        sampled_rejected_ids = random.sample(rejected_candidates, k=num_to_sample)
        
        # 为每个配对生成状态元组
        for rejected_project_id in sampled_rejected_ids:
            chosen_state_tuple = get_new_final_state_tuple(
                current_worker_id, chosen_project_id, current_time_dt,
                proj_info, wq_map, global_stats, cat_perf_map
            )
            rejected_state_tuple = get_new_final_state_tuple(
                current_worker_id, rejected_project_id, current_time_dt,
                proj_info, wq_map, global_stats, cat_perf_map
            )
            
            if chosen_state_tuple and rejected_state_tuple:
                pair = PreferencePair(chosen_state_tuple, rejected_state_tuple)
                preference_data.append(pair)
            
    print(f"成功创建 {len(preference_data)} 条偏好数据。")
    return preference_data
class PolicyNetwork(nn.Module):
    """神经网络模型 (Q-Network)"""
    def __init__(self, num_categories_embed_size, num_sub_categories_embed_size,
                 num_industries_embed_size, cat_embed_dim, sub_cat_embed_dim, ind_embed_dim,
                 total_numeric_features, seed, fc1_units=128, fc2_units=32):
        
        super(PolicyNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # 定义嵌入层
        self.category_embedding = nn.Embedding(num_categories_embed_size, cat_embed_dim)
        self.sub_category_embedding = nn.Embedding(num_sub_categories_embed_size, sub_cat_embed_dim)
        self.industry_embedding = nn.Embedding(num_industries_embed_size, ind_embed_dim)
        
        # 计算全连接层的输入维度
        total_embed_dim =cat_embed_dim + sub_cat_embed_dim + ind_embed_dim
        fc_input_dim = total_embed_dim + total_numeric_features
        
        # 定义全连接层和Dropout层
        self.fc1 = nn.Linear(fc_input_dim, fc1_units)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, id_features_batch, numeric_features_batch):
        # 分割ID特征
        cat_ids, sub_cat_ids, ind_ids = id_features_batch.split(1, dim=1)
        
        # 获取嵌入向量
        cat_embed = self.category_embedding(cat_ids.squeeze(-1))
        sub_cat_embed = self.sub_category_embedding(sub_cat_ids.squeeze(-1))
        ind_embed = self.industry_embedding(ind_ids.squeeze(-1))
        
        # 拼接所有特征
        embedded_features = torch.cat([cat_embed, sub_cat_embed, ind_embed], dim=1)
        all_features = torch.cat([embedded_features, numeric_features_batch], dim=1)
        
        # 通过网络层进行前向传播
        x = F.relu(self.fc1(all_features))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


Experience = namedtuple("Experience",
                        field_names=["state_tuple", "action_project_id", "reward", "next_state_options_tuples", "done"])
# 3. 定义全新的 DPOAgent
class DPOAgent:
    """
    使用DPO算法进行学习的智能体。
    """
    def __init__(self, num_categories_embed_size, num_sub_categories_embed_size,
                 num_industries_embed_size, cat_embed_dim, sub_cat_embed_dim, ind_embed_dim,
                 total_numeric_features, seed):
        
        # DPO只有一个策略网络，没有目标网络
        policy_network_base = PolicyNetwork(
            num_categories_embed_size, num_sub_categories_embed_size,
            num_industries_embed_size, cat_embed_dim, sub_cat_embed_dim,
            ind_embed_dim, total_numeric_features, seed
        ).to(device)

        if torch.cuda.device_count() > 1:
            print(f"检测到 {torch.cuda.device_count()} 个GPU，将使用 DataParallel。")
            self.policy_network = nn.DataParallel(policy_network_base)
        else:
            self.policy_network = policy_network_base
            
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LEARNING_RATE)

    def learn(self, batch_of_pairs):
        """
        使用一个批次的偏好对来更新策略网络。
        """
        chosen_states, rejected_states = batch_of_pairs
        
        # 分离ID和数值特征
        chosen_id_feats, chosen_num_feats = chosen_states
        rejected_id_feats, rejected_num_feats = rejected_states
        
        # 计算偏好分值 (logits)
        chosen_logits = self.policy_network(chosen_id_feats, chosen_num_feats)
        rejected_logits = self.policy_network(rejected_id_feats, rejected_num_feats)
        
        # DPO 损失函数的核心
        log_probs_diff = chosen_logits - rejected_logits
        loss = -F.logsigmoid(DPO_BETA * log_probs_diff).mean()
        
        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def act(self, state_options_with_id, eps=0.):
        """
        根据学到的策略选择动作 (项目)。
        """
        # DPO是确定性策略，eps参数可以忽略，但保留以兼容评估函数
        if not state_options_with_id: return None
        
        self.policy_network.eval() # 切换到评估模式
        with torch.no_grad():
            all_id_features, all_numeric_features = [], []
            for pid, s_tup in state_options_with_id:
                id_features = [s_tup[0], s_tup[1], s_tup[2]]
                numeric_features = s_tup[3]
                all_id_features.append(id_features)
                all_numeric_features.append(numeric_features)
            
            id_features_tensor = torch.LongTensor(all_id_features).to(device)
            numeric_features_tensor = torch.FloatTensor(np.array(all_numeric_features)).to(device)
            
            # 获得所有选项的偏好分
            logits = self.policy_network(id_features_tensor, numeric_features_tensor)
            
            # 选择分数最高的动作
            action_idx = torch.argmax(logits).item()
        
        self.policy_network.train() # 恢复训练模式
        return state_options_with_id[action_idx][0]


def calculate_unified_reward(chosen_project_id, current_worker_id, chosen_project_state_tuple,
                             project_info_map, entry_info_map, 
                             scale_reference, scaling_factor):
    """
    一个统一的、标准的奖励计算函数。
    它计算一个基础奖励，然后应用缩放因子。
    """
    
    # --- 1. 查找历史表现 ---
    actual_award_hist = 0.0
    hist_score = 0
    is_winner_hist = False
    hq_award_hist = False
    worker_participated_hist = False
    
    if chosen_project_id in entry_info_map:
        for _, ed in entry_info_map[chosen_project_id].items():
            if ed["worker_id"] == current_worker_id and not ed.get("withdrawn", False):
                worker_participated_hist = True
                current_score_val = ed.get("score", 0)
                if current_score_val > hist_score:
                    hist_score = current_score_val
                
                award_val_raw = ed.get("award_value")
                if award_val_raw is not None:
                    try:
                        award_val_float = float(award_val_raw)
                        if award_val_float > 0:
                            hq_award_hist = True
                            actual_award_hist = award_val_float
                            if ed.get("winner"):
                                is_winner_hist = True
                                break 
                    except (ValueError, TypeError):
                        pass

                if not hq_award_hist and ed.get("winner"):
                    is_winner_hist = True
                    proj_d = project_info_map.get(chosen_project_id)
                    if proj_d:
                        award_from_proj = proj_d.get("total_awards", 0)
                        if award_from_proj > 0:
                            actual_award_hist = award_from_proj
                            hq_award_hist = True
                            break

    # --- 2. 基于历史表现计算基础奖励 ---
    base_reward = 0.0
    REWARD_ACTION_COST = -0.01 # 每次行动的基础成本
    final_reward = 0.0  # <--- 在这里初始化 final_reward
    
    if hq_award_hist:
        bonus_awd = (actual_award_hist / scale_reference) if scale_reference > 0 else 0
        base_reward = 1.0 + np.clip(bonus_awd, 0, 1.0)
        final_reward = base_reward * scaling_factor 
    elif worker_participated_hist and hist_score >= 4:
        base_reward = 0.5 + (hist_score - 4) * 0.2
        final_reward = base_reward * scaling_factor 
    elif worker_participated_hist and hist_score >= 3:
        base_reward = 0.2
        final_reward = base_reward * scaling_factor 
    elif not worker_participated_hist:
        # 潜力奖励
        numeric_features = chosen_project_state_tuple[3] # 数值特征在索引3
        wq_n = numeric_features[0]   # 归一化的工人质量
        rps_n = numeric_features[8]  # 归一化的单位槽位奖励
        potential_r = 0.2 * (wq_n + rps_n)
        final_reward = potential_r
    # 最终的奖励是 (行动成本 + 基础奖励) * 缩放因子
    
    return final_reward
# --- 新增：评估函数 (最终修正版) ---
def evaluate_agent(agent_to_eval, evaluation_events, proj_info, ent_info, wq_map, 
                   global_reward_scale_ref):
    
    agent_to_eval.policy_network.eval()  # 切换到评估模式
    total_eval_reward = 0
    num_eval_recommendations = 0

    # ⭐ 重要优化：在评估时禁用梯度计算
    with torch.no_grad():
        for event in evaluation_events:
            current_worker_id, current_time_dt = event['worker_id'], event['arrival_time_dt']

            # ... (获取可用项目和构建状态的逻辑保持不变) ...
            available_project_ids = get_available_projects(current_time_dt, proj_info, ent_info)
            if not available_project_ids:
                continue

            state_options_with_proj_id = []
            for proj_id in available_project_ids:
                final_state_tuple = get_new_final_state_tuple(
                    current_worker_id, proj_id, current_time_dt,
                    project_info, worker_quality_map, 
                    worker_global_stats, worker_cat_performance # 传入新的两个特征字典
                    )
                if final_state_tuple is not None:
                    state_options_with_proj_id.append((proj_id, final_state_tuple))
            
            if not state_options_with_proj_id:
                continue

            # ⭐ 关键修正：评估时使用 eps=0.0，完全利用模型学到的策略
            chosen_project_id = agent_to_eval.act(state_options_with_proj_id, eps=0.0)
            if chosen_project_id is None:
                continue

            chosen_project_state_tuple = next(s_tuple for pid, s_tuple in state_options_with_proj_id if pid == chosen_project_id)
            
            # --- 调用统一的奖励函数 ---
            reward = calculate_unified_reward(
                chosen_project_id=chosen_project_id,
                current_worker_id=current_worker_id,
                chosen_project_state_tuple=chosen_project_state_tuple,
                project_info_map=proj_info, # 注意这里变量名叫 proj_info
                entry_info_map=ent_info,   # 注意这里变量名叫 ent_info
                scale_reference=global_reward_scale_ref, # 注意这里变量名
                scaling_factor=REWARD_SCALING_FACTOR
            )
            
            total_eval_reward += reward
            num_eval_recommendations += 1

    agent_to_eval.policy_network.train()  # 恢复训练模式
    
    avg_eval_reward = total_eval_reward / num_eval_recommendations if num_eval_recommendations > 0 else 0.0
    return avg_eval_reward

# --- 工人到达事件生成 (V2修正版，携带更丰富信息) ---
print("\n开始生成更丰富的工人活跃事件池 (按天去重)...")
all_event_candidates = []
if entry_info:
    all_potential_events_temp = []
    # 首先收集所有有效的提交事件
    for project_id, entries_in_project in entry_info.items():
        for entry_number, entry_data in entries_in_project.items():
            if not entry_data.get("withdrawn", False):
                # --- 核心改动：在事件中保存 project_id 和 entry_number ---
                all_potential_events_temp.append({
                    'worker_id': entry_data["worker_id"],
                    'project_id': project_id, # <-- 保存项目ID
                    'entry_number': entry_number, # <-- 保存提交编号
                    'arrival_time_dt': entry_data["entry_created_at_dt"]
                })

    if all_potential_events_temp:
        # 按工人和时间排序，方便后续去重
        all_potential_events_temp.sort(key=lambda x: (x['worker_id'], x['arrival_time_dt']))
        
        processed_worker_days = set()
        for event_data in all_potential_events_temp:
            date_str = event_data['arrival_time_dt'].strftime('%Y-%m-%d')
            worker_day_key = (event_data['worker_id'], date_str)
            # 确保每个工人每天只取第一个事件
            if worker_day_key not in processed_worker_days:
                all_event_candidates.append(event_data) # 添加完整的事件字典
                processed_worker_days.add(worker_day_key)

        # 最终按全局时间排序
        all_event_candidates.sort(key=lambda x: x['arrival_time_dt'])
        print(f"去重后，总共生成了 {len(all_event_candidates)} 个工人活跃事件。")
    else:
        print("警告: 未能从entry_info生成任何潜在的工人活跃事件。")
else:
    print("警告: entry_info 为空，无法生成工人活跃事件池。")

# 确保 sorted_all_events 使用新的事件列表
sorted_all_events = all_event_candidates
# --- 数据集划分 (基于 sorted_all_events) ---
train_events, validation_events, test_events = [], [], []
if sorted_all_events:  # 使用新的 sorted_all_events进行划分
    split_ratio_train = 0.7
    split_ratio_val = 0.15

    num_total_events = len(sorted_all_events)

    if num_total_events < 100:
        print("警告: 总事件数过少，将大部分用于训练，少量用于验证，不设测试集。")
        split_idx_train_end = int(num_total_events * 0.85)
        train_events = sorted_all_events[:split_idx_train_end]
        validation_events = sorted_all_events[split_idx_train_end:]
        test_events = []
    else:
        split_ratio_test = 1.0 - split_ratio_train - split_ratio_val  # 确保总和为1

        train_idx_end = int(num_total_events * split_ratio_train)
        val_idx_end = train_idx_end + int(num_total_events * split_ratio_val)

        train_events = sorted_all_events[:train_idx_end]  # 修改这里，确保正确切分
        validation_events = sorted_all_events[train_idx_end:val_idx_end]
        test_events = sorted_all_events[val_idx_end:]

    print(
        f"数据集划分: 训练集 {len(train_events)} 条, 验证集 {len(validation_events)} 条, 测试集 {len(test_events)} 条")
else:
    print("警告: 完整工人活跃事件池为空，无法划分数据集。")
# --- 在 "数据集划分" 和 "预计算 Min/Max" 之间，加入这段代码 ---
# --- 修改：预计算 Min/Max 特征值 (仅使用训练集数据) ---
if train_events: # **确保 train_events 不是空的**
    precompute_feature_min_max(train_events, project_info, entry_info, worker_quality_map) # **传入 train_events**
else:
    print("警告: 训练事件集为空，跳过特征 Min/Max 预计算，将使用默认范围(0,1)。")
    for name in ["worker_quality","time_until_deadline_sec","task_age_sec","project_duration_sec","reward_per_slot","current_time_val"]:
        if name not in feature_min_max: feature_min_max[name] = (0,1)

### --- 主程序执行流程 --- ###
# 建议将主流程代码放在 if __name__ == "__main__": 中
# 这是一个Python编程的好习惯，尤其是在使用多进程（如DataLoader的num_workers > 0）时，可以避免不必要的问题。

### --- 主程序执行流程 (DPO版本) --- ###
if __name__ == "__main__":

    # --- 1. DPO专属的数据准备 ---
    # 核心改动：不再使用事件流直接训练，而是先从事件流中生成DPO所需的偏好对。
    # 注意：这个过程可能会花费一些时间，是必要的一次性预处理。
    dpo_train_data = create_preference_dataset_v4(
        train_events, project_info, entry_info, worker_quality_map, 
        worker_global_stats, worker_cat_performance
    )
    dpo_val_data = create_preference_dataset_v4(
        validation_events, project_info, entry_info, worker_quality_map, 
        worker_global_stats, worker_cat_performance
    )

    # 如果生成的数据过少，后续可能无法训练
    if len(dpo_train_data) < BATCH_SIZE:
        print(f"错误：生成的训练偏好数据（{len(dpo_train_data)}条）不足一个批次（{BATCH_SIZE}条），无法继续。")
        sys.exit() # 退出程序

    # --- 2. 实例化 DPO Agent ---
    # 核心改动：使用我们新定义的 DPOAgent
    agent = DPOAgent(
        num_categories_embed_size=NUM_CATEGORIES_EMBED_SIZE,
        num_sub_categories_embed_size=NUM_SUB_CATEGORIES_EMBED_SIZE,
        num_industries_embed_size=NUM_INDUSTRIES_EMBED_SIZE,
        cat_embed_dim=CATEGORY_EMBED_DIM,
        sub_cat_embed_dim=SUB_CATEGORY_EMBED_DIM,
        ind_embed_dim=INDUSTRY_EMBED_DIM,
        total_numeric_features=TOTAL_NUMERIC_FEATURES, 
        seed=0
    )

    # DPO不需要学习率调度器，可以先移除
    # scheduler = optim.lr_scheduler.StepLR(agent.optimizer, step_size=40, gamma=0.5)

    # --- 3. 创建新的 DataLoader ---
    # 核心改动：为DPO偏好对数据创建一个新的、更标准的DataLoader
    def dpo_collate_fn(batch):
        """一个简单的整理函数，用于将偏好对列表打包成tensor批次。"""
        chosen_id_list, chosen_num_list = [], []
        rejected_id_list, rejected_num_list = [], []
        
        for pair in batch:
            chosen_s, rejected_s = pair
            chosen_id_list.append([chosen_s[0], chosen_s[1], chosen_s[2]])
            chosen_num_list.append(chosen_s[3])
            rejected_id_list.append([rejected_s[0], rejected_s[1], rejected_s[2]])
            rejected_num_list.append(rejected_s[3])
            
        chosen_ids = torch.LongTensor(chosen_id_list)
        chosen_nums = torch.FloatTensor(np.array(chosen_num_list))
        rejected_ids = torch.LongTensor(rejected_id_list)
        rejected_nums = torch.FloatTensor(np.array(rejected_num_list))
        
        # 返回两个元组，每个元组包含(id_features, numeric_features)
        return (chosen_ids, chosen_nums), (rejected_ids, rejected_nums)

    train_loader = DataLoader(
        dpo_train_data, 
        batch_size=BATCH_SIZE,
        shuffle=True,       # DPO使用标准监督学习，打乱数据是好习惯
        num_workers=4,
        collate_fn=dpo_collate_fn,
        pin_memory=True
    )
    
    # --- 4. 初始化训练所需变量 ---
    num_epochs = 120  # DPO可能收敛更快，可适当调整
    all_train_epoch_losses = []
    all_val_epoch_rewards = []
    
    print("\n开始使用DPO进行训练...")
    
    # 早停机制变量 (保留，但修改文件名以作区分)
    best_val_reward = -float('inf')
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    best_model_path = os.path.join(script_dir, 'dpo_best_model.pth')
    patience = 10 # DPO训练可能更稳定，可以适当增加耐心
    epochs_no_improve = 0

    # --- 5. 开始新的DPO主训练循环 ---
    for i_epoch in range(1, num_epochs + 1):
        agent.policy_network.train() # 确保是训练模式
        epoch_losses = []
        
        # 核心改动：直接从DataLoader中获取预处理好的批次
        for batch in train_loader:
            # 将数据移动到GPU
            chosen_states, rejected_states = batch
            chosen_states_gpu = (chosen_states[0].to(device, non_blocking=True), chosen_states[1].to(device, non_blocking=True))
            rejected_states_gpu = (rejected_states[0].to(device, non_blocking=True), rejected_states[1].to(device, non_blocking=True))
            
            loss = agent.learn((chosen_states_gpu, rejected_states_gpu))
            epoch_losses.append(loss)
            
        avg_loss = np.mean(epoch_losses)
        all_train_epoch_losses.append(avg_loss)
        
        # --- 6. 验证、评估与早停 ---
        # 评估部分我们仍然沿用之前的 evaluate_agent 函数，因为它衡量的是在真实推荐场景下的业务奖励，
        # 这是比DPO自身的“分类准确率”更有价值、也更方便与CQL对比的指标。
        if i_epoch % 5 == 0 and dpo_val_data: # 可以更频繁地验证
            val_reward_epoch = evaluate_agent(agent, validation_events, project_info, entry_info, worker_quality_map, REWARD_SCALE_REFERENCE)
            all_val_epoch_rewards.append(val_reward_epoch)
            
            print(f"轮次 {i_epoch}/{num_epochs}\t训练损失: {avg_loss:.6f}\t验证奖励: {val_reward_epoch:.4f}")

            # 早停逻辑 (基本不变，只需修改保存的模型)
            if val_reward_epoch > best_val_reward:
                best_val_reward = val_reward_epoch
                epochs_no_improve = 0
                try:
                # 检查 agent.policy_network 是否为 DataParallel 实例
                    if isinstance(agent.policy_network, nn.DataParallel):
                    # 如果是，则我们只保存其内部的 .module 的状态字典
                        state_dict_to_save = agent.policy_network.module.state_dict()
                    else:
                    # 如果不是（例如在单GPU环境下运行），则直接保存
                        state_dict_to_save = agent.policy_network.state_dict()
    
                    torch.save(state_dict_to_save, best_model_path)
                    print(f"  🎉 新的最佳验证奖励！模型已保存到 {best_model_path}")

                except Exception as e_save_best:
                    print(f"  保存最佳模型失败: {e_save_best}")

            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"\n验证奖励已连续 {patience * 5} 个轮次没有提升，触发提前停止。")
                break
    
    print("\n训练完成。")

    # --- 7. 最终评估、绘图与保存 ---
    # --- 在 "7. 最终评估、绘图与保存" 部分 ---
    if test_events:
        print("\n开始在测试集上评估DPO最佳模型...")
        try:
            if os.path.exists(best_model_path):
                print(f"正在从 {best_model_path} 加载最佳模型...")
                # 加载我们保存的“干净”的状态字典
                state_dict = torch.load(best_model_path, map_location=device)
            
                # 获取核心模型（无论是单GPU还是多GPU）
                model_to_load = agent.policy_network.module if isinstance(agent.policy_network, nn.DataParallel) else agent.policy_network
            
                # 将状态字典加载到核心模型中
                model_to_load.load_state_dict(state_dict)
            
                print("DPO最佳模型加载成功。")
                test_reward = evaluate_agent(agent, test_events, project_info, entry_info, worker_quality_map, REWARD_SCALE_REFERENCE)
                print(f"🏆 DPO模型 - 测试集平均奖励: {test_reward:.4f}")
            else:
                print("警告: 未找到DPO最佳模型文件，无法进行测试。")
                test_reward = None
        except Exception as e_test:
            print(f"DPO模型测试评估失败: {e_test}")
            test_reward = None

    try:
        plt.figure(figsize=(16, 8))
        
        # 奖励图
        ax1 = plt.subplot(1, 2, 1)
        # DPO没有训练奖励，只绘制验证奖励
        if all_val_epoch_rewards:
            val_epochs_plot = range(5, 5 * len(all_val_epoch_rewards) + 1, 5)
            ax1.plot(val_epochs_plot, all_val_epoch_rewards, label='Avg Reward per Validation Epoch', marker='o', linestyle='--')
        if test_reward is not None:
            ax1.axhline(y=test_reward, color='r', linestyle=':', linewidth=2, label=f'Test Set Avg Reward: {test_reward:.4f}')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Average Reward'); ax1.set_title('DPO Validation & Test Reward'); ax1.legend(); ax1.grid(True)
        
        # 损失图
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(all_train_epoch_losses, label='Avg DPO Loss per Training Epoch', color='green')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.set_title('DPO Training: Average Loss per Epoch'); ax2.legend(); ax2.grid(True)
        
        plt.tight_layout()
        curves_save_path = os.path.join(script_dir, 'dpo_training_curves.png')
        plt.savefig(curves_save_path)
        print(f"\nDPO训练曲线已保存到 {curves_save_path}")
    except Exception as e_plot:
        print(f"\n绘制DPO训练曲线失败: {e_plot}")