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

# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- 常量和配置 ---
ACTION_SIZE = 1
GAMMA = 0.99
LEARNING_RATE = 1e-4
BUFFER_SIZE = int(5e4)
BATCH_SIZE = 64
TAU = 1e-3
UPDATE_EVERY = 4
TARGET_UPDATE_EVERY = 100
CATEGORY_EMBED_DIM = 5
SUB_CATEGORY_EMBED_DIM = 8
INDUSTRY_EMBED_DIM = 5
WORKER_ID_EMBED_DIM = 8
NUM_NUMERIC_WORKER_FEATURES = 1
NUM_NUMERIC_PROJECT_FEATURES = 4
NUM_NUMERIC_CONTEXT_FEATURES = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备: {device}")

all_begin_time_dt = parse("2018-01-01T0:0:0Z")
feature_min_max = {}
REWARD_SCALE_REFERENCE = 20.0  # 先给一个默认值，会被后续分析覆盖
# --- 新增/确保以下定义存在 ---
OBJECTIVE_MODE = "REQUESTER_PROFIT"  # 或者设置为 "WORKER_PROFIT"，取决于你当前想运行的模式
print(f"当前优化目标模式: {OBJECTIVE_MODE}")

# 根据 OBJECTIVE_MODE 定义特定模式下的奖励常量 (如果之前没有定义的话)
if OBJECTIVE_MODE == "REQUESTER_PROFIT":
    REQUESTER_REWARD_FOR_WINNER = 2.0
    REQUESTER_REWARD_HIGH_SCORE_BASE = 1.0
    REQUESTER_REWARD_MID_SCORE_BASE = 0.3
    REQUESTER_PENALTY_LOW_SCORE = -0.2
    REQUESTER_PENALTY_USELESS_SUBMISSION = -0.5
    REQUESTER_PENALTY_IGNORED_RECOMMENDATION = -0.1
    # 你可能还需要调整 REWARD_SCALE_REFERENCE 的使用，或者为请求者模式定义新的缩放基准
elif OBJECTIVE_MODE == "WORKER_PROFIT":
    # 这里可以保留或调整工人利益模式下的特定常量
    # REWARD_SCALE_REFERENCE 在工人利益模式下更常用
    pass
# --- 定义结束 -
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


# --- QNetwork, ReplayBuffer, DQNAgentWithEmbeddings 类定义 ---
# ... (粘贴你已验证无误的这三个类的完整定义) ...
# (确保 ReplayBuffer.sample 和 DQNAgentWithEmbeddings.act/learn/step 正确处理特征元组)
# (此处粘贴上一回复中的 QNetworkWithEmbeddings, ReplayBuffer, DQNAgentWithEmbeddings 类定义)
class QNetworkWithEmbeddings(nn.Module):  # QNetworkWithEmbeddings 定义
    def __init__(self, num_workers_embed_size, num_categories_embed_size, num_sub_categories_embed_size,
                 num_industries_embed_size, worker_embed_dim, cat_embed_dim, sub_cat_embed_dim, ind_embed_dim,
                 num_numeric_worker_features, num_numeric_project_features, num_numeric_context_features, seed,
                 fc1_units=128, fc2_units=32):
        super(QNetworkWithEmbeddings, self).__init__();
        self.seed = torch.manual_seed(seed)
        self.worker_embedding = nn.Embedding(num_workers_embed_size, worker_embed_dim);
        self.category_embedding = nn.Embedding(num_categories_embed_size, cat_embed_dim);
        self.sub_category_embedding = nn.Embedding(num_sub_categories_embed_size, sub_cat_embed_dim);
        self.industry_embedding = nn.Embedding(num_industries_embed_size, ind_embed_dim)
        total_embed_dim = worker_embed_dim + cat_embed_dim + sub_cat_embed_dim + ind_embed_dim
        fc_input_dim = total_embed_dim + num_numeric_worker_features + num_numeric_project_features + num_numeric_context_features
        self.fc1 = nn.Linear(fc_input_dim, fc1_units);
        self.dropout1 = nn.Dropout(0.1);
        self.fc2 = nn.Linear(fc1_units, fc2_units);
        self.dropout2 = nn.Dropout(0.1);
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, worker_ids, cat_ids, sub_cat_ids, ind_ids, numeric_worker_feats, numeric_project_feats,
                numeric_context_feats):
        w_embed, cat_embed, sub_cat_embed, ind_embed = self.worker_embedding(
            worker_ids.squeeze(-1)), self.category_embedding(cat_ids.squeeze(-1)), self.sub_category_embedding(
            sub_cat_ids.squeeze(-1)), self.industry_embedding(ind_ids.squeeze(-1))
        embedded_features = torch.cat([w_embed, cat_embed, sub_cat_embed, ind_embed], dim=1)
        all_features = torch.cat(
            [embedded_features, numeric_worker_feats, numeric_project_feats, numeric_context_feats], dim=1)
        x = F.relu(self.fc1(all_features));
        x = self.dropout1(x);
        x = F.relu(self.fc2(x));
        x = self.dropout2(x);
        return self.fc3(x)


Experience = namedtuple("Experience",
                        field_names=["state_tuple", "action_project_id", "reward", "next_state_options_tuples", "done"])


class ReplayBuffer:  # ReplayBuffer 定义
    def __init__(self, bs, bsize, seed):
        self.memory = deque(maxlen=bs); self.batch_size = bsize; random.seed(seed)

    def add(self, st, ap, r, nsot, d):
        self.memory.append(Experience(st, ap, r, nsot, d))

    def sample(self):
        exps = random.sample(self.memory, k=self.batch_size)
        s_tups = [e.state_tuple for e in exps]
        s_wid = torch.tensor([[s[0]] for s in s_tups], dtype=torch.long, device=device)
        s_cid = torch.tensor([[s[1]] for s in s_tups], dtype=torch.long, device=device)
        s_scid = torch.tensor([[s[2]] for s in s_tups], dtype=torch.long, device=device)
        s_iid = torch.tensor([[s[3]] for s in s_tups], dtype=torch.long, device=device)
        s_nwf = torch.tensor(np.vstack([s[4] for s in s_tups]), dtype=torch.float, device=device)
        s_npf = torch.tensor(np.vstack([s[5] for s in s_tups]), dtype=torch.float, device=device)
        s_ncf = torch.tensor(np.vstack([s[6] for s in s_tups]), dtype=torch.float, device=device)
        current_s_batch = (s_wid, s_cid, s_scid, s_iid, s_nwf, s_npf, s_ncf)
        rwd = torch.tensor([e.reward for e in exps], dtype=torch.float, device=device).unsqueeze(1)
        dns = torch.tensor([e.done for e in exps], dtype=torch.float, device=device).unsqueeze(1)
        next_q_vals_list = []
        for e in exps:
            if not e.done and e.next_state_options_tuples:
                q_opts = []
                for next_s_tup in e.next_state_options_tuples:
                    next_s_tens = [torch.tensor([[c]], dtype=torch.long, device=device) if i < 4 else torch.tensor(c,
                                                                                                                   dtype=torch.float,
                                                                                                                   device=device).unsqueeze(
                        0) for i, c in enumerate(next_s_tup)]
                    with torch.no_grad(): q_opts.append(agent.q_target(*next_s_tens).item())
                next_q_vals_list.append(max(q_opts) if q_opts else 0.0)
            else:
                next_q_vals_list.append(0.0)
        next_q_vals = torch.tensor(next_q_vals_list, dtype=torch.float, device=device).unsqueeze(1)
        return current_s_batch, rwd, next_q_vals, dns

    def __len__(self):
        return len(self.memory)


class DQNAgentWithEmbeddings:
    def __init__(self, num_workers_embed_size, num_categories_embed_size, num_sub_categories_embed_size,
                 num_industries_embed_size, worker_embed_dim, cat_embed_dim, sub_cat_embed_dim, ind_embed_dim,
                 num_numeric_worker_features, num_numeric_project_features, num_numeric_context_features, seed): # 使用完整参数名
        self.q_local = QNetworkWithEmbeddings(
            num_workers_embed_size, num_categories_embed_size, num_sub_categories_embed_size, num_industries_embed_size,
            worker_embed_dim, cat_embed_dim, sub_cat_embed_dim, ind_embed_dim,
            num_numeric_worker_features, num_numeric_project_features, num_numeric_context_features, seed
        ).to(device)
        self.q_target = QNetworkWithEmbeddings(
            num_workers_embed_size, num_categories_embed_size, num_sub_categories_embed_size, num_industries_embed_size,
            worker_embed_dim, cat_embed_dim, sub_cat_embed_dim, ind_embed_dim,
            num_numeric_worker_features, num_numeric_project_features, num_numeric_context_features, seed
        ).to(device)
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=LEARNING_RATE,weight_decay=1e-5)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step, self.c_step = 0, 0
        self.soft_update(self.q_local, self.q_target, 1.0)

    def step(self, st, ap, r, nsot, d):  # 参数名与你代码一致
        self.memory.add(st, ap, r, nsot, d)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        loss_val = None  # 初始化损失值，以防本次step不进行学习
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            loss_val = self.learn(self.memory.sample(), GAMMA)  # <--- 调用 learn 并接收返回的损失值

        self.c_step = (self.c_step + 1) % TARGET_UPDATE_EVERY
        if self.c_step == 0:
            self.soft_update(self.q_local, self.q_target, TAU)  # 使用 self.q_local 和 self.q_target

        return loss_val  # <--- 返回损失值 (如果进行了学习则为具体值，否则为None)

    def act(self, state_options_with_id, eps=0.):
        if not state_options_with_id: return None
        if random.random() > eps:
            best_pid, max_q = None, -np.inf;
            self.q_local.eval()
            with torch.no_grad():
                for pid, s_tup in state_options_with_id:
                    s_tens = [torch.tensor([[c]], dtype=torch.long, device=device) if i < 4 else torch.tensor(c,
                                                                                                              dtype=torch.float,
                                                                                                              device=device).unsqueeze(
                        0) for i, c in enumerate(s_tup)]
                    q = self.q_local(*s_tens).item()
                    if q > max_q: max_q, best_pid = q, pid
            self.q_local.train();
            return best_pid
        return random.choice([pid for pid, _ in state_options_with_id])

    def learn(self, exps, gamma):
        states_b_tup, rwds, next_qs, dns = exps
        Q_targets = rwds + (gamma * next_qs * (1 - dns))
        Q_expected = self.q_local(*states_b_tup)
        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()  # 返回计算得到的损失值
    def step(self, st, ap, r, nsot, d):
        self.memory.add(st, ap, r, nsot, d)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        loss_val = None  # 初始化损失值
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            loss_val = self.learn(self.memory.sample(), GAMMA)  # <--- 接收损失值

        self.c_step = (self.c_step + 1) % TARGET_UPDATE_EVERY
        if self.c_step == 0:
            self.soft_update(self.q_local, self.q_target, TAU)

        return loss_val  # <--- 返回损失值
    def soft_update(self, local, target, tau):
        for t_p, l_p in zip(target.parameters(), local.parameters()): t_p.data.copy_(
            tau * l_p.data + (1.0 - tau) * t_p.data)


# --- 工人到达事件生成 (生成完整的事件池) ---
print("\n开始生成更丰富的工人活跃事件池 (按天去重)...")
all_event_candidates = []  # 使用一个新的名字来存储所有候选事件
if entry_info:
    all_potential_events_temp = []
    for project_id, entries_in_project in entry_info.items():
        for entry_number, entry_data in entries_in_project.items():
            if not entry_data.get("withdrawn", False):
                all_potential_events_temp.append({
                    'worker_id': entry_data["worker_id"],
                    'arrival_time_dt': entry_data["entry_created_at_dt"]
                })

    if all_potential_events_temp:
        all_potential_events_temp.sort(key=lambda x: (x['worker_id'], x['arrival_time_dt']))
        processed_worker_days = set()
        for event_data in all_potential_events_temp:  # 改名为 event_data 避免与循环变量 event 混淆
            date_str = event_data['arrival_time_dt'].strftime('%Y-%m-%d')
            worker_day_key = (event_data['worker_id'], date_str)
            if worker_day_key not in processed_worker_days:
                all_event_candidates.append({  # 添加到新的列表
                    'worker_id': event_data['worker_id'],
                    'arrival_time_dt': event_data['arrival_time_dt']
                })
                processed_worker_days.add(worker_day_key)

        all_event_candidates.sort(key=lambda x: x['arrival_time_dt'])  # 最终按全局时间排序
        print(f"去重后，总共生成了 {len(all_event_candidates)} 个工人活跃事件（按工人每天首次活动）。")
    else:
        print("警告: 未能从entry_info生成任何潜在的工人活跃事件。")
else:
    print("警告: entry_info 为空，无法生成工人活跃事件池。")

# sorted_all_events 将是我们后续划分和查找next_state的基础
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

# --- 修改：预计算 Min/Max 特征值 (仅使用训练集数据) ---
if train_events: # **确保 train_events 不是空的**
    precompute_feature_min_max(train_events, project_info, entry_info, worker_quality_map) # **传入 train_events**
else:
    print("警告: 训练事件集为空，跳过特征 Min/Max 预计算，将使用默认范围(0,1)。")
    for name in ["worker_quality","time_until_deadline_sec","task_age_sec","project_duration_sec","reward_per_slot","current_time_val"]:
        if name not in feature_min_max: feature_min_max[name] = (0,1)

# --- Agent 实例化 ---
agent = DQNAgentWithEmbeddings(
    num_workers_embed_size=NUM_WORKERS_EMBED_SIZE,
    num_categories_embed_size=NUM_CATEGORIES_EMBED_SIZE,
    num_sub_categories_embed_size=NUM_SUB_CATEGORIES_EMBED_SIZE,
    num_industries_embed_size=NUM_INDUSTRIES_EMBED_SIZE,
    worker_embed_dim=WORKER_ID_EMBED_DIM,
    cat_embed_dim=CATEGORY_EMBED_DIM,
    sub_cat_embed_dim=SUB_CATEGORY_EMBED_DIM,
    ind_embed_dim=INDUSTRY_EMBED_DIM,
    num_numeric_worker_features=NUM_NUMERIC_WORKER_FEATURES,
    num_numeric_project_features=NUM_NUMERIC_PROJECT_FEATURES,
    num_numeric_context_features=NUM_NUMERIC_CONTEXT_FEATURES,
    seed=0
)

# --- 修改：评估函数 (evaluate_agent) ---
def evaluate_agent(agent_to_eval, evaluation_events, proj_info, ent_info, wq_map,
                   objective_mode_param, global_reward_scale_ref=REWARD_SCALE_REFERENCE):  # 添加 objective_mode_param
    # print(f"\n开始在 {len(evaluation_events)} 个事件上进行评估 (目标: {objective_mode_param})...")
    agent_to_eval.q_local.eval()
    total_eval_reward = 0
    num_eval_recommendations = 0

    for event_idx in range(len(evaluation_events)):
        event = evaluation_events[event_idx]
        current_worker_id, current_time_dt = event['worker_id'], event['arrival_time_dt']

        worker_id_embed, numeric_worker_f = get_worker_features_simplified(current_worker_id, wq_map)
        numeric_context_f = get_context_features_simplified(current_time_dt)
        available_project_ids = get_available_projects(current_time_dt, proj_info, ent_info)
        if not available_project_ids: continue

        state_options_with_proj_id = []
        for proj_id in available_project_ids:
            cat_id, sub_cat_id, ind_id, numeric_project_f_val = get_project_features_simplified(proj_id, proj_info,
                                                                                                current_time_dt)  # 重命名避免与外部numeric_project_f冲突
            if numeric_project_f_val is not None:
                state_tuple = (worker_id_embed, max(0, cat_id), max(0, sub_cat_id), max(0, ind_id),
                               numeric_worker_f, numeric_project_f_val, numeric_context_f)
                state_options_with_proj_id.append((proj_id, state_tuple))

        if not state_options_with_proj_id: continue

        chosen_project_id = agent_to_eval.act(state_options_with_proj_id, eps=0.01)
        if chosen_project_id is None: continue

        # --- 计算评估奖励 ---
        REWARD_ACTION_COST = -0.01
        eval_reward = REWARD_ACTION_COST

        hist_score_eval = 0
        is_winner_hist_eval = False
        worker_participated_hist_eval = False

        if chosen_project_id in ent_info:
            for _, ed in ent_info[chosen_project_id].items():
                if ed["worker_id"] == current_worker_id and not ed.get("withdrawn", False):
                    worker_participated_hist_eval = True
                    current_score_eval = ed.get("score", 0)
                    if current_score_eval > hist_score_eval:
                        hist_score_eval = current_score_eval
                    if ed.get("winner"):
                        is_winner_hist_eval = True

        if objective_mode_param == "REQUESTER_PROFIT":
            if worker_participated_hist_eval:
                if is_winner_hist_eval:
                    eval_reward += REQUESTER_REWARD_FOR_WINNER
                elif hist_score_eval >= 4:
                    eval_reward += REQUESTER_REWARD_HIGH_SCORE_BASE + (hist_score_eval - 4) * 0.25
                elif hist_score_eval >= 3:
                    eval_reward += REQUESTER_REWARD_MID_SCORE_BASE
                elif hist_score_eval > 0:
                    eval_reward += REQUESTER_PENALTY_LOW_SCORE
                else:
                    eval_reward += REQUESTER_PENALTY_USELESS_SUBMISSION
            else:
                eval_reward += REQUESTER_PENALTY_IGNORED_RECOMMENDATION

        elif objective_mode_param == "WORKER_PROFIT":
            actual_award_hist_eval = 0
            hq_award_hist_eval = False  # 需要为工人利益模式重新评估此值
            if chosen_project_id in ent_info:
                for _, ed in ent_info[chosen_project_id].items():
                    if ed["worker_id"] == current_worker_id and not ed.get("withdrawn", False):
                        award_val_eval = ed.get("award_value")
                        if award_val_eval is not None:
                            try:
                                if float(award_val_eval) > 0:
                                    hq_award_hist_eval = True
                                    actual_award_hist_eval = max(actual_award_hist_eval, float(award_val_eval))
                            except:
                                pass
                        if not hq_award_hist_eval and ed.get("winner"):
                            proj_d_eval = proj_info.get(chosen_project_id)
                            if proj_d_eval and proj_d_eval.get("required_answers", 0) > 0:
                                current_potential_award = proj_d_eval.get("total_awards", 0) / proj_d_eval.get(
                                    "required_answers", 1)
                                actual_award_hist_eval = max(actual_award_hist_eval, current_potential_award)
                                if actual_award_hist_eval > 0: hq_award_hist_eval = True

            if hq_award_hist_eval:
                bonus_awd_eval = (
                                             actual_award_hist_eval / global_reward_scale_ref) * 1.0 if global_reward_scale_ref > 0 else 0
                eval_reward += 1.0 + np.clip(bonus_awd_eval, 0, 1.0)
            elif worker_participated_hist_eval and hist_score_eval >= 4:
                eval_reward += 0.5 + (hist_score_eval - 4) * 0.2
            elif worker_participated_hist_eval and hist_score_eval >= 3:
                eval_reward += 0.2
            elif not worker_participated_hist_eval:
                # 需要 chosen_project_state_tuple 来获取工人质量和项目单位时隙奖励
                chosen_project_state_tuple_eval = next(
                    (s_tuple for pid, s_tuple in state_options_with_proj_id if pid == chosen_project_id), None)
                if chosen_project_state_tuple_eval:
                    pot_bonus_eval = 0.0
                    wq_n_eval, tr_n_eval = chosen_project_state_tuple_eval[4][0], chosen_project_state_tuple_eval[5][3]
                    if wq_n_eval > 0.6 and tr_n_eval > 0.4:
                        pot_bonus_eval = 0.15
                    elif wq_n_eval > 0.4 and tr_n_eval > 0.2:
                        pot_bonus_eval = 0.05
                    if pot_bonus_eval > 0: eval_reward += pot_bonus_eval

        total_eval_reward += eval_reward
        num_eval_recommendations += 1

    agent_to_eval.q_local.train()
    avg_eval_reward = total_eval_reward / num_eval_recommendations if num_eval_recommendations > 0 else 0.0
    return avg_eval_reward
# --- 修改后的训练循环，包含验证阶段 ---
num_epochs = 600 # 增加轮次以进行更充分的训练和验证
eps_start, eps_end, eps_decay = 1.0, 0.01, 0.998
eps = eps_start
all_train_epoch_rewards, all_val_epoch_rewards, all_train_epoch_losses = [], [], [] # 用于绘图
train_scores_window = deque(maxlen=20) # 用于打印最近的训练表现，窗口可以小一点

# --- 修改后的训练循环 ---
num_epochs = 300  # 和原来一致或按需调整
# 注意这里 eps_start_val, eps_end_val, eps_decay_val 的重命名，以避免与外部可能存在的eps冲突
# 但如果你之前的eps只在这个块内用，直接用eps_start, eps_end, eps_decay也可以
eps_start_val, eps_end_val, eps_decay_val_const = 1.0, 0.01, 0.998  # _const后缀表示这是原始的常量，下面会计算实际的eps_decay
eps = eps_start_val  # 使用重命名的变量初始化当前eps

all_train_epoch_rewards, all_val_epoch_rewards, all_train_epoch_losses = [], [], []
train_scores_window = deque(maxlen=20)

N_for_eps_decay_target = max(1, int(num_epochs * 0.75))
# 使用 eps_start_val 和 eps_end_val 进行计算
if N_for_eps_decay_target > 0 and eps_start_val > 0 and eps_end_val > 0 and eps_end_val < eps_start_val:
    eps_decay_calc = math.pow(eps_end_val / eps_start_val, 1.0 / N_for_eps_decay_target)
    eps_decay = np.clip(eps_decay_calc, 0.985, 0.999)  # 计算得到的eps_decay直接赋给变量eps_decay
else:
    eps_decay = 0.990  # 如果计算条件不满足，使用一个相对安全的默认值

print(f"训练总轮次: {num_epochs}")
print(f"Epsilon 将从 {eps_start_val} 衰减到接近 {eps_end_val} (目标在约 {N_for_eps_decay_target} 轮次)")
print(f"计算得到的 Epsilon 衰减率 (eps_decay): {eps_decay:.6f}")
# eps 已经在上面用 eps_start_val 初始化过了

print("\n开始训练 (包含验证)...")

# --- 修改：模型路径和输出文件路径根据 OBJECTIVE_MODE 变化 ---
# OBJECTIVE_MODE 应该在脚本更早的地方定义，例如在所有常量定义之后
# global OBJECTIVE_MODE (如果是在函数内部修改全局变量，但这里是在主脚本流程)
# 假设 OBJECTIVE_MODE 已经是 "REQUESTER_PROFIT" 或 "WORKER_PROFIT"
model_objective_suffix = "_worker_profit" if OBJECTIVE_MODE == "WORKER_PROFIT" else "_requester_profit"
base_output_dir = os.path.dirname(os.path.abspath(__file__))

# 全局定义 output_base_dir_container，确保绘图和模型保存时可用
# 例如，如果你在Docker容器内运行，并且映射了 /app/outputs
# output_base_dir_container = "/app/outputs"
# 如果在本地运行，可以这样设置：
output_base_dir_container = os.path.join(base_output_dir, "outputs",
                                         OBJECTIVE_MODE.replace("_PROFIT", "").lower())  # e.g., outputs/requester
if not os.path.exists(output_base_dir_container):
    os.makedirs(output_base_dir_container, exist_ok=True)
print(f"输出文件将保存在: {output_base_dir_container}")

best_val_reward = -float('inf')
best_model_path = os.path.join(output_base_dir_container,
                               f'best_model_checkpoint{model_objective_suffix}.pth')  # 使用新的路径规则

for i_epoch in range(1, num_epochs + 1):
    agent.q_local.train()
    total_reward_this_epoch_train = 0
    num_recommendations_made_train = 0
    current_epoch_step_losses_train = []

    for event_idx in range(len(train_events)):
        event = train_events[event_idx]
        current_worker_id, current_time_dt = event['worker_id'], event['arrival_time_dt']

        worker_id_for_embed, numeric_worker_f = get_worker_features_simplified(current_worker_id, worker_quality_map)
        numeric_context_f = get_context_features_simplified(current_time_dt)
        available_project_ids = get_available_projects(current_time_dt, project_info, entry_info)
        if not available_project_ids: continue

        state_options_with_proj_id = []
        for proj_id in available_project_ids:
            # 注意这里变量名是 numeric_project_f_val 避免覆盖外部同名变量（如果存在的话）
            cat_id, sub_cat_id, ind_id, numeric_project_f_val = get_project_features_simplified(proj_id, project_info,
                                                                                                current_time_dt)
            if numeric_project_f_val is not None:
                state_tuple = (worker_id_for_embed, max(0, cat_id), max(0, sub_cat_id), max(0, ind_id),
                               numeric_worker_f, numeric_project_f_val, numeric_context_f)
                state_options_with_proj_id.append((proj_id, state_tuple))

        if not state_options_with_proj_id: continue
        chosen_project_id = agent.act(state_options_with_proj_id, eps)
        if chosen_project_id is None: continue
        chosen_project_state_tuple = next(
            s_tuple for pid, s_tuple in state_options_with_proj_id if pid == chosen_project_id)

        REWARD_ACTION_COST = -0.01
        reward = REWARD_ACTION_COST

        hist_score = 0
        is_winner_hist = False
        worker_participated_hist = False

        if chosen_project_id in entry_info:
            for _, ed in entry_info[chosen_project_id].items():
                if ed["worker_id"] == current_worker_id and not ed.get("withdrawn", False):
                    worker_participated_hist = True
                    current_score_val = ed.get("score", 0)
                    if current_score_val > hist_score:
                        hist_score = current_score_val
                    if ed.get("winner"):
                        is_winner_hist = True

        # --- 根据 OBJECTIVE_MODE 计算奖励 ---
        if OBJECTIVE_MODE == "REQUESTER_PROFIT":
            if worker_participated_hist:
                if is_winner_hist:
                    reward += REQUESTER_REWARD_FOR_WINNER
                elif hist_score >= 4:
                    reward += REQUESTER_REWARD_HIGH_SCORE_BASE + (hist_score - 4) * 0.25
                elif hist_score >= 3:
                    reward += REQUESTER_REWARD_MID_SCORE_BASE
                elif hist_score > 0:  # score is 1 or 2
                    reward += REQUESTER_PENALTY_LOW_SCORE
                else:  # score is 0 or not available but participated
                    reward += REQUESTER_PENALTY_USELESS_SUBMISSION
            else:  # worker did not participate
                reward += REQUESTER_PENALTY_IGNORED_RECOMMENDATION

        elif OBJECTIVE_MODE == "WORKER_PROFIT":  # 保留原始的工人利益奖励逻辑
            actual_award_hist = 0  # 初始化实际历史奖励
            hq_award_hist = False  # 高质量奖励标志
            # 以下逻辑需要与你原始代码中计算 hq_award_hist 和 actual_award_hist 的部分完全一致
            if chosen_project_id in entry_info:
                for _, ed in entry_info[chosen_project_id].items():
                    if ed["worker_id"] == current_worker_id and not ed.get("withdrawn", False):
                        # worker_participated_hist 和 hist_score 已设置
                        award_val = ed.get("award_value")
                        if award_val is not None:
                            try:
                                if float(award_val) > 0:
                                    hq_award_hist = True
                                    actual_award_hist = max(actual_award_hist, float(award_val))  # 取最大奖励
                            except (ValueError, TypeError):
                                pass
                        # 如果没有直接奖励但被标记为赢家，尝试从项目总奖励中分配
                        if not hq_award_hist and ed.get("winner"):
                            # is_winner_hist 已经设置
                            proj_d = project_info.get(chosen_project_id)
                            if proj_d and proj_d.get("required_answers", 0) > 0:  # 避免除以零
                                # 简单分配平均奖励，实际可能更复杂
                                potential_award = proj_d.get("total_awards", 0) / proj_d.get("required_answers", 1)
                                actual_award_hist = max(actual_award_hist, potential_award)  # 取最大值
                                if actual_award_hist > 0: hq_award_hist = True

            if hq_award_hist:  # 这是工人利益模式下的 hq_award_hist
                bonus_awd = (actual_award_hist / REWARD_SCALE_REFERENCE) * 1.0  # REWARD_SCALE_REFERENCE 在此模式下使用
                reward += 1.0 + np.clip(bonus_awd, 0, 1.0)
            elif worker_participated_hist and hist_score >= 4:
                reward += 0.5 + (hist_score - 4) * 0.2
            elif worker_participated_hist and hist_score >= 3:
                reward += 0.2
            elif not worker_participated_hist:  # 工人未参与，但可能是个好匹配（基于工人质量和项目单位时隙奖励）
                pot_bonus = 0.0;
                wq_n, tr_n = chosen_project_state_tuple[4][0], chosen_project_state_tuple[5][3]
                if wq_n > 0.6 and tr_n > 0.4:
                    pot_bonus = 0.15
                elif wq_n > 0.4 and tr_n > 0.2:
                    pot_bonus = 0.05
                if pot_bonus > 0: reward += pot_bonus
        # --- 奖励计算结束 ---

        total_reward_this_epoch_train += reward
        num_recommendations_made_train += 1

        next_worker_state_options_tuples, done = [], True
        try:
            current_event_original_idx = sorted_all_events.index(event)
        except ValueError:
            done = True;
            current_event_original_idx = -1  # 标记为无效索引
        if current_event_original_idx != -1:  # 确保索引有效
            next_overall_event_idx = current_event_original_idx + 1
            if next_overall_event_idx < len(sorted_all_events):
                done = False
                next_event_overall = sorted_all_events[next_overall_event_idx]
                next_worker_id, next_time_dt = next_event_overall['worker_id'], next_event_overall['arrival_time_dt']
                next_w_id_e, next_num_w_f = get_worker_features_simplified(next_worker_id, worker_quality_map)
                next_num_c_f = get_context_features_simplified(next_time_dt)
                next_avail_p_ids = get_available_projects(next_time_dt, project_info, entry_info)
                for next_p_id in next_avail_p_ids:
                    # 同样，变量名用 _val 后缀避免潜在冲突
                    next_cat, next_sub_cat, next_ind, next_num_p_f_val = get_project_features_simplified(next_p_id,
                                                                                                         project_info,
                                                                                                         next_time_dt)
                    if next_num_p_f_val is not None:
                        next_s_t = (next_w_id_e, max(0, next_cat), max(0, next_sub_cat), max(0, next_ind), next_num_w_f,
                                    next_num_p_f_val, next_num_c_f)
                        next_worker_state_options_tuples.append(next_s_t)
        step_loss = agent.step(chosen_project_state_tuple, chosen_project_id, reward, next_worker_state_options_tuples,
                               done)
        if step_loss is not None:
            current_epoch_step_losses_train.append(step_loss)

    avg_reward_train_epoch = total_reward_this_epoch_train / num_recommendations_made_train if num_recommendations_made_train > 0 else 0.0
    avg_loss_train_epoch = np.mean(current_epoch_step_losses_train) if current_epoch_step_losses_train else 0.0
    train_scores_window.append(avg_reward_train_epoch)
    all_train_epoch_rewards.append(avg_reward_train_epoch)
    all_train_epoch_losses.append(avg_loss_train_epoch)

    eps = max(eps_end_val, eps * eps_decay)  # 使用计算的衰减率和新的eps_end_val

    val_reward_epoch = 0.0  # 初始化验证奖励
    if i_epoch % 10 == 0 and validation_events:
        # 在调用evaluate_agent时传递 OBJECTIVE_MODE
        # REWARD_SCALE_REFERENCE 也作为参数传递，虽然在REQUESTER_PROFIT模式下可能不直接用，但函数定义需要它
        val_reward_epoch = evaluate_agent(agent, validation_events, project_info, entry_info, worker_quality_map,
                                          OBJECTIVE_MODE, REWARD_SCALE_REFERENCE)
        all_val_epoch_rewards.append(val_reward_epoch)
        print(
            f'\r轮次 {i_epoch}/{num_epochs}\t训练奖励: {avg_reward_train_epoch:.4f}\t训练损失: {avg_loss_train_epoch:.4f}\t验证奖励: {val_reward_epoch:.4f}\tEpsilon: {eps:.3f}')

        if val_reward_epoch > best_val_reward:
            best_val_reward = val_reward_epoch
            try:
                torch.save(agent.q_local.state_dict(), best_model_path)  # best_model_path 已包含基于OBJECTIVE_MODE的后缀
                print(f"  新最佳模型已保存到 {best_model_path} (验证奖励: {best_val_reward:.4f})")
            except Exception as e_save_best:
                print(f"  保存最佳模型失败: {e_save_best}")
    else:  # 非验证轮次或验证集为空
        print(
            f'\r轮次 {i_epoch}/{num_epochs}\t训练奖励: {avg_reward_train_epoch:.4f}\t训练损失: {avg_loss_train_epoch:.4f}\tEpsilon: {eps:.3f}\t最近{len(train_scores_window)}训练轮平均: {np.mean(train_scores_window):.4f}',
            end="")
        if i_epoch % 20 == 0 and (i_epoch % 10 != 0 or not validation_events): print()  # 每20轮且非验证轮次时换行

print("\n训练完成。")

# --- （可选）最终在测试集上评估最佳模型 ---
if test_events:
    print("\n开始在测试集上评估最佳模型...")
    try:
        if os.path.exists(best_model_path):
            agent.q_local.load_state_dict(torch.load(best_model_path, map_location=device))
            test_reward = evaluate_agent(agent, test_events, project_info, entry_info, worker_quality_map, REWARD_SCALE_REFERENCE)
            print(f"测试集平均奖励: {test_reward:.4f}")
        else:
            print("警告: 未找到最佳模型文件 (best_model_checkpoint.pth)，使用当前最终模型进行测试。")
            test_reward = evaluate_agent(agent, test_events, project_info, entry_info, worker_quality_map, REWARD_SCALE_REFERENCE)
            print(f"测试集 (使用最终模型) 平均奖励: {test_reward:.4f}")

    except Exception as e_test:
        print(f"测试评估失败: {e_test}")

# --- 绘制并保存训练曲线 (使用 all_train_epoch_rewards, all_val_epoch_rewards, all_train_epoch_losses) ---
final_test_reward_str = "N/A" # 默认值
if 'test_reward' in locals() and test_reward is not None: # 检查test_reward是否已定义且有值
    final_test_reward_str = f"{test_reward:.4f}"

try:
    plt.figure(figsize=(12, 10))

    # 图1: 训练奖励和验证奖励 vs. Epochs
    plt.subplot(2, 1, 1)
    plt.plot(all_train_epoch_rewards, label='Avg Reward per Training Epoch')
    if all_val_epoch_rewards:
        val_epochs_plot = [i_epoch for i_epoch in range(1, num_epochs + 1) if i_epoch % 10 == 0 and i_epoch <= 10 * len(all_val_epoch_rewards)]
        if val_epochs_plot:
             plt.plot(val_epochs_plot[:len(all_val_epoch_rewards)], all_val_epoch_rewards, label='Avg Reward per Validation Epoch', marker='o', linestyle='--')

    # 在标题中加入测试奖励
    title_reward_plot = f'Training & Validation: Avg Reward ({OBJECTIVE_MODE})\nTest Avg Reward: {final_test_reward_str}'
    plt.xlabel('Epoch'); plt.ylabel('Average Reward'); plt.title(title_reward_plot); plt.legend(); plt.grid(True)

    # 图2: 训练损失 vs. Epochs
    plt.subplot(2, 1, 2)
    plt.plot(all_train_epoch_losses, label='Avg Loss per Training Epoch', color='green')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Training: Average Loss per Epoch ({OBJECTIVE_MODE})'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    curves_save_path = os.path.join(output_base_dir_container, f'training_curves_validation{model_objective_suffix}.png')
    plt.savefig(curves_save_path)
    print(f"\n训练曲线已保存到 {curves_save_path}")
except ImportError: print("\n未安装 matplotlib，无法绘制训练曲线。")
except Exception as e_plot: print(f"\n绘制训练曲线失败: {e_plot}")
# --- 模型保存 ---
try:
    model_save_path = os.path.join(output_base_dir_container, 'final_worker_profit_simplified_v1.pth')
    print(f"\n准备保存模型到: {model_save_path}")
    torch.save(agent.q_local.state_dict(), model_save_path)
    print("模型保存成功。")
except Exception as e_save:
    print(f"\n保存模型失败: {e_save}")


