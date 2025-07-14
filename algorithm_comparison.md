# Algorithm Implementation Comparison
## CQL vs DPO vs DDQN Analysis

### Overview
This document compares the three RL algorithm implementations to identify shared components and unique features that must be preserved during refactoring.

---

## 1. Algorithm-Specific Hyperparameters

### 1.1 CQL (Conservative Q-Learning)
```python
# Core CQL parameters
CQL_ALPHA = 0.15          # Conservative loss weight
GAMMA = 0.99              # Discount factor
TAU = 1e-3                # Soft update rate
LEARNING_RATE = 1e-5      # Learning rate
TARGET_UPDATE_EVERY = 100 # Target network update frequency

# Training parameters
BATCH_SIZE = 64
BUFFER_SIZE = int(5e4)
UPDATE_EVERY = 4
```

### 1.2 DPO (Direct Policy Optimization)
```python
# Core DPO parameters
DPO_BETA = 0.3                    # Policy deviation control
LEARNING_RATE = 3e-5              # Learning rate
NUM_REJECTED_SAMPLES = 3          # Rejected samples per chosen

# No discount factor (GAMMA) - direct policy optimization
# No target network (TAU) - single policy network
```

### 1.3 DDQN (Double Deep Q-Network)
```python
# Core DDQN parameters
GAMMA = 0.99              # Discount factor
TAU = 1e-3                # Soft update rate
LEARNING_RATE = 1e-4      # Learning rate
TARGET_UPDATE_EVERY = 100 # Target network update frequency

# Same training parameters as CQL
BATCH_SIZE = 64
BUFFER_SIZE = int(5e4)
UPDATE_EVERY = 4
```

---

## 2. Network Architectures

### 2.1 CQL Network (QNetworkWithEmbeddings)
```python
class QNetworkWithEmbeddings(nn.Module):
    def __init__(self, num_categories_embed_size, num_sub_categories_embed_size,
                 num_industries_embed_size, cat_embed_dim, sub_cat_embed_dim, 
                 ind_embed_dim, total_numeric_features, seed, 
                 fc1_units=128, fc2_units=32):
        
        # Embedding layers
        self.category_embedding = nn.Embedding(num_categories_embed_size, cat_embed_dim)
        self.sub_category_embedding = nn.Embedding(num_sub_categories_embed_size, sub_cat_embed_dim)
        self.industry_embedding = nn.Embedding(num_industries_embed_size, ind_embed_dim)
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_dim, fc1_units)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(fc2_units, 1)  # Q-value output
```

### 2.2 DPO Network (PolicyNetwork)
```python
class PolicyNetwork(nn.Module):
    def __init__(self, num_categories_embed_size, num_sub_categories_embed_size,
                 num_industries_embed_size, cat_embed_dim, sub_cat_embed_dim,
                 ind_embed_dim, total_numeric_features, seed):
        
        # Same embedding layers as CQL
        self.category_embedding = nn.Embedding(num_categories_embed_size, cat_embed_dim)
        self.sub_category_embedding = nn.Embedding(num_sub_categories_embed_size, sub_cat_embed_dim)
        self.industry_embedding = nn.Embedding(num_industries_embed_size, ind_embed_dim)
        
        # Different output - preference logits instead of Q-values
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)  # Preference logit output
```

### 2.3 DDQN Network
```python
# Similar to CQL but without conservative regularization
# Standard Q-network architecture with target network
```

---

## 3. Loss Functions

### 3.1 CQL Loss
```python
def learn(self, experiences, gamma):
    # Standard Bellman loss
    loss_bellman = F.mse_loss(q_expected, q_targets)
    
    # Conservative Q-Learning loss
    current_q_values = self.q_local(id_features, numeric_features)
    ood_q_values = self.q_local(ood_id_features, ood_numeric_features)
    
    # CQL regularization term
    loss_cql = torch.logsumexp(ood_q_values, dim=0).mean() - current_q_values.mean()
    
    # Combined loss
    total_loss = loss_bellman + CQL_ALPHA * loss_cql
```

### 3.2 DPO Loss
```python
def learn(self, batch_of_pairs):
    chosen_states, rejected_states = batch_of_pairs
    
    # Get preference logits
    chosen_logits = self.policy_network(chosen_id_feats, chosen_num_feats)
    rejected_logits = self.policy_network(rejected_id_feats, rejected_num_feats)
    
    # DPO loss function
    log_probs_diff = chosen_logits - rejected_logits
    loss = -F.logsigmoid(DPO_BETA * log_probs_diff).mean()
```

### 3.3 DDQN Loss
```python
def learn(self, experiences, gamma):
    # Standard Double Q-Learning loss
    # Action selection with local network
    # Q-value evaluation with target network
    loss = F.mse_loss(q_expected, q_targets)
```

---

## 4. Training Data Preparation

### 4.1 CQL Data Processing
```python
def cql_collate_fn(experiences_list):
    # In-distribution data from replay buffer
    experiences = experiences_list[0]
    
    # Out-of-distribution data generation
    num_random_actions = 10
    random_project_ids = np.random.choice(all_project_ids, 
                                        size=batch_size * num_random_actions)
    
    # Next state processing for Bellman updates
    # Complex state tuple handling
```

### 4.2 DPO Data Processing
```python
def dpo_collate_fn(batch):
    # Preference pair generation
    chosen_samples, rejected_samples = [], []
    
    for exp in batch:
        if exp.reward > reward_threshold:
            chosen_samples.append(exp.state_tuple)
            
            # Generate rejected samples
            for _ in range(NUM_REJECTED_SAMPLES):
                rejected_samples.append(random_negative_sample)
```

### 4.3 DDQN Data Processing
```python
# Standard experience replay
# No special data augmentation
# Simple state-action-reward-next_state tuples
```

---

## 5. Action Selection Strategies

### 5.1 CQL Action Selection
```python
def act(self, state_options_with_id, eps=0.):
    # Epsilon-greedy with Q-values
    if random.random() > eps:
        # Greedy action selection
        q_values = self.q_local(id_features, numeric_features)
        action_idx = np.argmax(q_values.cpu().data.numpy())
    else:
        # Random exploration
        action_idx = random.choice(range(len(state_options_with_id)))
```

### 5.2 DPO Action Selection
```python
def act(self, state_options_with_id, eps=0.):
    # Deterministic policy (eps ignored)
    logits = self.policy_network(id_features_tensor, numeric_features_tensor)
    action_idx = torch.argmax(logits).item()
```

### 5.3 DDQN Action Selection
```python
# Similar to CQL but with double Q-learning
# Action selection with local network
# Q-value evaluation with target network
```

---

## 6. Evaluation Metrics

### 6.1 Common Metrics (All Algorithms)
```python
def evaluate_agent(agent, test_events, project_info, entry_info, worker_quality_map):
    total_reward = 0.0
    total_actions = 0
    
    for event in test_events:
        # Get available actions
        # Select action using agent
        # Calculate reward
        # Accumulate metrics
    
    return total_reward / total_actions if total_actions > 0 else 0.0
```

### 6.2 Algorithm-Specific Metrics
- **CQL**: Q-value distributions, conservative loss components
- **DPO**: Preference accuracy, policy deviation from reference
- **DDQN**: Standard Q-learning metrics, target network sync

---

## 7. Key Differences Summary

| Aspect | CQL | DPO | DDQN |
|--------|-----|-----|------|
| **Network Type** | Q-Network | Policy Network | Q-Network |
| **Loss Function** | Bellman + Conservative | Preference-based | Bellman |
| **Data Augmentation** | OOD sampling | Preference pairs | None |
| **Action Selection** | ε-greedy | Deterministic | ε-greedy |
| **Target Network** | Yes | No | Yes |
| **Exploration** | ε-greedy | None | ε-greedy |
| **Offline Adaptation** | Conservative regularization | Preference learning | None |

---

## 8. Shared Components for Refactoring

### 8.1 Can Be Unified
- Data loading and preprocessing
- Feature engineering functions
- Embedding layer architectures
- Basic network components
- Evaluation framework
- Logging and visualization

### 8.2 Must Remain Separate
- Loss function implementations
- Data augmentation strategies
- Action selection mechanisms
- Algorithm-specific hyperparameters
- Network output layers
- Training loop specifics

---

## 9. Refactoring Strategy

### 9.1 Base Classes
```python
class BaseRLAlgorithm:
    # Common functionality
    
class ValueBasedAlgorithm(BaseRLAlgorithm):
    # Q-learning specific (CQL, DDQN)
    
class PolicyBasedAlgorithm(BaseRLAlgorithm):
    # Policy-based specific (DPO)
```

### 9.2 Preserved Uniqueness
- Each algorithm keeps its specific loss function
- Hyperparameters remain algorithm-specific
- Data processing can be customized per algorithm
- Action selection strategies remain distinct

This analysis ensures that refactoring will improve code structure while preserving the unique characteristics that make each algorithm effective for the crowdsourcing task.

