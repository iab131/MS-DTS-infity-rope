# Attention-as-Memory-Policy: Training-Free Streaming Multi-Shot AR Video Generation

## 1. 问题定义

给定一个因果自回归视频扩散模型（以Self-Forcing蒸馏的Wan2.1-1.3B为例，即Infinity-RoPE），实现以下目标：

- 输入：顺序到达的 (prompt, duration) 序列
- 输出：流式生成的多镜头视频，跨镜头的角色/物体保持视觉一致性
- 约束：不修改模型权重，不引入额外训练

基线系统（Infinity-RoPE）的生成方式：逐block生成（每block 3 latent frames），每block执行4步去噪 + 1次clean context pass更新KV cache。KV cache结构为 `[sink(1帧, 1560 tokens) | rolling_window(5帧, 7800 tokens)]`，总容量9360 tokens。

基线系统的镜头切换机制（`kv_flush`）：保留sink + 最后2帧，清空其余cache，重置cross-attention。该机制导致旧镜头的身份信息丢失。

---

## 2. 算法概述

### 2.1 核心观察

Infinity-RoPE的实现中，self-attention的keys在写入cache时**不施加RoPE**（存储raw key），RoPE在每次attention计算时由`block_relativistic_rope`函数实时施加相对位置编码。参见 `wan/modules/causal_model.py` 第249-250行。

这意味着：cache中任意帧的KV可以被赋予任意的relative position。模型感知到的是相对时间距离，而非绝对生成时刻。

由此推论：从历史第N帧检索KV放入当前attention context并赋予relative position $p$，与直接从position $p$ 处生成的帧，在模型看来是不可区分的——前提是总attention span不超过模型训练时见过的最大跨度。

### 2.2 算法原理

基于上述观察，本算法将KV cache从固定的position-indexed滑动窗口扩展为content-addressable memory。每步生成时：

1. 维护一个无界的Memory Store，存储全历史帧的KV及其content descriptor
2. 每个block生成前，基于当前query与历史帧的content相关性检索top-k帧
3. 将检索帧与local window组装后施加block-relative RoPE，总span控制在训练范围内
4. 利用attention weights的副产品作为memory utility信号，驱动遗忘/压缩/淘汰

### 2.3 与现有方法的关系

| 本算法组件 | 对应已有方法 | 已有验证结论 |
|-----------|------------|------------|
| Content-routed retrieval | CausalCine CAMR | Inter-Shot Consistency +23% |
| Block-relative RoPE (keys不旋转) | CausalCine + Infinity-RoPE | 模型正常工作 |
| Difference-aware decay | Echo-Forcing | Subject Consistency +9.24 |
| Scene compression | Echo-Forcing Scene Recall | 120秒不退化 |
| RoPE discontinuity | ShotStream | Transition Control 0.978 |

---

## 3. 数据结构

### 3.1 Local Cache（已有，不修改）

与Infinity-RoPE相同：

```
local_cache: {
    "k": Tensor[B, 9360, 12, 128],   # sink(1560) + window(7800)
    "v": Tensor[B, 9360, 12, 128],
    "local_end_index": int,
    "scene_cut": bool
}
```

共30层，每层一个这样的dict。

### 3.2 Memory Store（新增）

```python
class MemoryStore:
    descriptors: Tensor[N_frames, 12, 128]   # 每帧的content descriptor（GPU常驻）
    kv_data: List[{                           # 每帧的完整KV（CPU offload）
        "k": Tensor[1, 1560, 12, 128],       # 单帧KV，仅存储代表层
        "v": Tensor[1, 1560, 12, 128],
    }]
    utility: Tensor[N_frames]                 # 标量，累积被attend程度
    scene_id: Tensor[N_frames]               # 所属scene编号
    frame_id: Tensor[N_frames]               # 全局帧编号
```

**Content Descriptor 定义**：

对第$f$帧，选取layer $l^*$（代表层，选取方法见3.4），其在clean pass后写入cache的key为 $K_f \in \mathbb{R}^{1560 \times 12 \times 128}$。

$$d_f = \frac{1}{1560} \sum_{p=1}^{1560} K_f[p, :, :] \in \mathbb{R}^{12 \times 128}$$

即对spatial tokens取均值，得到每个head一个128维向量。

### 3.3 Scene Archive（新增）

```python
class SceneArchive:
    entries: List[{
        "scene_id": int,
        "descriptor": Tensor[12, 128],        # representative descriptor
        "compressed_kv": {                     # 压缩后的KV
            "k": Tensor[1, 1560, 12, 128],
            "v": Tensor[1, 1560, 12, 128],
        },
        "utility": float
    }]
```

### 3.4 代表层选取

不需要存储所有30层的KV到Memory Store中。选取少数代表层降低存储开销。

选取依据：从我们的head stability实验结果中，选取**稳定层**（L0-2, L12-19中选3层）和**volatile层**（L3-9中选2层），共5层作为代表层。

稳定层的descriptor编码身份信息（变化慢），volatile层的descriptor编码场景信息（变化快）。Routing时两者互补。

初始设定：代表层 = {0, 1, 14, 16, 5}。如存储预算允许，可增加到全部30层。

---

## 4. 算法流程

### 4.1 每个Block的生成流程

在Infinity-RoPE的block循环（`causal_inference.py` 第224行起）中，在标准的4步去噪之前插入retrieval步骤，在clean pass之后插入memory write步骤。

#### Step 1: Content-Routed Retrieval

输入：Memory Store中的全部descriptors，当前local cache中最后一帧的key。

计算query descriptor：
$$q = \frac{1}{1560} \sum_{p=1}^{1560} K_{last}[p, :, :] \in \mathbb{R}^{12 \times 128}$$

其中 $K_{last}$ 是local cache中最新一帧的raw key（来自前一block的clean pass）。

计算relevance score：对Memory Store中每个帧$f$：
$$s_f = \sum_{h=1}^{12} \sum_{d=1}^{128} q[h, d] \cdot d_f[h, d]$$

即head-aggregated的点积。复杂度为 $O(N_{frames} \times 12 \times 128)$。

选取 $\text{top-}k$ 帧（默认$k=5$），加载其完整KV到GPU。

#### Step 2: 组装Attention Context

将检索到的帧按原始生成时间排序后，与local cache和当前block拼接：

$$\text{Context} = [\underbrace{R_1, R_2, \ldots, R_k}_{\text{retrieved, sorted by time}}, \underbrace{W_1, \ldots, W_5}_{\text{local window}}, \underbrace{C_1, C_2, C_3}_{\text{current block}}]$$

赋予block-relative position：
$$\text{positions} = [0, 1, \ldots, k-1, \; k, k+1, \ldots, k+14, \; k+15, k+16, k+17]$$

总span = $k + 5 \times 3 + 3 = k + 18$。当$k=5$时span=23，远小于模型训练时的最大span（$F_{train} \approx 61$ latent frames for Infinity-RoPE）。

#### Step 3: 去噪 + Clean Pass

使用组装后的context进行标准的4步去噪和clean pass。代码逻辑不变，仅attention的KV输入增加了retrieved部分。

#### Step 4: Memory Write + Utility Update

Clean pass结束后：

1. 计算新生成3帧的content descriptor，写入Memory Store：
$$d_{new} = \frac{1}{1560} \sum_p K_{new}[p, :, :]$$
初始 utility = 0。

2. 更新retrieved帧的utility。在Step 3的attention计算中，当前block的query对retrieved帧分配了attention weights $\alpha_f$（softmax后的值）。将该值累加到对应帧的utility：
$$\text{utility}_f \leftarrow \text{utility}_f + \bar{\alpha}_f$$
其中 $\bar{\alpha}_f$ 是该帧在当前block所有query token上获得的平均attention weight。

**注意**：$\bar{\alpha}_f$ 是attention计算的副产品，不需要额外前向传播。但从FlashAttention中提取attention weights需要设置`return_attn_weights=True`或使用替代方案（见5.3节）。

### 4.2 Shot Transition

当prompt切换时（对应`causal_inference.py`中的`kv_flush`调用点），执行以下操作替代原始的`kv_flush`：

#### Step T1: Scene Compression

从Memory Store中选取当前scene的所有帧，按utility排序，取top-$M$帧（$M=3$）：

$$\{f_1^*, f_2^*, f_3^*\} = \text{top-}M\text{-by-utility}(\{f : f.\text{scene\_id} = s_{current}\})$$

按utility归一化加权融合为单帧KV：
$$w_i = \frac{\text{utility}_{f_i^*}}{\sum_j \text{utility}_{f_j^*}}$$
$$K_{compressed} = \sum_{i=1}^{M} w_i \cdot K_{f_i^*}$$
$$V_{compressed} = \sum_{i=1}^{M} w_i \cdot V_{f_i^*}$$

存入Scene Archive。

#### Step T2: Difference-Aware Local Cache Decay

对local cache中sink以外的entries，计算每个entry与新prompt的语义冲突度，决定衰减幅度。

冲突度计算：利用cross-attention cache中存储的旧prompt的text keys。对local cache中第$i$个entry：

$$\text{conflict}_i = 1 - \cos(d_i^{self}, d_{shared}^{cross})$$

其中：
- $d_i^{self}$：该entry的self-attention key mean
- $d_{shared}^{cross}$：新旧prompt中共有实体tokens对应的cross-attention key mean

如果新旧prompt没有共有实体（完全硬切），则 $\text{conflict}_i = 1$（全部衰减）。

衰减操作：
$$K_i \leftarrow (1 - \text{conflict}_i) \cdot K_i$$
$$V_i \leftarrow (1 - \text{conflict}_i) \cdot V_i$$

这使得与新prompt兼容的entries（如角色identity）保留，与旧场景强相关的entries（如背景）被抑制。

**简化版本**（如果上述cross-attention分析不易实现）：直接使用Echo-Forcing验证的固定策略——对所有non-sink entries施加均匀衰减系数$\beta = 0.3$：
$$K_i \leftarrow \beta \cdot K_i, \quad V_i \leftarrow \beta \cdot V_i$$

#### Step T3: RoPE Discontinuity Signal

设置 `kv_cache[layer]['scene_cut'] = True`，使`block_relativistic_rope`在下一block计算时引入时间坐标跳跃（Infinity-RoPE已有此机制）。

#### Step T4: Cross-Attention Reset

```python
for layer in range(n_layers):
    crossattn_cache[layer]['is_init'] = False
```

新prompt在下一次forward时自动重新编码。

#### Step T5: Local Cache 重置

```python
# 保留sink（第一帧），清空rolling window
for layer in range(n_layers):
    kv_cache[layer]['local_end_index'] = torch.tensor([1560], device=device)
```

**不需要Recache**：下一block生成时，Content-Routed Retrieval会自动从Memory Store中检索与新prompt相关的identity帧，提供跨镜头连续性。

### 4.3 Memory Consolidation（无限生成支持）

每完成一个shot（或当Memory Store帧数超过阈值$N_{max}=200$时），执行一次consolidation：

```
def consolidate(memory_store, budget=150):
    if len(memory_store) <= budget:
        return
    
    # 按utility排序
    sorted_frames = sort_by_utility(memory_store)
    
    # 保留top-50%（高utility = 被模型频繁attend = 重要）
    keep = sorted_frames[:budget//2]
    candidates = sorted_frames[budget//2:]
    
    # 在candidates中保留descriptor diverse的（冗余删除）
    for c in candidates:
        max_sim = max(cosine_sim(c.descriptor, k.descriptor) for k in keep)
        if max_sim < 0.9:
            keep.append(c)
        if len(keep) >= budget:
            break
    
    memory_store.retain_only(keep)
```

Scene Archive也做总量控制：保留最近10个scene的archive + 全局utility最高的5个archive entry。

---

## 5. 基于Infinity-RoPE代码的实现设计

### 5.1 需要修改的文件

| 文件 | 修改内容 |
|------|---------|
| `pipeline/causal_inference.py` | Block循环中插入retrieval/write逻辑，替换`kv_flush` |
| `wan/modules/causal_model.py` | `CausalWanSelfAttention.forward`中扩展KV输入维度 |
| 新文件: `memory_store.py` | Memory Store和Scene Archive的数据结构实现 |
| 新文件: `content_routing.py` | Retrieval和consolidation逻辑 |

### 5.2 关键代码修改点

#### 5.2.1 `causal_inference.py` Block循环修改

当前代码结构（简化）：
```python
for block_idx in range(num_blocks):
    # Step 3.1-3.2: 4步去噪
    for step in denoising_steps:
        output = generator.forward(noisy_input, timestep, kv_cache, crossattn_cache)
    
    # Step 3.3: Clean context pass
    _ = generator.forward(clean_output, timestep=0, kv_cache, crossattn_cache)
```

修改为：
```python
for block_idx in range(num_blocks):
    # [新增] Retrieval
    retrieved_kv = memory_store.retrieve(
        query=get_last_frame_descriptor(kv_cache),
        k=5
    )
    
    # Step 3.1-3.2: 4步去噪（传入retrieved_kv）
    for step in denoising_steps:
        output = generator.forward(noisy_input, timestep, kv_cache, crossattn_cache,
                                   retrieved_kv=retrieved_kv)
    
    # Step 3.3: Clean context pass
    _ = generator.forward(clean_output, timestep=0, kv_cache, crossattn_cache,
                         retrieved_kv=retrieved_kv)
    
    # [新增] Memory write
    new_descriptor = compute_descriptor(kv_cache, block_idx)
    memory_store.write(new_descriptor, kv_cache.last_block_kv())
    
    # [新增] Utility update（如可获取attention weights）
    memory_store.update_utility(retrieved_indices, attn_weights)
```

#### 5.2.2 `CausalWanSelfAttention.forward` 修改

当前实现中，attention的key/value来源于local cache：
```python
# 原始代码 (causal_model.py ~line 260-270)
key = torch.cat([kv_cache[layer]['k'][:, :local_end], current_key], dim=1)
value = torch.cat([kv_cache[layer]['v'][:, :local_end], current_value], dim=1)
```

修改为在local cache之前拼接retrieved KV：
```python
# 修改后
if retrieved_kv is not None:
    ret_k = retrieved_kv[layer]['k']  # [B, k*1560, 12, 128]
    ret_v = retrieved_kv[layer]['v']
    key = torch.cat([ret_k, kv_cache[layer]['k'][:, :local_end], current_key], dim=1)
    value = torch.cat([ret_v, kv_cache[layer]['v'][:, :local_end], current_value], dim=1)
else:
    key = torch.cat([kv_cache[layer]['k'][:, :local_end], current_key], dim=1)
    value = torch.cat([kv_cache[layer]['v'][:, :local_end], current_value], dim=1)
```

#### 5.2.3 `block_relativistic_rope` 修改

当前函数为local cache中的entries计算relative position。需扩展以支持retrieved帧前缀。

```python
# 原始: positions for [sink | window | current]
# 修改: positions for [retrieved | sink | window | current]

def block_relativistic_rope_with_retrieval(freqs, k_retrieved, n_local, n_current, scene_cut):
    """
    freqs: 预计算的RoPE频率表
    k_retrieved: retrieved帧数量
    n_local: local cache帧数
    n_current: 当前block帧数
    
    Position assignment:
    retrieved: [0, 1, ..., k_retrieved-1]
    local:     [k_retrieved, ..., k_retrieved + n_local - 1]
    current:   [k_retrieved + n_local, ..., k_retrieved + n_local + n_current - 1]
    
    如果有scene_cut，在local和current之间插入gap。
    """
    total_span = k_retrieved + n_local + n_current
    positions = torch.arange(total_span)
    
    if scene_cut:
        # current block的position额外加一个gap Δ
        positions[k_retrieved + n_local:] += SCENE_CUT_DELTA
    
    return compute_rope_from_positions(freqs, positions)
```

### 5.3 Utility Update的实现方案

FlashAttention默认不返回attention weights。有两种替代方案：

**方案A（推荐）：用descriptor similarity近似utility**

不获取真实attention weights，而是用content routing时已计算的relevance score作为utility的近似：

$$\text{utility}_f \leftarrow \text{utility}_f + s_f$$

每步routing时已经计算了所有帧的$s_f$，取top-k帧的$s_f$值累加即可。零额外计算。

**方案B：额外计算attention weights**

在clean pass中对retrieved帧单独做一次small-scale attention（不用FlashAttention，用naive attention on retrieved tokens only）获取weights。开销：$O(k \times 1560 \times 12 \times 128)$，约 $5 \times 1560 \times 1536 \approx 12M$ FLOPs per block，可忽略。

**本文档后续实验中使用方案A。**

### 5.4 存储开销估算

| 组件 | 内容 | 单帧大小 (bf16) | 200帧总计 |
|------|------|----------------|----------|
| Descriptor (GPU) | [12, 128] per frame | 3 KB | 600 KB |
| KV Data (CPU) | [5层, 2(KV), 1560, 12, 128] per frame | 47 MB | 9.4 GB |
| Utility (GPU) | 标量 per frame | 2 B | 400 B |

如果只存1个代表层（而非5层），CPU占用降至 9.4/5 ≈ 1.9 GB。

Descriptor常驻GPU用于routing计算（600KB可忽略）。KV数据在CPU，检索后按需load到GPU（每次load 5帧 × 47/5 ≈ 47 MB，PCIe传输 ~1ms）。

---

## 6. 验证实验

### 6.1 实验1（最关键）：Non-Contiguous Context Tolerance

**目的**：验证从远程历史检索帧放入block-relative position后模型能否正常生成。这是整个算法成立的前提。

**方法**：

生成一段单prompt视频 "A girl in red dress dancing in kitchen" [10s]（约30 latent frames = 10 blocks）。

在第8个block生成时，人为修改attention context：

| 条件 | Context组成 |
|------|-----------|
| A (baseline) | 正常local window（最近5帧） |
| B (retrieval) | block 2-4的KV + local window最近2帧 |
| C (random) | 随机选3个历史block的KV + local window最近2帧 |

所有条件下总attention span相同。B和C中的历史KV通过block-relative RoPE赋予position [0,1,2]，local window接续。

**评估指标**：
1. FID/FVD：B和C相对A的质量退化
2. DINO cosine similarity：B条件下第8 block生成的人物与block 2-4中人物的外观相似度
3. 人眼检查：是否有明显artifact

**预期结果**：
- B条件质量与A接近（无明显退化）且人物相似度显著高于A（因为直接attend到了角色历史）
- C条件可能有轻微退化（随机帧可能不相关），但不应产生catastrophic failure

**判断标准**：
- PASS：B条件 FVD ≤ A条件 × 1.1，且无可见artifact
- FAIL：B条件出现明显visual artifact或质量大幅下降

**代码实现要点**：
- 在`causal_inference.py`的block循环中，block 2-4结束时将其clean pass的KV保存到side buffer
- Block 8生成时，将side buffer的KV拼接到attention输入的前部
- 修改`block_relativistic_rope`使其为前缀帧分配position [0,1,2]

### 6.2 实验2：Content Descriptor语义准确性

**目的**：验证mean-pooled key能否区分同entity/不同entity的帧。

**方法**：

生成多prompt视频：
- Prompt 1: "A girl in red dress dancing in kitchen" [5s]
- Prompt 2: "A boy in blue jacket running in garden" [5s]
- Prompt 3: "A girl in red dress walking on beach" [5s]

生成过程中记录每个block的content descriptor（clean pass后的key spatial mean）。

计算所有帧对之间的descriptor cosine similarity，绘制heatmap。

**预期结果**：
- Prompt 1帧与Prompt 3帧（同entity不同scene）的相似度 > Prompt 1帧与Prompt 2帧（不同entity）的相似度
- 各shot内部帧之间相似度高（块状结构）

**判断标准**：
- PASS：inter-entity similarity < 0.7，same-entity cross-shot similarity > 0.8
- FAIL：所有帧pair相似度差异不显著（descriptor不携带entity信息）

**如果FAIL的后备方案**：
- 改用只取stable layers (L0-2, L12-19)的key mean作为descriptor
- 或使用cross-attention weighted key mean（对"girl"等entity token注意力高的spatial位置加权）

### 6.3 实验3：完整系统的多镜头一致性

**目的**：评估完整系统在多镜头生成中的跨镜头一致性。

**方法**：

使用5-shot benchmark：
```
Shot 1: "A girl in red dress dancing in kitchen" [5s]
Shot 2: "A gray cat sleeping on sofa in living room" [5s]
Shot 3: "A girl in red dress reading book in library" [5s]
Shot 4: "A gray cat playing with ball in garden" [5s]
Shot 5: "A girl in red dress and a gray cat sitting in park" [5s]
```

对比方法：
| 方法 | 配置 |
|------|------|
| Baseline | Infinity-RoPE原始kv_flush |
| LongLive-style | Sink(1帧) + Window(5帧)，无检索 |
| Ours (k=3) | Content routing, 检索3帧 |
| Ours (k=5) | Content routing, 检索5帧 |
| Ours (full) | Content routing + decay + compression |

**评估指标**：
1. Subject Consistency：DINO cosine similarity（同entity跨shot的外观相似度）
2. Scene Independence：CLIP similarity（不同shot的背景应该不同）
3. Visual Quality：LAION Aesthetic Score
4. Shot-Cut Accuracy：TransNetV2检测的切换时间与目标时间的偏差

**预期结果**：

| 方法 | Subject Cons. ↑ | Scene Indep. ↑ | Aesthetic ↑ | SCA ↑ |
|------|----------------|----------------|-------------|-------|
| Baseline (flush) | ~0.84 | 高 | 0.62 | ~0.78 |
| LongLive-style | ~0.93 | 中 | 0.62 | ~0.50 |
| Ours (k=5, full) | **~0.94-0.95** | **高** | **≥0.62** | **~0.95** |

关键预期：本方法同时达到高Subject Consistency（通过检索identity帧）和高Scene Independence（通过decay抑制旧场景），而LongLive-style牺牲了SCA（因为没有显式flush导致场景切换不清晰）。

### 6.4 实验4：无限生成稳定性

**目的**：验证Memory Consolidation使系统能持续生成而不退化。

**方法**：

生成60秒（~20个shot，每shot 3秒）视频，评估以下随时间变化的指标：
- 每5秒窗口的Aesthetic Quality
- Subject Consistency（每个shot与该entity首次出现时的相似度）
- Memory Store大小（验证consolidation有效控制）

对比：
- Infinity-RoPE baseline（预期20-30秒后开始退化）
- Echo-Forcing（报告120秒不退化）
- 本方法

**预期结果**：
- Aesthetic Quality在60秒内保持稳定（不下降超过0.02）
- Subject Consistency不随时间下降（consolidation保留了高utility的identity帧）
- Memory Store大小稳定在budget（150帧）附近

### 6.5 实验5（Ablation）：各组件贡献

| 配置 | Content Routing | Decay | Compression | Consolidation |
|------|:-:|:-:|:-:|:-:|
| A: 仅检索 | ✓ | × | × | × |
| B: 检索+衰减 | ✓ | ✓ | × | × |
| C: 检索+衰减+压缩 | ✓ | ✓ | ✓ | × |
| D: 完整系统 | ✓ | ✓ | ✓ | ✓ |
| E: 仅检索(k=0) | × | ✓ | ✓ | ✓ |

在实验3的benchmark上评估。

预期：
- A vs E：验证content routing的贡献（Subject Consistency差异）
- B vs A：验证decay对Scene Independence的贡献
- C vs B：验证compression对60秒+生成稳定性的贡献
- D vs C：验证consolidation对memory budget控制的贡献

---

## 7. 超参数汇总

| 参数 | 默认值 | 含义 |
|------|-------|------|
| $k$ | 5 | 每步检索的历史帧数 |
| $M$ | 3 | Scene Compression选取的representative帧数 |
| $N_{max}$ | 200 | Memory Store最大帧数（触发consolidation） |
| $N_{budget}$ | 150 | Consolidation后保留帧数 |
| Redundancy threshold | 0.9 | Consolidation中判断冗余的cosine sim阈值 |
| Decay $\beta$ (简化版) | 0.3 | Shot transition时non-sink entries的保留系数 |
| SCENE_CUT_DELTA | 3 | RoPE discontinuity的时间跳跃量（单位：帧） |
| 代表层 | {0, 1, 5, 14, 16} | 用于存储KV和计算descriptor的层 |

---

## 8. 实验执行优先级

```
Phase 1: 实验1 (Non-Contiguous Tolerance)
         ├─ PASS → Phase 2
         └─ FAIL → 调整k到1-2，或对retrieved帧attention logit加衰减系数重试

Phase 2: 实验2 (Descriptor Accuracy)
         ├─ PASS → Phase 3
         └─ FAIL → 改用stable-layer-only descriptor重试

Phase 3: 实验3 (Full System Multi-Shot)  +  实验5 (Ablation)

Phase 4: 实验4 (60秒无限生成)
```

实验1和2仅需修改inference代码的记录/分析逻辑（不需要完整系统实现），可在1-2天内完成验证。通过后再投入完整系统实现。

---

## 9. 与Infinity-RoPE代码的兼容性

本算法的设计确保与Infinity-RoPE的核心代码兼容：

1. **不修改模型权重**：所有操作在inference层面
2. **不修改去噪循环逻辑**：4步去噪 + 1次clean pass的流程不变
3. **不修改KV cache的基本结构**：local cache仍然是[sink + window]的格式
4. **复用已有的RoPE机制**：block_relativistic_rope已有scene_cut支持
5. **复用已有的flush接口**：在flush点替换为新的transition逻辑

主要新增代码：
- `memory_store.py`: ~150行（数据结构 + write/retrieve/consolidate）
- `content_routing.py`: ~80行（descriptor计算 + top-k选择）
- `causal_inference.py`修改: ~50行（block循环中的hook点）
- `causal_model.py`修改: ~30行（attention forward中拼接retrieved KV）

总计新增约310行代码，修改约80行。
