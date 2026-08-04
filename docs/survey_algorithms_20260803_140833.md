# Training-Free Multi-Shot / Multi-Event 视频生成算法详细总结

本文档详细记录了22篇training-free多镜头/多事件视频生成论文的算法设计，包含完整的公式、原理和实现细节。

---

## 1. CineWeaver (arXiv:2607.26529)

**标题**: CineWeaver: Training-Free Reference-Controllable Multi-Shot Long Video Generation for Cinematic Storytelling

**作者**: Yuyang Huang, Yabo Chen, Wenrui Dai, Ziyang Zheng, Haibin Huang, Chi Zhang, Junni Zou, Hongkai Xiong, Xuelong Li

**机构**: 上海交通大学、中国电信TeleAI

**基础模型**: Wan2.1-14B (多镜头T2V) / Phantom-14B (参考图引导)

### 核心问题

电影级视频生成需要同时满足：多镜头生成（视觉上截然不同的镜头+清晰切换）、精细参考图控制、长视频全局一致性。预训练模型的时间连续性偏置（RoPE使相邻帧高注意力 + 全局self-attention + 共享prompt + VAE temporal caching）阻止了镜头切割。

### 核心洞见

多镜头不需要学习新能力，只需在推理时显式打断时间连续性偏置。

### 算法组件

#### 4.1 Shot-aware RoPE with Gap Frames

标准DiT中RoPE的相对时间偏移决定注意力强度：

$$\text{Self-Attn}(i,j) \propto \mathbf{q}_i^\top R_{j-i} \mathbf{k}_j$$

|Δd|越小→注意力越高→时间连续。在相邻镜头间插入N_G=5个Gap Frames，将相对偏移增大为|Δd|+N_G。G-frame的token被完全mask掉，不参与attention。

#### 4.2 Masked Self-Attention with Transition Frames

每个镜头开头N_S=2帧为transition frames。Attention mask：

$$M_{i,j} = \begin{cases} 0, & \text{if } i \in S_k \text{ and } j \in V_k \\ 0, & \text{if } i \in U_k \text{ and } j \in \bigcup_m V_m \\ -\infty, & \text{otherwise} \end{cases}$$

- Transition frames (S_k)：仅attend本镜头video tokens → 边界隔离
- Non-transition frames (U_k)：attend所有镜头（除G-frames）→ 跨镜头全局上下文

#### 4.3 Shot-wise Cross-Attention and FFN

Cross-attention严格限制在镜头内部：

$$\text{Cross-Attn}(\mathbf{Q}_k, \mathbf{K}, \mathbf{V}) \rightarrow \text{Cross-Attn}(\mathbf{Q}_k, \mathbf{K}_k, \mathbf{V}_k)$$

FFN按镜头独立计算，避免残差特征跨镜头混合。

#### 4.4 Shot-wise VAE Decoding

将最终latents按镜头分割，独立解码（重置temporal cache）。Boundary Refinement：每镜头前N_S个transition latent frames替换为dummy frames后丢弃。

#### 4.5 Reference Token Routing

参考图通过VAE编码为clean latents（无噪声，time-invariant），patchify后拼接：

$$\mathbf{H}_t = [\mathbf{h}_t^{vid}; \mathbf{h}^{ref}]$$

扩展mask使reference tokens仅attend本镜头：

$$M_{i,j} = \begin{cases} 0, & i \in S_k, j \in (V_k \cup R_k) \\ 0, & i \in U_k, j \in (\bigcup_m V_m \cup R_k) \\ 0, & i \in R_k, j \in (V_k \cup R_k) \\ -\infty, & \text{otherwise} \end{cases}$$

#### 4.6 Anchor Memory Mechanism

从第一个segment第一镜头选N_A=5帧通过VAE编码为anchor tokens（无噪声，time-invariant）：

$$\mathbf{H}'_t = [\mathbf{h}_t^{vid}; \mathbf{h}^{ref}; \mathbf{h}^{anchor}]$$

Anchor-aware mask：Non-transition tokens额外attend anchor；Anchor tokens仅self-attend(A→A)保持表示纯净。后续segment可并行生成（仅依赖固定anchor）。

### 推理流程

1. 预处理：编码reference images，插入gap frames，标记transition frames，构造mask
2. Denoising：masked self-attention + shot-wise cross-attention + shot-wise FFN
3. 后处理：按镜头分割latents → 独立VAE解码 → boundary refinement → 拼接

---

## 2. EM-Vid (arXiv:2605.23610)

**标题**: EM-Vid: Training-Free Entity-Centric Memory for Efficient and Consistent Multi-Shot Video Generation

**作者**: Jente Vandersanden, Matheus Gadelha, Chun-Hao P. Huang, Hyeonho Jeong, Yulia Gryaditskaya

**机构**: Max Planck Institute for Informatics, Adobe Research

**基础模型**: Wan2.2-I2V + StoryMem LoRA (M2V模型)

### 核心问题

现有memory-frame方法将持久实体信息与瞬时场景上下文纠缠，导致背景泄漏、计算成本高、模型易copy-paste而非遵循prompt。

### 核心洞见

将记忆从"帧级别"提升到"实体级别"——只存储与各实体相关的稀疏VAE latent patches。

### 算法组件

#### Entity Bank初始化

1. **实体分割**：SAM3对角色/物体分割；场景取前景补集
2. **VAE编码+patch对齐**：编码到latent空间，mask下采样对齐patch grid
3. **外观描述符**：DINOv2 embedding + CLIP相似度
4. **存储**：每个被选patch的latent值、坐标(x,y)、帧索引、DINOv2描述符、CLIP分数

#### 推理时Scatter-Prune-Scatter

1. **Scatter**：将稀疏patches散布回完整Memory Canvas的原始时空位置
2. **Patchify + Tokenize**：对dense latent应用模型卷积patchification
3. **Prune**：生成binary token mask，只保留包含实体的tokens（减少88.9% memory tokens）
4. **DiT处理**：稀疏kept tokens送入Video DiT
5. **Unpatchify + 更新**：输出scatter回dense layout，unpatchify回VAE latent空间

#### Entity Bank更新

接受准则（二选一）：
- bank为空
- `s_max^DINO(c_i^e) ∈ [τ_minmatch, τ_redundant]`

Budget管理——relevance-to-cost ratio贪心选择：

$$s^{\text{keep}}(c_i^e) = s^{\text{CLIP}}(c_i^e) / N_{\text{keep}}(c_i^e)$$

第一个条目永不移除（防止漂移）。

#### 噪声注入控制

- **背景泄漏缓解**：对边界patch中非实体区域在VAE编码前注入pixel-space高斯噪声
- **选择性外观保留**：当prompt要求修改实体属性时，移除完全在修改区域内的patches，对边界patches注入噪声

### 推理流程

逐镜头自回归：解析实体标识符 → 从entity bank检索 → 构建Memory Canvas → Scatter-Prune → DiT去噪 → 更新entity bank

---

## 3. IAMFlow (arXiv:2605.18733)

**标题**: Advancing Narrative Long Video Generation via Training-Free Identity-Aware Memory

**作者**: Jinzhuo Liu, Jiangning Zhang, Wencan Jiang等

**机构**: 浙江大学、腾讯优图、华中科技、上海交通大学

**基础模型**: MemFlow (基于Wan2.1的AR扩展) + Qwen3-4B (实体提取) + Qwen3-VL-2B (视觉验证)

### 核心问题

AR视频生成的三大瓶颈：有限历史记忆、交互式提示适应、低效上下文扩展。根本原因是缺乏对持久实体的显式状态变量。

### 算法组件

#### Identity-Aware Memory Preservation

维护三个数据结构：全局实体注册表R、帧存档F、活跃身份记忆m^{id}。

**四阶段运作**：

**阶段一：实体提取与ID分配**
LLM解析prompt为结构化实体描述符，与注册表比对分配/匹配ID。

**阶段二：身份感知帧评分**

Entity-token权重向量构建ω（实体名称token权重2.5，其余按位置1.0/0.7/0.5）。

Entity Score：
$$s_{\text{entity}}(f) = \frac{1}{H} \sum_{h=1}^H \frac{\langle \mathbf{r}_{\text{id},h}, \bar{\mathbf{k}}_{f,h} \rangle}{\sqrt{d}}$$

融合：$s(f) = (1-\lambda)\hat{s}_{\text{entity}}(f) + \lambda \cdot s_{\text{visual}}(f)$，λ=0.3

**贪心集覆盖检索**：建模为最大覆盖问题，贪心近似确保每个活跃身份至少一帧代表。

**阶段三：异步视觉验证**
VLM异步评分+属性校正，利用3-chunk逐出滞后隐藏延迟。

#### Adaptive Prompt Transition (APT)

在cross-attention KV条件空间中平滑切换：

$$\mathbf{K}_\tau = (1-\alpha_\tau)\mathbf{K}_{\text{old}} + \alpha_\tau \mathbf{K}_{\text{new}}$$

余弦调度：
$$\alpha_\tau = \frac{1}{2}\left(1 - \cos\left(\pi \frac{\tau - d_{\text{delay}}}{W_{\text{apt}}}\right)\right)$$

自适应窗口：$W_{\text{apt}} = \text{snap}(W_{\min} + \delta(W_{\max} - W_{\min}))$，δ为相邻prompt embedding余弦距离。

---

## 4. Prompt Relay (arXiv:2604.10030)

**标题**: Prompt Relay: Inference-Time Temporal Control for Multi-Event Video Generation

**作者**: Gordon Chen, Ziqi Huang, Ziwei Liu

**机构**: S-Lab, Nanyang Technological University

**基础模型**: Wan2.2-T2V-A14B

### 核心问题

Cross-attention让每帧所有像素同时关注整个prompt所有token，缺乏时间感知，导致多事件语义纠缠。

### 算法组件

#### Temporal Prompt Routing

在cross-attention logits中引入Gaussian惩罚项：

$$\text{Attn} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} - C(Q, K)\right) V$$

惩罚函数：
$$C(i, j) = \frac{\text{ReLU}(|f(i) - m_s| - w)^2}{2\sigma^2}$$

- f(i)：query token对应的latent帧索引
- m_s：prompt p_s对应时间段中点
- w：自由注意力窗口（默认w=L-2）
- σ：衰减速率

窗口内惩罚为零；窗口外Gaussian衰减。

#### Boundary-Attention Decay

选择σ使端点注意力衰减到ε=0.1：

$$\sigma = \frac{L - w}{\sqrt{2\ln(1/\epsilon)}}$$

当w=L-2时σ简化为常数。Soft decay在边界处同时激活相邻prompt，允许模型联合规划过渡。

### 推理流程

每个去噪步：计算标准attention logits → 对每个query-key对计算惩罚C(i,j)（全局prompt不惩罚）→ 减去惩罚 → softmax → value加权。Self-attention完全不变。

---

## 5. BachVid (arXiv:2510.21696)

**标题**: BachVid: Training-Free Video Generation with Consistent Background and Character

**作者**: Han Yan, Xibin Song, Yifu Wang, Hongdong Li, Pan Ji, Chao Ma

**机构**: 上海交通大学、Vertex Lab、澳大利亚国立大学

**基础模型**: CogVideoX-5B (42层DiT)

### 核心问题

多视频间保持角色和背景的双重一致性，无需参考图像、无需训练。

### 核心洞见

DiT的cross-attention天然编码前景/背景分离，attention output编码语义对应关系。

### 算法组件

#### 前景Mask提取

Prompt结构化为"[Background],[Character],[Action]"。比较背景/前景attention权重：

$$\mathcal{M}^{t,l} = \left(\frac{1}{L_{bg}} \sum_i W_{bg}^{t,l}[:,i]\right) \leq \left(\frac{1}{L_{fg}} \sum_i W_{fg}^{t,l}[:,i]\right)$$

使用top-15层(indices 6-20)在timestep τ_mask=10聚合得到鲁棒mask。

#### 匹配点识别

帧级余弦相似度矩阵：
$$S^{t,l} = \tilde{O}_{frm}^{t,l} (\tilde{O}_{id}^{t,l})^T \in \mathbb{R}^{T \times HW \times HW}$$

使用bottom-15层(indices 2-16)聚合：
$$S_{match} = \sum_{l \in \mathcal{L}_{match}} O_{frm}^{\tau_{match},l} (O_{id}^{\tau_{match},l})^T$$

#### Vital Layers确定

跳过第l层生成视频，计算美学评分。L_kv = {1,2,12,13,14,15,16,18,20,21,22,24,30,35,42}（共15层）。

#### KV注入+RoPE重编码

对vital layers在每个timestep：
1. 按mask+匹配映射确定对应索引
2. 提取identity video对应位置的KV
3. **RoPE重编码**（关键）：用frame video位置索引重新编码

$$K'_{id,fg} = \text{RoPE}(K_{id,fg}, I_{frm,fg})$$

4. 拼接：$K^*_{frm} = K'_{frm} \oplus K'_{id,fg} \oplus K'_{id,bg}$
5. 带mask注意力：前景只attend前景，背景只attend背景

---

## 6. VideoMerge (arXiv:2503.09926)

**标题**: VideoMerge: Towards Training-free Long Video Generation

**作者**: Siyang Zhang, Harry Yang, Ser-Nam Lim

**机构**: University of Central Florida, HKUST

**基础模型**: HunyuanVideo (DiT)

### 核心问题

长视频生成中计算-训练不匹配问题和身份一致性问题。

### 算法组件

#### Multi-Tile Latent Fusion（正弦加权）

融合公式：
$$\varepsilon_t = \frac{\sum_{i=j}^{k} \omega(t - i(n-o)) \cdot \epsilon_i}{\sum_{i=j}^{k} \omega(t - i(n-o))}$$

正弦权重：
$$\omega(s) = \sin\left(\frac{s\pi}{n} + \frac{\pi}{2n}\right), \quad s \in \mathbb{Z}_n$$

权重在tile中央最大、边缘趋零，相邻tile自然交叉淡入淡出。

#### Long Noise Initialization

1. 短噪声复制n次构建长噪声（保持低频一致）
2. 非重叠区域shuffle（引入随机性）
3. **频域高频混合**：3D FFT分离低频/高频 → Butterworth LPF(阈值0.25, 阶数4) → 新随机噪声高频与原始高频按渐进权重(0→w_max=0.1)混合 → 逆FFT

归一化因子：$d = \sqrt{w^2 + (1-w)^2}$

效果：低频保身份一致，高频渐进替换保动态多样。

#### Prompt Refining

用ChatGPT-O3扩写简短prompt为包含外观/场景/光照细节的详细描述，减少幻觉自由度。

---

## 7. Scene-Action Prompt Fusion (arXiv:2503.06310)

**标题**: Scene-Action Prompt Fusion for Coherent Text-to-Video Storytelling

**作者**: Taewon Kang, Divya Kothandaraman, Ming C. Lin

**机构**: University of Maryland, Dolby Laboratories

**基础模型**: Mochi-1

### 算法组件

#### DIPW (Dynamics-Informed Prompt Weighting)

三信号自适应权重：
$$s^i_{2N-1} = \lambda_1 \text{sim}^i_{2N-1} + \lambda_2 \text{prev\_sim}^i_{2N-1} + \lambda_3 \text{prior}^i_{2N-1}$$

- CLIP相似度（当前对齐）
- 余弦相似度（时间平滑）
- 线性递进先验：$\text{prior}^i_{scene} = 1 - i/T$, $\text{prior}^i_{action} = i/T$

温度softmax归一化(τ=0.5)后混合conditioning：
$$\mathbf{E}_i = \alpha^i_{2N-1} \mathbf{E}_{P_{scene}} + \alpha^i_{2N} \mathbf{E}_{P_{action}}$$

#### TWB (Time-Weighted Latent Space Blending)

指数衰减加权：$w_i = \beta^{(K-i-1)}$

$$z_{new,0} \leftarrow \gamma \cdot z_{prev,T} + \gamma \cdot \bar{z}_{new,0} + (1 - 2\gamma) \cdot z_{new,0}$$

仅在片段转换时执行一次。

#### SAR (Semantic Action Representation)

动作相似度调制：$\alpha' = \alpha \cdot (1 - S_A(P_{scene}, P_{action}))$

动作越相似→blending越弱（自然过渡不需额外耦合）。

---

## 8. MEVG (arXiv:2312.04086)

**标题**: MEVG: Multi-event Video Generation with Text-to-Video Models

**作者**: Gyeongrok Oh等

**机构**: Korea University, NVIDIA

**发表**: ECCV 2024

**基础模型**: LVDM

### 算法组件

#### Last Frame-aware Latent Initialization

**(a) Dynamic Noise**：噪声调度$\mathcal{F}(n) = \exp(-n)$，单调递减。

$$\epsilon_t^{inv_p}[n] = \frac{\kappa_n}{\sqrt{1+\kappa_n^2}} \epsilon_t^{inv_p}[n] + \epsilon_t^{dyn}$$

开头帧κ大（约束强，接近前帧），结尾帧κ小（自由度高，变化大）。

**(b) LFAI正则化**：

$$\mathcal{L}_{\text{LFAI}} = ||\hat{x}_t^{sam_{p-1}}[-1] - \hat{x}_t^{inv_p}[0]||_2^2$$

梯度更新使当前视频第一帧对齐前一视频最后帧。

#### Structure-guided Sampling (SGS)

$$\mathcal{L}_{\text{SGS}} = ||\hat{x}_t^{sam_p}[1:n] - \hat{x}_t^{sam_p}[:n-1]||_2^2$$

每步逐帧迭代使去噪观察帧间结构一致。δ_LFAI=1000, δ_SGS=7。

---

## 9. Gen-L-Video (arXiv:2305.18264)

**标题**: Gen-L-Video: Multi-Text to Long Video Generation via Temporal Co-Denoising

**作者**: Fu-Yun Wang, Wenshuo Chen等

**机构**: Shanghai AI Lab, CUHK, Sensetime, NJU, Tsinghua

**基础模型**: Stable Diffusion / VideoCrafter (LDM)

### 核心算法：Temporal Co-Denoising

将长视频视为时间重叠短片段的集合，联合去噪。

映射$F_i(v_t) = v_t^i = v_{t, S*i:S*i+M}$，S=stride, M=单片段帧数。

每步对所有片段独立去噪后，通过加权聚合恢复长视频——闭式最优解：

$$v_{t-1,j} = \frac{\sum_{i \in \mathcal{I}^j} (W_{i,j^*})^2 \otimes v_{t-1,j^*}^i}{\sum_{i \in \mathcal{I}^j} (W_{i,j^*})^2}$$

最优重叠：S=M/2或S=M/4。

#### Bi-Directional Cross-Frame Attention

Anchor frame设为中间帧（非第一帧），双向信息传播。重叠区域的anchor frames相互影响，全局传播一致性。

#### Condition Interpolation

稀疏标注prompt间线性插值：$c^{ki+j} = \frac{k-j}{k} c^{ki} + \frac{j}{k} c^{k(i+1)}$

---

## 10. Infinity-RoPE (arXiv:2511.20649)

**标题**: Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout

**作者**: Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, Pinar Yanardag

**机构**: Virginia Tech, fal

**基础模型**: Wan2.1-T2V-1.3B (Self-Forcing蒸馏)

### 三个核心组件

#### Block-Relativistic RoPE

将时间编码从绝对坐标改为移动参考系。当前生成块固定在f_0位置，之前帧向后旋转：

$$\bar{\mathbf{B}}_i = \begin{cases} \mathbf{B}_i, & \text{if } i \leq f_0 \\ \mathbf{B}_{f_0} = \{f_0-2, f_0-1, f_0\}, & \text{otherwise} \end{cases}$$

**Unbounded cache**：超出f_limit后最早帧坐标坍缩为共享最小索引（语义化）：
$\mathbf{B}_3 = \{1,2,3\} \rightarrow \mathbf{B}_1 = \{1,1,1\}$

#### KV Flush

新prompt到来时flush所有cached tokens，仅保留：
- Global sink（第一帧latent）：稳定attention normalization
- 最后生成的latent frame：保持局部时间连续性

实现即时动作响应 + 平滑时间连续性。

#### RoPE Cut

对当前块执行时间坐标不连续跳跃：
$$\mathbf{B}_{f \to f+\Delta} = \{f-2, f+\Delta-1, f+\Delta\}$$

Δ为场景间时间gap。Attention map分裂为两个不相交对角块，实现单次连续rollout中的场景转换。

### 关键参数

- Cache size: 6帧最优
- Onset index f_0 = 21
- CFG scale: 3.0
- 自回归块大小: 3帧

---

## 11. Keyframe-Anchored Identity Preservation (arXiv:2607.17985)

**标题**: Keyframe-Anchored Identity Preservation for Sequential-Action Video Generation

**作者**: Zhenjie Liu, Binyan Chen, Hao Chen, Tong Pan, Shangfei Wang

**机构**: 中国科学技术大学

**发表**: MM '26 (ACM Multimedia 2026)

**基础模型**: Z-Image (关键帧) + LTX-2.3 (视频插值) + Qwen3.6-27B (prompt改写)

### 算法组件

#### Stage 1: Action-Aware Prompt Polishment

LLM将视频级叙述改写为每个动作段终态的静态画面描述。强调终态而非过程。

#### Stage 2: ID-Preserving Chained Keyframe Generation

链式生成关键帧K_0,...,K_N。三输入正交约束：
- I_ref：身份恒常性（时间不变）
- K_{i-1}：时间变化状态（姿态/朝向）
- ĉ_i：目标终态描述

#### Stage 3: Multi-Reference Guidance

扩展采样为多模态引导：

$$\tilde{\epsilon}_\theta(x_t) = \epsilon_\theta(x_t|C) + w_{\text{cfg}}\Delta_{\text{cfg}} + w_{\text{stg}}\Delta_{\text{stg}} + w_{\text{mod}}\Delta_{\text{mod}}$$

- Δ_cfg：标准CFG（语义对齐）
- Δ_stg：时空引导（扰动block 29）
- Δ_mod：模态引导（放大视觉条件的身份信息）

w_cfg=5.0, w_stg=1.5, w_mod=3.0

#### Noise Searching

采样K个候选噪声，短程前瞻去噪后选择人脸相似度最高的：

$$R(z_T^{(k)}) = \text{sim}(\phi(D(z_{T-T'}^{(k)})), \phi(I_{\text{ref}}))$$

---

---

## 12. TS-Attn (arXiv:2604.19473)

**标题**: TS-Attn: Temporal-wise Separable Attention for Multi-Event Video Generation

**作者**: Hongyu Zhang, Yufan Deng, Zilin Pan等

**机构**: 北京大学深圳研究生院、浙江大学、南开大学、MIT、南京大学

**发表**: ICLR 2026

**基础模型**: CogVideoX-5B / Wan2.1-T2V-14B / Wan2.2-T2V-A14B

### 核心问题

单一复杂prompt的多事件生成存在时序错位和注意力耦合——不同事件动词在同一帧同时强响应，模型无法区分时序顺序。

### 算法组件

#### Motion Region Extraction (MRE)

计算主体语义图并阈值化：

$$A_s = \text{Mean}\left(\mathcal{I}_s\left(\frac{QK^\top}{\sqrt{d}}\right)\right)$$

$$M_s = \mathcal{F}_{\mathcal{K}}\left(\mathbb{I}(A_s \geq \text{Mean}(A_s))\right)$$

腐蚀核K=3去除散点噪声。

#### Event-aware Attention Modulation (EAM)

完整调制公式：
$$A = \text{softmax}\left(\frac{QK^\top + M_s \odot \mathcal{R}(Q,K) \odot \mathcal{B}(Q,K)}{\sqrt{d}}\right)$$

**Attention Rearrangement（偏置）**：

正偏置：$b_i^+ = \max(Q_i K^\top) - \text{mean}(Q_i K^\top)$

负偏置：$b_i^- = \min(Q_i K^\top) - \text{mean}(Q_i K^\top)$

$$\mathcal{B}(Q_i, K)[x, y] = \begin{cases} b_i^+, & y \in e_i \text{（对齐事件）} \\ b_i^-, & y \in e_j, i \neq j \text{（非对齐事件）} \\ 0, & \text{otherwise} \end{cases}$$

**Attention Reinforcement（强化）**：

归一化注意力强度：$p_i' = (p_i - p_i^{\min}) / (p_i^{\max} - p_i^{\min} + \epsilon)$

$$r_i^+ = r^{\min} + (1 - p_i') \cdot (r^{\max} - r^{\min})$$
$$r_i^- = r^{\min} + p_i' \cdot (r^{\max} - r^{\min})$$

r_min=1.0, r_max=1.5。原始注意力越弱→正向强化越大；不应有的注意力越强→负向抑制越大。

### 关键设计

- 仅在去噪前20%(T2V)/40%(I2V)步骤的中间层cross-attention应用
- 推理时间仅增加2%
- 粗粒度时序分段即足够（均匀分段≈LLM规划）
- 多主体时迭代提取各主体mask并汇总

---

## 13. CoNo (arXiv:2406.05082)

**标题**: CoNo: Consistency Noise Injection for Tuning-free Long Video Diffusion

**作者**: Xingrui Wang, Xin Li, Zhibo Chen

**机构**: 中国科学技术大学

**基础模型**: VideoCrafter1 / Lavie

### 核心问题

无训练长视频生成中(1)片段间场景转换粗糙，(2)缺乏显式长期内容一致性建模。

### 算法组件

#### Long-term Consistency Regularization

预测噪声帧平均值编码"内容成分"：

$$g(\hat{\epsilon}_{t,content}^{P_1}, \hat{\epsilon}_{t,content}^{P_2}) = \left\|\frac{\sum_{i=0}^{N-1}\hat{\epsilon}_t^{P_1}}{N} - \frac{\sum_{i=0}^{N-1}\hat{\epsilon}_t^{P_2}}{N}\right\|_2^2$$

梯度更新：
$$\hat{\epsilon}_t^{P_2} \leftarrow \hat{\epsilon}_t^{P_2} - \delta \nabla_{\hat{\epsilon}_t^{P_2}} g$$

δ=140(VideoCrafter1) / 260(Lavie)。

#### "Look-Back" 三阶段机制

**阶段1 - Video Extending**：
- Extending Noise Shuffle：$z_T^{P_1}$整体反转+前N_1帧局部反转
- 效果：第一片段最后N_1帧噪声=第二片段前N_1帧噪声
- 引导：t≥T_d时前N_1帧预测噪声替换为第一片段已存储的

**阶段2 - Internal Noise Prediction**：
- 双侧约束：左侧N_1帧+右侧N_2帧同时被已生成内容约束
- 利用视频扩散模型内在时序插值能力生成自然过渡帧

**阶段3 - 进一步扩展**：可迭代添加更多prompt

### 关键参数

T_d=10, N_1=6, N_2=8, DDIM 50步, CFG=15

### 核心洞见

- 噪声帧平均=内容成分，帧间差异=运动信息
- 保持相同初始噪声集合对一致性至关重要
- 双侧约束优于单侧（避免渐进式漂移）
- 仅高噪声步(t≥T_d)替换噪声，低噪声步放开保证细节质量

---

---

## 14. DiTCtrl (arXiv:2412.18597)

**标题**: DiTCtrl: Exploring Attention Control in Multi-Modal Diffusion Transformer for Tuning-Free Multi-Prompt Longer Video Generation

**作者**: Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, Xiangyu Yue

**机构**: 香港中文大学MMLab、大湾区大学GVC Lab、腾讯PCG ARC Lab、腾讯AI Lab

**基础模型**: CogVideoX-2B (MM-DiT架构)

### 核心问题

首个在MM-DiT架构上实现无训练多提示词连贯长视频生成。MM-DiT将文本和视频映射到统一序列进行联合注意力，与UNet的独立cross-attention不同。

### 核心发现

MM-DiT注意力矩阵可分解为四个区域：Text-to-Text、Text-to-Video、Video-to-Text、Video-to-Video。其中Text-Video和Video-Text注意力类似UNet的cross-attention，可实现token级语义定位。

### 算法组件

#### Mask Extraction（掩码提取）

从MM-DiT的Text-Video和Video-Text注意力中提取前景对象语义掩码：
1. 对指定object token对应的注意力值在所有head和layer上取平均
2. 将注意力图reshape为 $F \times H \times W$
3. 二值化阈值处理（threshold=0.3）得到掩码 $M$

#### Mask-Guided KV-Sharing（核心机制）

前景注意力输出（物体区域仅从前一视频的物体区域查询）：
$$f_o^l = \text{Attention}(Q_i^l, K_{i-1}^l, V_{i-1}^l; M_{i-1})$$

背景注意力输出：
$$f_b^l = \text{Attention}(Q_i^l, K_{i-1}^l, V_{i-1}^l; 1 - M_{i-1})$$

最终注意力融合：
$$\bar{f}^l = f_o^l * M_i + f_b^l * (1 - M_i)$$

- $Q_i^l$：当前分支第$l$层Query
- $K_{i-1}^l, V_{i-1}^l$：前一分支第$l$层Key/Value
- 仅在kv-sharing layers [25,30]、kv-sharing steps [2,25] 内执行

#### Latent Blending（潜变量混合过渡）

重叠区域位置相关加权：
$$\mathbf{z}_t = \frac{\sum_{i=1}^{n} w(t_i) \cdot \mathbf{z}_{t_i}}{\sum_{i=1}^{n} w(t_i)}$$

对称三角权重函数：
$$w(t_i) = \min\left(\frac{2(t_i + 0.5)}{T}, 2 - \frac{2(t_i + 0.5)}{T}\right)$$

靠近片段中心权重高，边界处权重低。

### 推理流程

两个提示词分支作为一个batch并行送入MM-DiT去噪：
1. 每步在[2,25]步范围内提取掩码
2. 在[25,30]层范围内执行mask-guided KV共享
3. 重叠帧（6帧/13帧总帧）通过三角权重混合
4. 最终通过3D Causal VAE解码

### 关键参数

采样步数50, guidance scale=6, 分辨率480×720, 潜变量帧数13, 重叠6帧, KV共享步[2,25], KV共享层[25,30], 掩码阈值0.3

### 评估指标CSCV

$$\text{score} = \frac{1}{1 + \lambda \cdot \frac{\sigma(s)}{\mu(s)}}$$

λ=10，衡量帧间CLIP相似度的变异系数。

---

## 15. TunerDiT (arXiv:2605.31590)

**标题**: TunerDiT: Training-free Progressive Steering of Diffusion Transformer for Consistent Multi-Event Video Generation

**作者**: Ruotong Liao, Guowen Huang, Qing Cheng, Guangyao Zhai, Lei Zhang, Xun Xiao, Thomas Seidl, Daniel Cremers, Volker Tresp

**机构**: LMU Munich、TU Munich、MCML、University of Hamburg、Huawei European Research Institute

**基础模型**: OpenSora 1.2 (ST-DiT) / OpenSora 2.0 (Dual-stream DiT) / Wan 2.2

### 核心问题

多事件视频生成的三大挑战：事件排序（Event Ordering）、平滑过渡（Smooth Transitions）、语义一致性（Semantic Consistency）。现有DiT在多事件prompt下出现Event Fusion、Scrambling Order、Transition Collapse三种失败模式。

### 核心发现：内在转折点（Intrinsic Turning Point）

DiT去噪存在从粗到细的转折点——早期步骤决定全局布局，晚期步骤细化细节。设融合比例$x \in [0,1]$：

$$\text{cond}(n) = \begin{cases} P_1, & n < xN \\ P_2, & n \geq xN \end{cases}$$

转折点$\tau = \lfloor xN \rfloor$对应Text-Video Alignment分数交叉点，通常在前~30%步骤内。

### 算法组件

#### Cross-Event Prompt Fusion (PF)

时间门控文本条件：
$$\widehat{T}_e(n) = \begin{cases} T_1, & n < \tau_{\text{PF}} \\ T_e, & n \geq \tau_{\text{PF}} \end{cases}$$

- 粗布局阶段($n < \tau_{\text{PF}}$)：所有事件段统一条件在$T_1$上→全局一致布局
- 细节分化阶段($n \geq \tau_{\text{PF}}$)：各事件段条件在各自$T_e$上

#### Event-Partitioned Diagonal Mask (EM)

对角掩码使各视频段仅关注匹配文本token：
$$D_{\text{diag}} = \text{diag}(V_1 \leftrightarrow T_1, \; V_2 \leftrightarrow T_2, \; \ldots, \; V_E \leftrightarrow T_E)$$

**过渡带（Transition Bands）**：避免硬切割
- Inner-event band $D_{\text{inner}}^{v \leftrightarrow v}$：边界视频latent在自注意力中交换信息
- Inter-event band $D_{\text{inter}}^{v \leftrightarrow t}$：边界视频token关注相邻事件文本

带宽计算：$h_e^{vv} = h_e^{vt} = \lfloor r|V_e| \rfloor$, $w_e^{vt} = \lfloor r|T_e| \rfloor$

EM时间门控：
$$D_{\text{EM}}^{v \leftrightarrow t}(n) = \begin{cases} \mathbf{1}, & n < \tau_{\text{EM}} \\ D_{\text{diag}} \vee D_{\text{band}}, & n \geq \tau_{\text{EM}} \end{cases}$$

加性注意力掩码：$M_{\text{EM}}(n) = (-\infty) \cdot (\mathbf{1} - D_{\text{EM}}(n))$

### 推理三阶段

**Phase A — 粗布局**($n < \tau_{\text{PF}}$)：共享$T_1$，无掩码→建立全局一致场景布局

**Phase B — 过渡**($\tau_{\text{PF}} \leq n < \tau_{\text{EM}}$)：各自$T_e$，无掩码→开始分化但保持连贯

**Phase C — 细粒度分离**($n \geq \tau_{\text{EM}}$)：各自$T_e$，对角掩码+过渡带→强制事件边界

### 关键参数

| 模型 | $x$(转折点) | $\tau_{\text{EM}}$ | $\tau_{\text{PF}}$ | $\beta$(带宽) |
|------|------------|-------------------|-------------------|--------------|
| OpenSora 1.2 | 0.1 | $\lfloor 0.1N \rfloor$ | $\lfloor 0.1N \rfloor$ | 0.1 |
| OpenSora 2.0 | 0.2 | $\lfloor 0.2N \rfloor$ | $\lfloor 0.1N \rfloor$ | 0.2 |
| Wan 2.2 | 0.2 | $\lfloor 0.2N \rfloor$ | — | 0.2 |

### 架构适配

- ST-DiT (OpenSora 1.2)：EM仅应用于时间注意力块
- Dual-stream DiT (OpenSora 2.0)：掩码同时作用于cross-attention和self-attention（统一序列）

---

## 16. SwitchCraft (arXiv:2602.23956)

**标题**: SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls

**作者**: Qianxun Xu, Chenxi Song, Yujun Cai, Chi Zhang

**机构**: 西湖大学、昆山杜克大学、昆士兰大学

**基础模型**: Wan 2.1 T2V 14B (40层DiT, 40 heads)

### 核心问题

统一的prompt注入忽略了事件与帧之间的时间对应关系——cross-attention让所有帧同等关注整个文本序列，导致Event Fusion、Omission、Leakage。

### 算法组件

#### Event-Aligned Query Steering (EAQS)

**锚点识别**：LLM从多事件prompt中提取每个事件的anchor phrases（区分性短语），映射为文本token索引集合。

**时间窗口分配**：
$$N_i \approx F' \cdot \frac{w_i}{\sum_{j=1}^{A} w_j}$$

$w_i$为事件时长权重（默认等分）。

**投影器构造**（Ridge正则化）：
$$P_\text{tgt} = K_\text{tgt}^\top (K_\text{tgt} K_\text{tgt}^\top + \epsilon I)^{-1} K_\text{tgt}$$

类似构造$P_\text{oth}$。将query分解为目标事件子空间$\mathcal{T}$和竞争事件子空间$\mathcal{O}$的分量。

**Query更新**：
$$Q^* \leftarrow Q^* + \alpha \cdot Q^* P_\text{tgt} - \beta \cdot Q^* P_\text{oth}$$

- $\alpha$增强目标事件子空间分量
- $\beta$抑制竞争事件子空间分量
- 更新后row-wise renormalization

#### Auto-Balance Strength Solver (ABSS)

自适应求解$\alpha, \beta$，无需手动调参。

**SVD压缩**：对各事件key做SVD取主方向单位向量$k_\text{tgt}, k_{\text{oth},j}$。

**对齐得分与margin deficit**：
$$S_\text{tgt} = Q^* k_\text{tgt}, \quad S_\text{oth} = Q^* k_\text{oth}$$
$$d = S_\text{oth}^\text{max} - S_\text{tgt} + \varepsilon$$

**阻力矩阵**：
$$M = \begin{bmatrix} \|S_\text{tgt}\|_2^2 & 0 \\ 0 & \|S_\text{oth}^\text{max}\|_2^2 \end{bmatrix}$$

**凸优化求解**：$x = [\alpha, \beta]^\top$
$$\min_{x \geq 0} \frac{1}{2} x^\top M x + \frac{1}{2} \|\max(0, d - Cx)\|_2^2$$

闭式解：$(M + C^\top C)x = C^\top d$，投影$x \leftarrow \max(x, 0)$。

### 推理流程

- 仅在前20步（共50步）、前20层（共40层）的cross-attention中应用EAQS+ABSS
- 每个attention head独立应用
- Self-attention完全不变
- 后30步和后20层：原始模型自行细化外观

### 关键参数

去噪50步(UniPC), guidance=5.0, 分辨率832×480, latent 81帧, 引导步[1-20], 引导层[1-20]

### 推理开销

2事件17.6min vs基线15.2min (A100), 4事件22.3min。ABSS的线性系统求解是主要额外开销。

---

## 17. SynCoS (arXiv:2503.08605)

**标题**: Tuning-Free Multi-Event Long Video Generation via Synchronized Coupled Sampling

**作者**: Subin Kim, Seoung Wug Oh, Jui-Hsien Wang, Joon-Young Lee, Jinwoo Shin

**机构**: KAIST、Adobe Research

**基础模型**: CogVideoX-2B / Open-Sora Plan v1.3（架构无关，兼容v-prediction/epsilon-prediction/rectified flow）

### 核心问题

局部融合方法（Gen-L-Video）去噪路径发散→远距离chunk语义漂移；全局优化方法（CSD）直接应用长视频生成完全失败（缺乏结构先验→帧坍缩）。需要统一框架同时保证局部平滑和全局一致。

### 算法组件

#### 三阶段同步耦合（每个去噪步内执行）

**阶段1 — DDIM时序共去噪+Fusion**：

各chunk独立去噪→DDIM估计$\hat{\mathbf{x}}_{0|t}$→重叠区域按count归一化取平均融合。

**阶段2 — CSD优化精炼**（仅$t > t_{\min}$时执行）：

采样固定基线噪声$\boldsymbol{\epsilon}'_{\text{base}} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$（单步内不变）。

加噪：$\mathbf{x}_t^{(i)} = \sqrt{\bar{\alpha}_t}\mathbf{x}_0^{(i)} + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_{\text{base}}^{(i)}$

CSD梯度（基于SVGD + RBF核）：
$$\nabla_{\theta_i} \mathcal{L}_{\text{CSD}} = \frac{w(t)}{N} \sum_{j=1}^{N}\left[k(\mathbf{x}_t^{(j)}, \mathbf{x}_t^{(i)})(\boldsymbol{\epsilon}_\Phi(\mathbf{x}_t^{(i)}) - \boldsymbol{\epsilon}^{(i)}) + \nabla_{\mathbf{x}_t^{(j)}} k(\mathbf{x}_t^{(j)}, \mathbf{x}_t^{(i)})\right]\frac{\partial \mathbf{x}^{(i)}}{\partial \theta_i}$$

融合梯度后AdamW更新$\mathbf{x}'_0$，循环iters次。Minibatch随机采样B个chunk降低开销。

**阶段3 — DDIM回退**：
$$\mathbf{x}'_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}'_0 + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}'_{\text{base}}$$

使用相同固定基线噪声，确保方向对齐。

#### 三个关键机制

1. **Grounded Timestep**：阶段2时间步锚定在阶段1的DDIM调度上（非随机选取）
2. **Fixed Baseline Noise**：单步内固定噪声，防止不同prompt引导相互干扰和样本坍缩
3. **Coupled Sampling**：CSD仅在$t > t_{\min}$时执行（早期步骤建立全局语义，后期仅局部细化）

#### Structured Prompt

全局prompt $\mathbf{g}$（共享属性）+ 局部prompts $\{\mathbf{l}_i\}$（各事件动作）。GPT-4o分割。每个chunk条件为$(g, l_i)$组合。

#### Flow-based适配

$$\nabla_\theta \mathcal{L}_{\text{Flow-SDS}} = \mathbb{E}_{\boldsymbol{\epsilon}, t}\left[w(t)(\mathbf{v}_\Phi(\mathbf{x}_t, t) - (\boldsymbol{\epsilon} - \mathbf{x}))\frac{\partial \mathbf{x}}{\partial \theta}\right]$$

### 关键参数

DDIM 50步, η=0, stride=4(CogVideoX)/6(OSP), CFG=6/7.5, $t_{\min}$∈[800,900], lr∈[0.5,1], CSD iters∈[20,50], 优化器AdamW

### 计算开销

单H100生成4倍长视频约55min（vs Gen-L-Video 21min），约2.6×baseline。

---

---

## 18. FreeLong++ (arXiv:2507.00162)

**标题**: FreeLong++: Training-Free Long Video Generation via Multi-band SpectralFusion

**作者**: Yu Lu, Yi Yang

**机构**: 浙江大学ReLER, CCAI

**发表**: IEEE TPAMI

**基础模型**: Wan-2.1/Wan-1.3B (81帧) / LTX-Video (121帧) / VACE

### 核心问题

短视频扩散模型直接生成长视频时，高频分量SNR随长度急剧下降（8倍长度时SNR降至0.6），而低频保持稳定（0.97）。根本原因：注意力图在长序列上失去对角结构，模型无法捕获远距帧间相关。

### 算法组件

#### Multi-Scale Attention Decoupling（多尺度注意力解耦）

第$l$个分支掩码注意力：
$$A_l(i,j) = \begin{cases} \text{Softmax}\left(\frac{Q_i K_j^\top}{\sqrt{d}}\right) & \text{if } |i-j| < \lfloor \frac{\alpha_l T_\alpha}{2} \rfloor \\ 0 & \text{otherwise} \end{cases}$$

多尺度配置（4×生成）：$\alpha_1=1, \alpha_2=2, \alpha_3=4$，对应窗口$T_\alpha, 2T_\alpha, 4T_\alpha$。

**稀疏关键帧**（全局分支）：均匀采样50%帧作为键，降低计算开销。

#### Multi-band Spectral Fusion（多频段频谱融合）

1. 各分支输出做3D FFT：$\hat{Z}_l = \mathcal{F}_{\text{3D}}(Z_l)$
2. 乘以尺度特定带通滤波器：$\hat{Z}' = \sum_{l=1}^{L} \mathcal{P}_l \odot \hat{Z}_l$
3. 逆FFT：$Z' = \mathcal{F}_{\text{3D}}^{-1}(\hat{Z}')$

**频带划分**基于Nyquist准则：窗口$\alpha_l T_\alpha$对应最大可表达频率$\frac{1}{2\alpha_l}\pi$。
- 最粗尺度$\alpha=4$：保留$[0, \frac{1}{8}\pi]$（全局动态）
- 中等尺度$\alpha=2$：保留$[\frac{1}{8}\pi, \frac{1}{4}\pi]$
- 最细尺度$\alpha=1$：覆盖$[\frac{1}{4}\pi, \pi]$（快速局部运动）

#### SpecMix Noise Initialization（频谱混合噪声初始化）

一致性基底噪声$x_{\text{base}}$（滑动窗口混洗）+ 逐帧残差噪声$x_{\text{res}}$（独立高斯）。

归一化时间距离：$d_t = \frac{|t - (T-1)/2|}{(T-1)/2}$

混合角度：$\theta_t = d_t \cdot \frac{\pi}{2}$

频域混合：
$$\tilde{x}_t^{\mathcal{F}} = \cos\theta_t \cdot x_{\text{base},t}^{\mathcal{F}} + \sin\theta_t \cdot x_{\text{res},t}^{\mathcal{F}}$$

中间帧主要依赖一致性基底（$\cos\theta \approx 1$），两端帧引入更多高频随机性。$\cos^2\theta + \sin^2\theta = 1$保持方差不变。

### 推理流程

每个去噪步中的每个DiT Block：
1. Cross-Attention（文本条件）
2. 多分支并行计算多尺度注意力 → 各$Z_l$
3. 3D FFT → 带通滤波 → 求和 → 3D IFFT → 融合特征$Z'$
4. 送入后续模块

### 关键参数

$T_\alpha$=81(Wan)/121(LTX), 高斯LPF截止$D_0$=0.25, 4×生成用3分支($\alpha$=1,2,4), 8×生成用4分支($\alpha$=1,2,4,8), 稀疏采样50%

### 计算效率

Wan-1.3B 4×(324帧)：FreeLong++ 96s, +sparse 74s (vs Global-only 50s, Local-only 22s)

---

## 19. Motion by Queries (arXiv:2412.07750)

**标题**: Motion by Queries: Identity-Motion Trade-offs in Text-to-Video Generation

**作者**: Yuval Atzmon, Rinon Gal, Yoad Tewel, Yoni Kasten, Gal Chechik

**机构**: NVIDIA

**基础模型**: VideoCrafter2 (3D U-Net) / WAN 2.1-1.3B (DiT) / T2V-Turbo-V2 / LTX-Video

### 核心发现

T2V模型中Q特征同时编码运动(motion)和身份(identity)——与T2I模型中Q主要编码布局不同。注入Q传递运动时不可避免泄漏源视频身份。这是架构层面的根本权衡。

### 算法组件

#### 零样本运动迁移（Motion Transfer）

**Q特征提取**：对源视频加噪到多个时间步t=1000,980,...,600，每步执行单步DDPM去噪，记录所有空间self-attention层的Q特征$[Q_S(50),...,Q_S(30)]$。

**Q特征注入**：生成目标视频时，前k步（~40% DDPM steps）替换Q为$Q_S(t)$。

DiT适配（WAN）：从单个低噪声时间步提取Q，注入所有更高噪声步；仅注入第20-30层；注入比例58-60%。

#### 一致性多镜头生成（Video Storyboarding）

三次去噪迭代架构：

**Framewise Subject-Driven Self-Attention (Framewise-SDSA)**：

仅让时间索引匹配的帧互相attend（frame f in shot i → frame f in shot j），避免运动冻结。

扩展keys/values：$K_f^+ = [K_{1,f} \oplus K_{2,f} \oplus \cdots \oplus K_{N,f}]$

带mask注意力：$A_{i,f}^+ = \text{softmax}(Q_i K_f^+ / \sqrt{d_k} + \log M_{i,f}^+)$

subject mask通过估计$\hat{x}_0$后CLIPSeg+Otsu阈值法动态生成。

**Phase 1: Q Preservation**（T→$t_{\text{pres}}$=750）：直接注入vanilla预生成视频的Q特征，建立稳健运动结构。

**Phase 2: Q Flow**（$t_{\text{pres}}$→0）：基于光流的间接Q融合。

对每个位置(f,x,y)找最近两个关键帧的最相似位置：
$$(x_A, y_A) = \arg\max_{x_0, y_0} S_{\cos}(q_{fxy}, q_{f_A x_0 y_0})$$

生成融合Q：$\hat{q}_{fxy} = w \cdot \hat{q}_{f_A x_A y_A} + (1-w) \cdot \hat{q}_{f_B x_B y_B}$

权重：$w = \text{sigmoid}\left(\frac{f_B - f}{f_B - f_A}\right)$

Q值来自一致性模型（保留身份一致性），对应关系来自vanilla视频光流（保留运动）。

**Refinement Feature Injection**：注入对应subject patches的Output(O)特征。扩展至conditional和unconditional去噪步，使用相同DIFT correspondence map。

### 关键参数

50 DDPM步, guidance=12, 分辨率576×320, Q注入~40%步, $t_{\text{pres}}$=750, SDSA窗口[550,950], Refinement窗口[590,950]

### 效率

仅增加×1.2 overhead（70s vs基模58s），对比DMT ×45, MI ×23。

---

## 20. FreeNoise (arXiv:2310.15169)

**标题**: FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling

**作者**: Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, Ziwei Liu

**机构**: 南洋理工大学、腾讯AI Lab、香港科技大学

**基础模型**: VideoCrafter (VideoLDM架构, 16帧训练)

### 核心问题

视频扩散模型在有限帧数(16帧)训练，直接生成长视频触发"注意力范围敏感性"；简单滑动窗口导致"噪声引起的时序漂移"。

### 核心发现

时序注意力是帧序无关的（输出帧与初始噪声帧严格对应），时序卷积是帧序相关的（根据噪声帧顺序引入新内容）。因此逐帧噪声决定整体外观，帧的时序顺序影响构建内容。

### 算法组件

#### Local Noise Shuffle Unit（局部噪声打乱）

初始化$N_{\text{train}}$帧随机噪声，通过打乱复用扩展到$M$帧：

$$[\epsilon_1,...,\epsilon_{N_{\text{train}}}, \text{shuffle}(\epsilon_{\hat{S}_{(i+1)}},...,\epsilon_{\hat{S}_{(i+S)}}),...]$$

$\hat{S}_i = i \mod N_{\text{train}}$，$S$为打乱单元大小（=滑动步长=4）。

重复同一组噪声帧→长程相关性；打乱顺序→时序卷积产生内容变化。

#### Window-based Attention Fusion（窗口注意力融合）

窗口内时序注意力（$U = N_{\text{train}}$帧）：
$$F_{i:i+U}^j = \text{Softmax}\left(\frac{Q_{i:i+U} K_{i:i+U}^T}{\sqrt{d}}\right) V_{i:i+U}$$

基于窗口中心距离的加权融合：
$$F_i^o = \sum_j \frac{F_i^j \cdot (\frac{U}{2} - |i - c^j|)}{\sum_j (\frac{U}{2} - |i - c^j|)}$$

三角形权重分布：越靠近窗口中心权重越大。

#### Motion Injection（多提示视频生成）

在cross-attention中按时间步和层选择性切换文本条件：

$$\text{Motion Injection} := \begin{cases} \text{Attn}_{\text{cross}}(\widetilde{Q}, l_{\widetilde{K}}(\widetilde{P}), l_{\widetilde{V}}(\widetilde{P})), & \text{if } T_\alpha < t < T_\beta \text{ or } l > L \\ \text{Attn}_{\text{cross}}(\widetilde{Q}, l_{\widetilde{K}}(P_1), l_{\widetilde{V}}(P_1)), & \text{otherwise} \end{cases}$$

帧级文本嵌入线性插值实现平滑过渡：
$$\widetilde{P} = P_1 + \frac{n - N_\gamma}{N_\tau - N_\gamma}(P_2 - P_1), \quad N_\gamma \leq n < N_\tau$$

- Encoder层在早/晚时间步用$P_1$→保持场景布局
- Decoder层或物体形状时间步用$\widetilde{P}$→注入新运动

### 推理流程

1. 初始化$N_{\text{train}}$帧噪声，通过局部打乱扩展到$M$帧
2. 迭代去噪：空间模块按帧独立→时序卷积滑动窗口→窗口式时序注意力+加权融合
3. （可选）Motion Injection切换文本条件
4. VAE解码

### 关键参数

$N_{\text{train}}$=16, $M$=64, $U$=16, $S$=4, 额外开销仅17%

### 性能

64帧FVD=85.83（vs Gen-L-Video 177.63, Direct 737.61），推理25.75s (A100)。

---

---

---

## 21. SWIFT (arXiv:2605.09442)

**标题**: SWIFT: Prompt-Adaptive Memory for Efficient Interactive Long Video Generation

**作者**: Shanwen Tan, Hao Li, Jingtao Zhang, Xiaosong Jia, Xue Yang, Shaofeng Zhang, Yanyong Zhang

**机构**: 中国科学技术大学(USTC)、复旦大学、佐治亚理工学院、上海交通大学

**基础模型**: Wan2.1-T2V-1.3B（因果自回归视频扩散模型）

### 核心问题

多prompt长视频生成中，历史视频记忆(KV cache)与新激活prompt间存在语义不匹配。核心矛盾：内存保存视觉连续性 vs. prompt切换要求快速语义适应。现有方法(LongLive)在切换点重建整个cache(ReCache)，计算代价高且切换不平滑。

### 算法组件

#### 4.1 Semantic Injection Cache（语义注入缓存，SIC）

不重建整个cache，而是构造轻量级"语义桥接"注入现有cache的相关位置。

**(a) Prompt Transition Signatures（提示切换签名）**

语义位移：
$$\Delta p^{(m)} = p^{(m)} - p^{(m-1)}$$

切换强度（余弦距离）：
$$\rho^{(m)} = 1 - \cos\left(p^{(m-1)}, p^{(m)}\right)$$

$\rho^{(m)} \in [0,2]$，越大表示语义跳跃越大。

**(b) Motion-Neutral Projection（运动中性投影）**

关键洞察：直接将$\Delta p^{(m)}$注入cache会扰乱当前视频的局部运动趋势。需将语义注入限制在与局部运动**正交**的子空间中。

局部运动切线（有限差分估计）：
$$m = v_t - v_{t-1}$$

**定理1**：最小化偏离原始语义位移、同时满足运动中性约束的最优更新：

$$\Delta p^{(m)}_\perp = \arg\min_{\Delta x \in \mathcal{H}} \|\Delta x - \Delta p^{(m)}\|_2^2 \quad \text{s.t.} \quad \langle \Delta x, m \rangle = 0$$

闭合形式解：
$$\Delta p^{(m)}_\perp = \Delta p^{(m)} - \frac{\langle \Delta p^{(m)}, m \rangle}{\|m\|_2^2} m$$

数值稳定版本：
$$\Delta p^{(m)}_\perp = \Delta p^{(m)} - \frac{\langle \Delta p^{(m)}, m \rangle}{\|m\|_2^2 + \epsilon} m$$

本质是Hilbert空间投影定理：$\Delta p^{(m)}_\perp$是$\Delta p^{(m)}$在运动正交子空间$\mathcal{C} = \{z: \langle z, m \rangle = 0\}$上的正交投影。

**(c) Head-wise Semantic Injection（逐头语义注入）**

不同attention head关注不同时间尺度。从value cache提取：
- $r$：recent summary（最近帧局部视觉摘要）
- $s$：sink summary（持久sink区域全局视觉摘要）

逐头注入门控：
$$g_r = \rho^{(m)} \left[\cos\left(r, \Delta p^{(m)}_\perp\right)\right]_0^1, \quad g_s = \rho^{(m)} \left[\cos\left(s, \Delta p^{(m)}_\perp\right)\right]_0^1$$

$[\cdot]_0^1$截断到$[0,1]$。

桥接分量构造：
$$B_r = (1 - g_r)r + g_r \Delta p^{(m)}_\perp, \quad B_s = (1 - g_s)s + g_s \Delta p^{(m)}_\perp$$

拼接为transient bridge memory：$B_h = [B_s; B_r]$。

指数衰减：
$$B_h^{t+1} = \lambda B_h^t, \quad \lambda = 0.85$$

#### 4.2 Adaptive Dynamic Window（自适应动态窗口，ADW）

Segment age与距下一切换距离：
$$a_t = \max(0, t - s_m), \quad d_t = \begin{cases} \max(0, s_{m+1} - t) & \text{若存在} \\ +\infty & \text{otherwise} \end{cases}$$

Post-switch衰减与Pre-switch扩张：
$$w_{\text{post}}(t) = \exp\left(-\frac{a_t}{\tau_{\text{post}}}\right), \quad w_{\text{pre}}(t) = \exp\left(-\frac{d_t}{\tau_{\text{pre}}}\right)$$

有效阶段权重：
$$w_t = \max\{w_{\text{post}}(t),\ w_{\text{pre}}(t)\}$$

自适应窗口大小：
$$W_t = W_{\min} + (W_{\max} - W_{\min}) w_t$$

切换边界附近$W_t \to W_{\max}$，稳定段$W_t \to W_{\min}$。

#### 4.3 Segment-Level Semantic Anchors（段级语义锚点）

每个prompt段结束时构造压缩的语义锚点：
$$A^{(m)} = (1 - \alpha) u^{(m)} + \alpha p^{(m)}$$

- $u^{(m)}$：已完成段最近$R_{\text{anchor}}=6$帧的value cache视觉摘要
- $p^{(m)}$：该段的prompt签名
- $\alpha = 0.35$

最多保留$M_{\text{anchor}}=4$个历史锚点，注入缩放$\gamma_{\text{anchor}} = 0.8$。

### 统一内存视图

每步attention读取结构化内存：
| 成分 | 作用 | 更新 |
|---|---|---|
| Sink Memory ($N_{\text{sink}}=3$) | 持久全局上下文 | 固定 |
| Semantic Bridge ($B_h$) | 语义切换过渡 | 每块衰减$\lambda=0.85$ |
| Segment Anchors (最多4个) | 压缩长程历史 | 每段结束更新 |
| Adaptive Local Window ($W_t$帧) | 局部时序连续性 | 动态调整 |

### 关键参数

$W_{\max}$=12, $W_{\min}$=7, $\tau_{\text{post}}$=18, $\tau_{\text{pre}}$=9, $\lambda$=0.85, $\alpha$=0.35, $\gamma_{\text{anchor}}$=0.8, $N_{\text{sink}}$=3, $R_{\text{anchor}}$=6, $M_{\text{anchor}}$=4, $B$=3帧/块, $T$=240帧(60秒), 扩散步数$\mathcal{D}$={1000,750,500,250}

### 与其他方法核心区别

1. **vs LongLive**：LongLive在每个prompt边界重建全部KV cache(ReCache)，SWIFT仅构造$O(1)$大小语义桥接注入现有cache，FPS从20.7→22.6
2. **运动中性投影独特性**：唯一从Hilbert空间投影视角证明语义注入应限制在运动正交子空间的方法
3. **逐头差异化注入**：根据每个head对语义切换的对齐程度差异化注入，而非均匀施加

---

## 22. TiARA / ViDE (arXiv:2412.17254)

**标题**: Enhancing Multi-Text Long Video Generation Consistency without Tuning: Time-Frequency Analysis, Prompt Alignment, and Theory

**作者**: Xingyao Li, Fengzhuo Zhang (通讯), Jiachun Pan, Yunlong Hou, Vincent Y. F. Tan, Zhuoran Yang

**机构**: National University of Singapore (NUS)、Yale University

**基础模型**: 通用插件方法，验证于FIFO-Diffusion / FreeNoise / StreamingT2V（均为2D+1D U-Net架构的VideoLDM）

### 核心问题

长视频生成中帧间不一致性：物体数量/颜色/形状突变、背景剧变、运动不自然。关键观察：不一致帧在时序注意力分数矩阵中**对角线值过高**（帧过度关注自身而非邻帧），不一致部分在DSTFT频谱中**高频能量显著偏高**。

### 算法组件

#### 组件一：Temporal Attention Reweighting（时序注意力重加权）

**时序注意力基础**：

$$Q_{h,w} = Z_{:,:,h,w}^\top W_Q, \quad K_{h,w} = Z_{:,:,h,w}^\top W_K, \quad V_{h,w} = Z_{:,:,h,w}^\top W_V$$

$$\text{Att}(Q, K, V) = \text{sm}(d_k^{-1/2} Q K^\top) V$$

**重加权矩阵**：
$$\Lambda = -\alpha \cdot I_{N \times N}, \quad \alpha \geq 0$$

应用于softmax前的相关矩阵：
$$\overline{\text{Att}}(Q, K, V) = \text{sm}(QK^\top + \Lambda)V$$

原理：从对角线减去正值$\alpha$，鼓励每帧从其他帧获取更多信息。附加操作：对注意力矩阵**左下角和右上角**施加重加权（减少远距帧间关注）。

#### 组件二：TiARA（时频自适应重加权）

**问题**：固定$\alpha$在快速运动区域导致模糊（过度平滑高频运动）。

**离散短时傅里叶变换(DSTFT)**：
$$\text{DSTFT}(x, \psi, m, k) = \sum_{n=0}^{N-1} x_n \psi_{n-m} e^{-i \frac{2\pi kn}{N}}, \quad k, m \in [N]$$

$\psi$为窗函数（Blackman窗），支撑大小$L$。

**运动强度定义**：对注意力矩阵第$i$行：
$$\rho_i := \frac{\sum_{\phi_1 \leq k < \phi_2} |\text{DSTFT}(A_{i,:}, \psi, i, k)|^2}{\sum_{k < \phi_2} |\text{DSTFT}(A_{i,:}, \psi, i, k)|^2}$$

- $\phi_1$：高/低频运动分界阈值
- $\phi_2$：高频运动/异常运动分界阈值
- $\rho_i \in [0,1]$：第$i$帧的高频运动占比

**自适应重加权（线性关系）**：
$$\Lambda_{i,i} = -\alpha(1 - \rho_i)$$

- $\rho_i$大（快运动）→ $\Lambda_{i,i} \approx 0$，保留运动细节
- $\rho_i$小（慢运动）→ $\Lambda_{i,i} \approx -\alpha$，增强帧间一致性

**边界处理**：周期填充后再做DSTFT：
$$\tilde{A}_{i,:} \leftarrow \text{Pad}(A_{i,:}, \lfloor L/2 \rfloor, \lfloor L/2 \rfloor)$$

#### 组件三：PromptBlend（多提示词对齐与插值）

**(a) Prompt Organization**：用ChatGPT将每个prompt组织为统一格式：
`[Subject][Action][Time][Place][Video Quality]`

**(b) Token-level Alignment**：对每个组件$k$，找最大token长度$M = \max_i \text{length}(\mathcal{T}(P_{i,k}))$，较短组件通过**重复token**（非空白填充）对齐到$M$，然后拼接并嵌入：
$$\bar{\mathcal{P}}_i = \mathcal{E}(\text{concat}(\{\overline{\mathcal{T}(P_{i,k})}\}_{k=1}^5))$$

**(c) Interpolation Application**：过渡帧$n \in [n_i^e, n_{i+1}^s]$的文本条件：
$$\mathcal{P}_C(n, t, d) = \begin{cases} (1 - a_n)\bar{\mathcal{P}}_i + a_n \bar{\mathcal{P}}_{i+1} & t \in [t_1, t_2] \text{ or } d \geq D \\ \bar{\mathcal{P}}_i & \text{otherwise} \end{cases}$$

$$a_n = \frac{n - n_i^e}{n_{i+1}^s - n_i^e} \in [0,1]$$

设计依据：
- 后期去噪步$t \in [t_1, t_2]$主要影响物体形状
- U-Net decoder部分($d \geq D$)主要影响语义细节同时保持布局
- 仅在这些阶段插值，其余保持前一prompt稳定结构

### 完整推理流程

**单提示词**：在每个去噪步的每个时序注意力层中插入TiARA（计算注意力矩阵→DSTFT分析运动强度→自适应对角线重加权→重新softmax）。

**多提示词**：同时应用TiARA修改时序自注意力 + PromptBlend修改交叉注意力的文本条件。过渡窗口结束后完全切换到下一prompt。

### 关键参数

$\alpha$=5~7（重加权系数）, $L \in \{7,8,9\}$（Blackman窗支撑大小）, 过渡帧数=100帧, 周期填充边界处理, ChatGPT用于prompt组织

### 与其他方法核心区别

1. **频域自适应**：唯一用DSTFT分析运动强度来自适应调节注意力重加权的方法（vs固定重加权/固定掩码）
2. **理论保证**：提供了重加权对注意力一致性提升的频域理论分析
3. **通用插件**：不依赖特定生成架构，可插入任何2D+1D U-Net长视频方法
4. **Token对齐**：PromptBlend通过语义组件对齐实现结构化插值（vs简单嵌入SLERP）

---
