# 原架构图与实际代码的差异分析

## ❌ 原图存在的问题

### 1. **Encoder Blocks 不完整**
**原图问题**：
- Level 1: 只画了 ReducedAttn + FFW
- Level 3: 只画了 ChannelAttn + GFFW

**实际代码**：
```python
Enc_blocks: [2, 6, 10]  # 每层有多个blocks
```
- Level 1: 2个blocks，每个block: LayerNorm → ReducedAttn → LayerNorm → FFW
- Level 2: 6个blocks
- Level 3: 10个blocks，使用 ChannelAttn + GFFW

**应该画成**：显示多个连续的blocks，不是只画一个

---

### 2. **Middle Block 严重错误** ⚠️
**原图问题**：
- 只画了 "Fusion" 模块
- 没有体现FHR (Frame History Router)
- 没有显示有11个blocks

**实际代码**：
```python
Middle_blocks: 11
latent_attn_type1: "FHR"
latent_attn_type2: "Channel"
latent_attn_type3: "FHR"
```

**Middle Block实际包含**：
1. **11个Transformer blocks**
2. 使用 **FHR + Channel Attention** 组合
3. FHR会缓存 k/v 用于时序信息聚合

**应该画成**：
```
[LayerNorm → FHR/Channel → LayerNorm → GFFW] × 11 blocks
```

---

### 3. **Decoder Block 不准确**
**原图问题**：
- 画的是: Trans → FGMA → Fusion
- 没有明确显示FGMA如何工作

**实际代码流程**：
```python
decoder_attn_type2: "FGMA"  # 最后一个block用FGMA
```

**Decoder实际流程**：
1. 前面的blocks: ChannelAttn
2. **最后一个block**: FGMA (Flow-Guided Masked Attention)
   - FGMA输入: 当前帧特征 + 邻帧特征
   - FGMA输出: Concat[Warped_feat, Mask, Attention_feat]
3. 然后可能有FHR融合多帧信息

**应该画成**：
```
Decoder Level N:
├─ Block 1-9: [LayerNorm → ChannelAttn → LayerNorm → GFFW]
└─ Block 10 (last): [FGMA Alignment] → [FHR Fusion]
```

---

### 4. **缺少双编码器** ⚠️
**原图问题**：
- 完全没有画双编码器

**实际代码**：
```python
use_dual_encoder: True
self.long_exposure_projection = nn.Conv2d(...)
self.short_exposure_projection = nn.Conv2d(...)
```

**应该在最开始画**：
```
Input (5 frames: T-2,T-1,T,T+1,T+2)
         ↓
    ┌─────────────┐
    │ Dual Encoder│
    ├─────────────┤
    │ Long Exp    │ ← For frames with blur
    │ Short Exp   │ ← For frames with noise
    └─────────────┘
         ↓
    Encoder Level 1...
```

---

### 5. **FGMA模块画得太简单**
**原图问题**：
- FGMA只是一个蓝色方块
- 没有展示内部工作原理

**FGMA实际包含（论文核心创新）**：
```python
1. SPyNet → 计算双向光流 (forward & backward)
2. Forward-Backward Consistency Check → 生成 Mask
   - D(i,j) = |L_{t→t-1→t} - L_t|
   - M(i,j) = 1 if s·D(i,j)/255 > 0.5 else 0
3. Reliable regions (M=0) → 使用光流warp
4. Unreliable regions (M=1) → 使用sparse attention
5. Output: Concat[F_warped, M, F_attention]
```

**应该画成详细的流程图**：
```
Current Frame ──┬──→ SPyNet ──→ F-B Check ──→ Mask M
                │                                 ↓
Ref Frame ──────┴────────────────┬──→ Flow Warp (M=0)
                                 │              ↓
                                 └──→ Attention (M=1)
                                                 ↓
                                            Concat
```

---

### 6. **Skip Connections 没画清楚**
**原图问题**：
- 没有明确标注skip connections

**实际代码**：
```python
# Decoder Level 3
inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
```

**应该画成**：
- Encoder Level 1 ─────→ Decoder Level 1 (skip)
- Encoder Level 2 ─────→ Decoder Level 2 (skip)
- Encoder Level 3 ─────→ Decoder Level 3 (skip)

---

## ✅ 正确的架构应该包含

### 整体流程：
```
Input (5 frames)
    ↓
[Dual Encoder] (Long/Short)
    ↓
[Encoder L1: 2 blocks] ──skip──┐
    ↓ (downsample)              │
[Encoder L2: 6 blocks] ──skip──┤
    ↓ (downsample)              │
[Encoder L3: 10 blocks] ─skip──┤
    ↓ (downsample)              │
[Middle: 11 blocks with FHR]    │
    ↓ (upsample)                │
[Decoder L3: 10 blocks + FGMA] ←┘
    ↓ (upsample)                │
[Decoder L2: 6 blocks + FGMA] ←─┘
    ↓ (upsample)                │
[Decoder L1: 2 blocks + FGMA] ←─┘
    ↓
[Refinement: 2 blocks]
    ↓
Output (HDR)
```

### 关键配置：
```yaml
Enc_blocks: [2, 6, 10]
Middle_blocks: 11
Dec_blocks: [10, 6, 2]
num_refinement_blocks: 2

# Encoder
encoder1: ReducedAttn + FFW
encoder2: ReducedAttn + FFW  
encoder3: ChannelAttn + GFFW

# Middle
latent: FHR + Channel + GFFW

# Decoder (每层最后一个block用FGMA)
decoder1/2/3: Channel + FGMA + GFFW

# Refinement
refinement: ReducedAttn + GFFW
```

---

## 📋 建议

1. **重新绘制整体架构图**，包含：
   - 双编码器
   - 完整的block数量
   - Skip connections
   - Middle block的FHR

2. **详细绘制FGMA模块图**，展示：
   - SPyNet光流估计
   - Forward-backward consistency check
   - Binary mask生成 (Eq. 5)
   - Sparse attention只在mask区域计算
   - 最终Concat输出

3. **添加训练策略图**，说明：
   - 30% optical flow (s=0)
   - 30% attention (s=∞)
   - 40% FGMA (s随机采样)

4. **标注关键参数**：
   - Sensitivity parameter s
   - Frame caching (T-2,T-1,T+1,T+2)
   - 各层的通道数变化

---

## 🎨 如何运行生成脚本

我已经创建了正确的绘图脚本：`generate_correct_diagram.py`

运行方式：
```bash
cd h:\zzlzsh\Turtlenew
python generate_correct_diagram.py
```

会生成包含3个子图的完整架构图：
1. 整体架构 (左上+右上)
2. FGMA详细流程 (左下)
3. 训练策略 (右下)
