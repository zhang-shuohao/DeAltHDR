"""
DeAltHDR 正确架构图生成脚本
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 创建大图
fig = plt.figure(figsize=(20, 14))

# ==================== 整体架构图 ====================
ax_main = plt.subplot(2, 2, (1, 2))
ax_main.set_xlim(0, 20)
ax_main.set_ylim(0, 12)
ax_main.axis('off')
ax_main.set_title('DeAltHDR Overall Architecture', fontsize=16, fontweight='bold', pad=20)

# 颜色定义
color_encoder = '#E3F2FD'
color_decoder = '#FFF3E0'
color_middle = '#F3E5F5'
color_dual = '#E8F5E9'
color_fgma = '#FFE0B2'
color_fhr = '#FFCDD2'

# 绘制双编码器
y_start = 10
rect1 = FancyBboxPatch((0.5, y_start), 1.8, 1.2, boxstyle="round,pad=0.1", 
                        facecolor=color_dual, edgecolor='#4CAF50', linewidth=2)
ax_main.add_patch(rect1)
ax_main.text(1.4, y_start+0.9, 'Long Exp\nEncoder', ha='center', va='center', fontsize=9, fontweight='bold')

rect2 = FancyBboxPatch((0.5, y_start-1.5), 1.8, 1.2, boxstyle="round,pad=0.1", 
                        facecolor=color_dual, edgecolor='#4CAF50', linewidth=2)
ax_main.add_patch(rect2)
ax_main.text(1.4, y_start-0.9, 'Short Exp\nEncoder', ha='center', va='center', fontsize=9, fontweight='bold')

# Input
ax_main.text(0.2, y_start+0.6, 'T-2,T-1,T,\nT+1,T+2', ha='right', va='center', fontsize=8)
ax_main.arrow(0.2, y_start+0.6, 0.2, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
ax_main.arrow(0.2, y_start-0.9, 0.2, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Encoder Level 1
x_pos = 3.5
rect_enc1 = FancyBboxPatch((x_pos, y_start-0.5), 1.5, 1.8, boxstyle="round,pad=0.1",
                           facecolor=color_encoder, edgecolor='#2196F3', linewidth=2)
ax_main.add_patch(rect_enc1)
ax_main.text(x_pos+0.75, y_start+0.9, 'Encoder L1', ha='center', va='top', fontsize=10, fontweight='bold')
ax_main.text(x_pos+0.75, y_start+0.4, '2 Blocks', ha='center', va='center', fontsize=8)
ax_main.text(x_pos+0.75, y_start+0, 'ReducedAttn', ha='center', va='center', fontsize=8, style='italic')
ax_main.text(x_pos+0.75, y_start-0.3, 'FFW', ha='center', va='center', fontsize=8, style='italic')

# Downsample
ax_main.text(x_pos+1.5+0.3, y_start+0.4, '↓', ha='center', va='center', fontsize=20, color='red')

# Encoder Level 2
x_pos += 2.2
rect_enc2 = FancyBboxPatch((x_pos, y_start-0.5), 1.5, 1.8, boxstyle="round,pad=0.1",
                           facecolor=color_encoder, edgecolor='#2196F3', linewidth=2)
ax_main.add_patch(rect_enc2)
ax_main.text(x_pos+0.75, y_start+0.9, 'Encoder L2', ha='center', va='top', fontsize=10, fontweight='bold')
ax_main.text(x_pos+0.75, y_start+0.4, '6 Blocks', ha='center', va='center', fontsize=8)
ax_main.text(x_pos+0.75, y_start+0, 'ReducedAttn', ha='center', va='center', fontsize=8, style='italic')
ax_main.text(x_pos+0.75, y_start-0.3, 'FFW', ha='center', va='center', fontsize=8, style='italic')

ax_main.text(x_pos+1.5+0.3, y_start+0.4, '↓', ha='center', va='center', fontsize=20, color='red')

# Encoder Level 3
x_pos += 2.2
rect_enc3 = FancyBboxPatch((x_pos, y_start-0.5), 1.5, 1.8, boxstyle="round,pad=0.1",
                           facecolor=color_encoder, edgecolor='#2196F3', linewidth=2)
ax_main.add_patch(rect_enc3)
ax_main.text(x_pos+0.75, y_start+0.9, 'Encoder L3', ha='center', va='top', fontsize=10, fontweight='bold')
ax_main.text(x_pos+0.75, y_start+0.4, '10 Blocks', ha='center', va='center', fontsize=8)
ax_main.text(x_pos+0.75, y_start+0, 'ChannelAttn', ha='center', va='center', fontsize=8, style='italic')
ax_main.text(x_pos+0.75, y_start-0.3, 'GFFW', ha='center', va='center', fontsize=8, style='italic')

ax_main.text(x_pos+1.5+0.3, y_start+0.4, '↓', ha='center', va='center', fontsize=20, color='red')

# Middle Block (Latent)
x_pos += 2.2
rect_middle = FancyBboxPatch((x_pos, y_start-0.5), 1.8, 1.8, boxstyle="round,pad=0.1",
                             facecolor=color_middle, edgecolor='#9C27B0', linewidth=2.5)
ax_main.add_patch(rect_middle)
ax_main.text(x_pos+0.9, y_start+0.9, 'Middle', ha='center', va='top', fontsize=10, fontweight='bold')
ax_main.text(x_pos+0.9, y_start+0.4, '11 Blocks', ha='center', va='center', fontsize=8)
ax_main.text(x_pos+0.9, y_start+0, 'FHR + Channel', ha='center', va='center', fontsize=8, style='italic', color='#9C27B0')
ax_main.text(x_pos+0.9, y_start-0.3, 'GFFW', ha='center', va='center', fontsize=8, style='italic')

ax_main.text(x_pos+1.8+0.3, y_start+0.4, '↑', ha='center', va='center', fontsize=20, color='blue')

# Decoder Level 3
x_pos += 2.5
rect_dec3 = FancyBboxPatch((x_pos, y_start-0.5), 1.8, 1.8, boxstyle="round,pad=0.1",
                           facecolor=color_decoder, edgecolor='#FF9800', linewidth=2)
ax_main.add_patch(rect_dec3)
ax_main.text(x_pos+0.9, y_start+0.9, 'Decoder L3', ha='center', va='top', fontsize=10, fontweight='bold')
ax_main.text(x_pos+0.9, y_start+0.4, '10 Blocks', ha='center', va='center', fontsize=8)
ax_main.text(x_pos+0.9, y_start+0, 'FGMA + FHR', ha='center', va='center', fontsize=8, style='italic', color='#FF5722')
ax_main.text(x_pos+0.9, y_start-0.3, 'GFFW', ha='center', va='center', fontsize=8, style='italic')

# Skip connections
ax_main.arrow(8.75, y_start+1.5, 6.5, 0, head_width=0.15, head_length=0.2, 
              fc='gray', ec='gray', linestyle='--', linewidth=1.5, alpha=0.6)

ax_main.text(x_pos+1.8+0.3, y_start+0.4, '↑', ha='center', va='center', fontsize=20, color='blue')

# Decoder Level 2
x_pos += 2.5
rect_dec2 = FancyBboxPatch((x_pos, y_start-0.5), 1.8, 1.8, boxstyle="round,pad=0.1",
                           facecolor=color_decoder, edgecolor='#FF9800', linewidth=2)
ax_main.add_patch(rect_dec2)
ax_main.text(x_pos+0.9, y_start+0.9, 'Decoder L2', ha='center', va='top', fontsize=10, fontweight='bold')
ax_main.text(x_pos+0.9, y_start+0.4, '6 Blocks', ha='center', va='center', fontsize=8)
ax_main.text(x_pos+0.9, y_start+0, 'FGMA + FHR', ha='center', va='center', fontsize=8, style='italic', color='#FF5722')
ax_main.text(x_pos+0.9, y_start-0.3, 'GFFW', ha='center', va='center', fontsize=8, style='italic')

ax_main.arrow(6.45, y_start+1.7, 11.5, 0, head_width=0.15, head_length=0.2, 
              fc='gray', ec='gray', linestyle='--', linewidth=1.5, alpha=0.6)

ax_main.text(x_pos+1.8+0.3, y_start+0.4, '↑', ha='center', va='center', fontsize=20, color='blue')

# Decoder Level 1 + Refinement
x_pos += 2.5
rect_dec1 = FancyBboxPatch((x_pos, y_start-0.3), 1.8, 1.6, boxstyle="round,pad=0.1",
                           facecolor=color_decoder, edgecolor='#FF9800', linewidth=2)
ax_main.add_patch(rect_dec1)
ax_main.text(x_pos+0.9, y_start+0.9, 'Decoder L1', ha='center', va='top', fontsize=10, fontweight='bold')
ax_main.text(x_pos+0.9, y_start+0.4, '2 Blocks', ha='center', va='center', fontsize=8)
ax_main.text(x_pos+0.9, y_start+0, 'FGMA + FHR', ha='center', va='center', fontsize=8, style='italic', color='#FF5722')

ax_main.arrow(4.15, y_start+1.9, 15.5, 0, head_width=0.15, head_length=0.2, 
              fc='gray', ec='gray', linestyle='--', linewidth=1.5, alpha=0.6)

# Output
ax_main.arrow(x_pos+1.8, y_start+0.4, 0.3, 0, head_width=0.15, head_length=0.15, fc='green', ec='green', linewidth=2)
ax_main.text(x_pos+2.3, y_start+0.4, 'HDR\nOutput', ha='left', va='center', fontsize=9, fontweight='bold', color='green')

# 添加图例
legend_y = 2
ax_main.text(1, legend_y, 'Key Components:', fontsize=10, fontweight='bold')
ax_main.text(1, legend_y-0.5, '• FGMA: Flow-Guided Masked Attention', fontsize=8)
ax_main.text(1, legend_y-0.9, '• FHR: Frame History Router (caching)', fontsize=8)
ax_main.text(1, legend_y-1.3, '• Skip Connections: ---->', fontsize=8, color='gray')

# ==================== FGMA 详细图 ====================
ax_fgma = plt.subplot(2, 2, 3)
ax_fgma.set_xlim(0, 10)
ax_fgma.set_ylim(0, 8)
ax_fgma.axis('off')
ax_fgma.set_title('FGMA (Flow-Guided Masked Attention) Module', fontsize=12, fontweight='bold')

y = 6.5
# Current & Ref frames
rect1 = FancyBboxPatch((0.5, y), 1.2, 0.8, boxstyle="round,pad=0.05", 
                        facecolor='#E3F2FD', edgecolor='blue', linewidth=1.5)
ax_fgma.add_patch(rect1)
ax_fgma.text(1.1, y+0.4, 'Current\nFrame', ha='center', va='center', fontsize=8)

rect2 = FancyBboxPatch((0.5, y-1.5), 1.2, 0.8, boxstyle="round,pad=0.05", 
                        facecolor='#FFF3E0', edgecolor='orange', linewidth=1.5)
ax_fgma.add_patch(rect2)
ax_fgma.text(1.1, y-1.1, 'Ref Frame\n(T±1,T±2)', ha='center', va='center', fontsize=8)

# SPyNet
ax_fgma.arrow(1.7, y+0.4, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
ax_fgma.arrow(1.7, y-1.1, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

rect_spy = FancyBboxPatch((2.4, y-1), 1, 1.5, boxstyle="round,pad=0.05", 
                          facecolor='#FFCDD2', edgecolor='red', linewidth=1.5)
ax_fgma.add_patch(rect_spy)
ax_fgma.text(2.9, y-0.25, 'SPyNet', ha='center', va='center', fontsize=9, fontweight='bold')
ax_fgma.text(2.9, y-0.6, 'Optical\nFlow', ha='center', va='center', fontsize=7)

# Forward-backward check
ax_fgma.arrow(3.4, y-0.25, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

rect_fb = FancyBboxPatch((4.1, y-0.8), 1.3, 1.1, boxstyle="round,pad=0.05", 
                         facecolor='#FFF9C4', edgecolor='#FBC02D', linewidth=1.5)
ax_fgma.add_patch(rect_fb)
ax_fgma.text(4.75, y-0.25, 'F-B Check', ha='center', va='center', fontsize=9, fontweight='bold')
ax_fgma.text(4.75, y-0.55, '(Eq. 3-5)', ha='center', va='center', fontsize=7)

# Binary Mask
ax_fgma.arrow(5.4, y-0.25, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')

rect_mask = FancyBboxPatch((6.1, y-0.6), 0.9, 0.7, boxstyle="round,pad=0.05", 
                           facecolor='white', edgecolor='black', linewidth=1.5)
ax_fgma.add_patch(rect_mask)
ax_fgma.text(6.55, y-0.25, 'Mask M', ha='center', va='center', fontsize=8, fontweight='bold')

# Flow Warp (reliable regions)
rect_warp = FancyBboxPatch((2.4, y+1.2), 2, 0.9, boxstyle="round,pad=0.05", 
                           facecolor='#C8E6C9', edgecolor='green', linewidth=1.5)
ax_fgma.add_patch(rect_warp)
ax_fgma.text(3.4, y+1.65, 'Flow Warp', ha='center', va='center', fontsize=9, fontweight='bold')
ax_fgma.text(3.4, y+1.4, '(Reliable)', ha='center', va='center', fontsize=7, style='italic')

# Sparse Attention (unreliable regions)
rect_attn = FancyBboxPatch((2.4, y-2.5), 2, 0.9, boxstyle="round,pad=0.05", 
                           facecolor='#FFCCBC', edgecolor='#FF5722', linewidth=1.5)
ax_fgma.add_patch(rect_attn)
ax_fgma.text(3.4, y-2.05, 'Sparse Attn', ha='center', va='center', fontsize=9, fontweight='bold')
ax_fgma.text(3.4, y-2.3, '(Unreliable)', ha='center', va='center', fontsize=7, style='italic')

# Concat
ax_fgma.arrow(4.4, y+1.65, 2.3, -0.45, head_width=0.15, head_length=0.1, fc='black', ec='black')
ax_fgma.arrow(6.55, y+0.1, 0, 0.6, head_width=0.15, head_length=0.1, fc='black', ec='black')
ax_fgma.arrow(4.4, y-2.05, 2.3, 0.45, head_width=0.15, head_length=0.1, fc='black', ec='black')

rect_concat = FancyBboxPatch((7.1, y-0.6), 1.2, 1.8, boxstyle="round,pad=0.05", 
                             facecolor='#E1BEE7', edgecolor='purple', linewidth=2)
ax_fgma.add_patch(rect_concat)
ax_fgma.text(7.7, y+0.9, 'Concat', ha='center', va='center', fontsize=9, fontweight='bold')
ax_fgma.text(7.7, y+0.5, '[Warped,', ha='center', va='center', fontsize=7)
ax_fgma.text(7.7, y+0.2, 'Mask,', ha='center', va='center', fontsize=7)
ax_fgma.text(7.7, y-0.1, 'Attn]', ha='center', va='center', fontsize=7)

ax_fgma.arrow(8.3, y+0.3, 0.5, 0, head_width=0.15, head_length=0.15, fc='green', ec='green', linewidth=2)
ax_fgma.text(9.2, y+0.3, 'Aligned\nFeatures', ha='left', va='center', fontsize=8, fontweight='bold', color='green')

# Sensitivity parameter
ax_fgma.text(4.75, y-1.3, 's (sensitivity)', ha='center', va='center', fontsize=7, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
ax_fgma.arrow(4.75, y-1.1, 0, 0.3, head_width=0.1, head_length=0.1, fc='red', ec='red', linestyle='--')

# ==================== 训练策略图 ====================
ax_train = plt.subplot(2, 2, 4)
ax_train.set_xlim(0, 10)
ax_train.set_ylim(0, 8)
ax_train.axis('off')
ax_train.set_title('Mixed Training Strategy', fontsize=12, fontweight='bold')

# 饼图显示训练比例
y_pie = 5
sizes = [30, 30, 40]
colors_pie = ['#81C784', '#FFB74D', '#64B5F6']
labels_pie = ['30% Optical Flow\n(s=0)', '30% Attention\n(s=∞)', '40% FGMA\n(s=0.1~100)']

wedges, texts, autotexts = ax_train.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.0f%%',
                                         startangle=90, textprops={'fontsize': 9},
                                         center=(3, y_pie), radius=1.8)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 说明文字
y_text = 2
ax_train.text(1, y_text+0.5, 'Training Details:', fontsize=10, fontweight='bold')
ax_train.text(1, y_text, '• Each batch randomly assigns modes', fontsize=8)
ax_train.text(1, y_text-0.5, '• Sensitivity s controls mask ratio', fontsize=8)
ax_train.text(1, y_text-1, '  - s=0: Pure flow (fastest)', fontsize=7, style='italic')
ax_train.text(1, y_text-1.4, '  - s=15: Balanced (default)', fontsize=7, style='italic')
ax_train.text(1, y_text-1.8, '  - s=∞: Pure attention (best)', fontsize=7, style='italic')
ax_train.text(1, y_text-2.3, '• Loss: L1 + VGG (tone-mapped)', fontsize=8)

plt.tight_layout()
plt.savefig('h:/zzlzsh/Turtlenew/DeAltHDR_Correct_Architecture.png', dpi=300, bbox_inches='tight')
print("✅ 正确的架构图已保存: DeAltHDR_Correct_Architecture.png")
plt.show()
