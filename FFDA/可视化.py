import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# 颜色方案
colors = {
    'input': '#4CAF50',      # 绿色 - 输入
    'feature': '#2196F3',    # 蓝色 - 特征处理
    'fourier': '#FF9800',    # 橙色 - 傅里叶域
    'learnable': '#E91E63',  # 粉色 - 可学习组件
    'output': '#9C27B0',     # 紫色 - 输出
    'loss': '#F44336'        # 红色 - 损失函数
}

# 标题
ax.text(8, 11.5, '可学习自适应带宽FDA神经网络架构', 
        fontsize=20, weight='bold', ha='center')

# 1. 输入层
ax.text(2, 10.8, '输入图像', fontsize=14, weight='bold', ha='center')

# 源图像框
src_rect = FancyBboxPatch((0.5, 9.5), 3, 1.2, boxstyle="round,pad=0.1",
                         facecolor=colors['input'], alpha=0.3)
ax.add_patch(src_rect)
ax.text(2, 10.2, '源图像\n(Source Image)', ha='center', va='center', fontsize=10)

# 目标图像框
trg_rect = FancyBboxPatch((3.5, 9.5), 3, 1.2, boxstyle="round,pad=0.1",
                         facecolor=colors['input'], alpha=0.3)
ax.add_patch(trg_rect)
ax.text(5, 10.2, '目标图像\n(Target Image)', ha='center', va='center', fontsize=10)

# 2. 预处理和拼接
ax.text(8, 10.2, '通道拼接\n(Channel Concatenation)', ha='center', va='center', 
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['feature'], alpha=0.5))

# 箭头: 输入 -> 拼接
ax.arrow(3.5, 9.5, 0.8, -0.5, head_width=0.1, head_length=0.1, 
         fc=colors['feature'], ec=colors['feature'])
ax.arrow(6.5, 9.5, -0.8, -0.5, head_width=0.1, head_length=0.1, 
         fc=colors['feature'], ec=colors['feature'])

# 3. 带宽预测器 (Bandwidth Predictor)
ax.text(8, 8.5, '带宽预测器\n(Bandwidth Predictor)', fontsize=12, weight='bold', 
        ha='center', bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['learnable'], alpha=0.7))

# 预测器内部结构
predictor_layers = [
    ('Conv2d\n(6→16, 3×3)', (7, 7.8), 2, 0.4),
    ('BN+ReLU\n+MaxPool', (7, 7.2), 2, 0.4),
    ('Conv2d\n(16→32, 3×3)', (7, 6.6), 2, 0.4),
    ('BN+ReLU', (7, 6.0), 2, 0.4),
    ('AdaptiveAvgPool\n+Flatten', (7, 5.4), 2, 0.4),
    ('Linear\n(32→16)+ReLU', (7, 4.8), 2, 0.4),
    ('Linear\n(16→1)+Sigmoid', (7, 4.2), 2, 0.4)
]

for i, (label, pos, width, height) in enumerate(predictor_layers):
    rect = FancyBboxPatch((pos[0]-width/2, pos[1]-height/2), width, height,
                         boxstyle="round,pad=0.05", facecolor=colors['learnable'], alpha=0.5)
    ax.add_patch(rect)
    ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=8)

# 箭头: 拼接 -> 预测器
ax.arrow(8, 9.0, 0, -0.3, head_width=0.1, head_length=0.1, 
         fc=colors['learnable'], ec=colors['learnable'])

# 4. 带宽输出和强度控制
ax.text(10.5, 7.0, '自适应带宽\nb_pred', ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="circle,pad=0.8", facecolor=colors['learnable'], alpha=0.7))

# 强度控制
ax.text(10.5, 5.8, '强度控制\n(Strength = 1.0)', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['feature'], alpha=0.5))

# 箭头: 预测器 -> 带宽
ax.arrow(9, 4.2, 1.3, 2.6, head_width=0.1, head_length=0.1, 
         fc=colors['learnable'], ec=colors['learnable'])

# 箭头: 带宽 -> 强度控制
ax.arrow(10.5, 6.7, 0, -0.7, head_width=0.1, head_length=0.1, 
         fc=colors['feature'], ec=colors['feature'])

# 5. 傅里叶变换路径
# 源图像FFT路径
ax.text(2, 7.5, '快速傅里叶变换\n(FFT)', ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['fourier'], alpha=0.5))

# 目标图像FFT路径  
ax.text(5, 7.5, '快速傅里叶变换\n(FFT)', ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['fourier'], alpha=0.5))

# 箭头: 源图像 -> FFT
ax.arrow(2, 9.5, 0, -1.7, head_width=0.1, head_length=0.1, 
         fc=colors['fourier'], ec=colors['fourier'])

# 箭头: 目标图像 -> FFT
ax.arrow(5, 9.5, 0, -1.7, head_width=0.1, head_length=0.1, 
         fc=colors['fourier'], ec=colors['fourier'])

# 6. 频谱分解
# 源图像频谱
ax.text(1.5, 5.8, '幅度谱\n(Amp_src)', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['fourier'], alpha=0.3))
ax.text(1.5, 5.2, '相位谱\n(Pha_src)', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['fourier'], alpha=0.3))

# 目标图像频谱
ax.text(5.5, 5.8, '幅度谱\n(Amp_trg)', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['fourier'], alpha=0.3))

# 箭头: FFT -> 频谱分解
ax.arrow(2, 7.0, -0.3, -0.8, head_width=0.1, head_length=0.1, 
         fc=colors['fourier'], ec=colors['fourier'])
ax.arrow(5, 7.0, 0.3, -0.8, head_width=0.1, head_length=0.1, 
         fc=colors['fourier'], ec=colors['fourier'])

# 7. 频率掩码生成
ax.text(8, 4.2, '软频率掩码生成\n(Soft Frequency Mask)', ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['learnable'], alpha=0.7))

# 掩码公式
ax.text(8, 3.5, 'soft_mask = σ((b_pred - freq_dist) × 10)', ha='center', va='center', 
        fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.2", facecolor='white'))

# 箭头: 强度控制 -> 掩码生成
ax.arrow(10.5, 5.5, -2.3, -0.8, head_width=0.1, head_length=0.1, 
         fc=colors['learnable'], ec=colors['learnable'])

# 8. 频谱混合
ax.text(3.5, 3.8, '频谱混合\n(Amplitude Mixing)', ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['fourier'], alpha=0.7))

# 混合公式
ax.text(3.5, 3.2, 'A_mixed = soft_mask × Amp_trg + (1-soft_mask) × Amp_src', 
        ha='center', va='center', fontsize=9, style='italic')

# 箭头: 源幅度 -> 混合
ax.arrow(1.5, 5.5, 1.7, -1.5, head_width=0.1, head_length=0.1, 
         fc=colors['fourier'], ec=colors['fourier'])

# 箭头: 目标幅度 -> 混合  
ax.arrow(5.5, 5.5, -1.7, -1.5, head_width=0.1, head_length=0.1, 
         fc=colors['fourier'], ec=colors['fourier'])

# 箭头: 掩码生成 -> 混合
ax.arrow(8, 3.8, -4.3, 0, head_width=0.1, head_length=0.1, 
         fc=colors['learnable'], ec=colors['learnable'])

# 9. 逆傅里叶变换
ax.text(3.5, 2.0, '逆傅里叶变换\n(IFFT)', ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['fourier'], alpha=0.5))

# 箭头: 混合 -> IFFT
ax.arrow(3.5, 3.3, 0, -1.0, head_width=0.1, head_length=0.1, 
         fc=colors['fourier'], ec=colors['fourier'])

# 10. 输出
ax.text(3.5, 0.8, '风格化输出图像\n(Stylized Output)', ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['output'], alpha=0.7))

# 箭头: IFFT -> 输出
ax.arrow(3.5, 1.7, 0, -0.7, head_width=0.1, head_length=0.1, 
         fc=colors['output'], ec=colors['output'])

# 11. 损失函数和梯度回传
ax.text(12, 3.0, '任务损失函数\n(Task Loss)', ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['loss'], alpha=0.7))

# 梯度回传箭头
ax.annotate('', xy=(10.5, 4.0), xytext=(12, 2.5),
            arrowprops=dict(arrowstyle='->', color=colors['loss'], lw=2, 
                           connectionstyle="arc3,rad=-0.2"))

ax.text(13.5, 3.0, '梯度反向传播\n(Gradient Backpropagation)', 
        ha='center', va='center', fontsize=10, rotation=90,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['loss'], alpha=0.5))

# 图例
legend_elements = [
    patches.Patch(facecolor=colors['input'], alpha=0.5, label='输入/输出'),
    patches.Patch(facecolor=colors['feature'], alpha=0.5, label='特征处理'),
    patches.Patch(facecolor=colors['fourier'], alpha=0.5, label='傅里叶域操作'),
    patches.Patch(facecolor=colors['learnable'], alpha=0.5, label='可学习组件'),
    patches.Patch(facecolor=colors['loss'], alpha=0.5, label='损失与梯度')
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
          fontsize=10, framealpha=0.9)

# 添加说明文本
ax.text(0.5, 0.2, '关键创新: 带宽预测器通过学习自动确定最优频带交换范围\n'
                  '优势: 端到端可训练，自适应不同图像对，无需手动调参', 
        fontsize=10, style='italic', ha='left', va='bottom')

plt.tight_layout()
plt.savefig('learnable_fda_architecture.png', dpi=300, bbox_inches='tight')
plt.savefig('learnable_fda_architecture.pdf', bbox_inches='tight')
plt.show() 