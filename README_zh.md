# ComfyUI-FireRedTTS

这是一个用于 ComfyUI 的自定义节点集成，当前支持 FireRedTTS‑2，并计划兼容更多 FireRedTTS 系列模型。它能够实现高质量、富有情感表达的多说话人对话与独白语音合成。该集成采用流式架构和上下文感知的韵律建模技术，支持自然的说话人切换与稳定的长文本生成，特别适合互动聊天和播客等应用场景。

![ComfyUI-FireRedTTS](https://github.com/user-attachments/assets/81978360-47aa-4f09-861d-edf13ec96187)

## 功能特点

* **对话生成**：多说话人对话语音生成
* **独白生成**：单说话人叙述语音生成
* **声音克隆**：零样本语音克隆功能
* **多语言支持**：中文、英文、日文、韩文、法文、德文、俄文
* **自动模型下载**：首次使用时自动下载模型
* **设备自适应**：自动选择最佳设备（CUDA/MPS/CPU）

## 安装方法

### 方法一：使用 ComfyUI Manager（推荐）

1. 打开 \[ComfyUI Manager]
2. 在 ComfyUI Manager 中搜索 “ComfyUI-FireRedTTS”
3. 点击安装

### 方法二：手动安装

1. 克隆此仓库至你的 ComfyUI 自定义节点目录中：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-FireRedTTS.git
```

2. 安装依赖：

```bash
cd ComfyUI-FireRedTTS
pip install -r requirements.txt
```

3. 重启 ComfyUI

### 模型下载

首次使用时，系统将自动从 Hugging Face 下载 FireRedTTS2 模型：

* 模型来源：[FireRedTeam/FireRedTTS2](https://huggingface.co/FireRedTeam/FireRedTTS2)
* 存储位置：`ComfyUI\models\TTS\FireRedTTS2`
* 下载大小：约 2GB

下载过程中会显示进度条。下载完成后，模型将被缓存以供后续使用。

## 节点说明

### FireRedTTS2 对话节点

生成多说话人的对话语音。

**输入：**

* `text_list`（字符串）：包含说话人标签的对话文本（如：\[S1], \[S2]）
* `temperature`（浮点数）：控制生成的随机性（0.1-2.0，默认值：0.9）
* `topk`（整数）：控制采样范围（1-100，默认值：30）
* `S1`（音频，可选）：说话人1的参考音频
* `S1_text`（字符串，可选）：说话人1的参考文本
* `S2`（音频，可选）：说话人2的参考音频
* `S2_text`（字符串，可选）：说话人2的参考文本

**输出：**

* `audio`（音频）：生成的对话语音
* `sample_rate`（整数）：音频采样率（24000Hz）

### FireRedTTS2 独白节点

生成单说话人的独白语音。

**输入：**

* `text`（字符串）：输入文本内容
* `temperature`（浮点数）：温度参数（0.1-2.0，默认值：0.75）
* `topk`（整数）：TopK 参数（1-100，默认值：20）
* `prompt_wav`（字符串，可选）：参考音频文件路径
* `prompt_text`（字符串，可选）：参考文本内容

**输出：**

* `audio`（音频）：生成的独白语音
* `sample_rate`（整数）：音频采样率（24000Hz）

## 使用方法

### 说话人标签格式

在对话文本中使用方括号标记不同的说话人：

```
[S1]Hello, what a nice day![S2]Yes, perfect for a walk.[S1]Shall we go to the park?[S2]Great idea!
```

**支持的说话人标签：**

* `[S1]` - 说话人1
* `[S2]` - 说话人2

### 声音克隆设置

若使用声音克隆功能，请为每位说话人提供音频与文本：

**说话人1 (S1)：**

* 将参考音频连接到 `S1` 输入
* 在 `S1_text` 字段输入参考文本

**说话人2 (S2)：**

* 将参考音频连接到 `S2` 输入
* 在 `S2_text` 字段输入参考文本

## 示例

### 基础对话生成

1. 添加 “FireRedTTS2 Dialogue” 节点
2. 在 `text_list` 输入：

   ```
   [S1]Welcome to our podcast![S2]Today we'll discuss AI development.[S1]That's a fascinating topic indeed.
   ```
3. 调整 `temperature` 和 `topk` 参数
4. 将音频输出连接至预览或保存节点

### 声音克隆对话

1. 准备每位说话人的参考音频文件
2. 将说话人1的参考音频连接至 `S1`
3. 在 `S1_text` 中输入：

   ```
   This is a voice sample for speaker one
   ```
4. 将说话人2的参考音频连接至 `S2`
5. 在 `S2_text` 中输入：

   ```
   This is a voice sample for speaker two
   ```

### 独白生成

1. 添加 “FireRedTTS2 Monologue” 节点
2. 在 `text` 字段中输入长文本内容
3. 可选提供 `prompt_wav` 与 `prompt_text` 用于声音克隆
4. 调整参数后生成音频

## 参数指南

### Temperature（温度）

* **低（0.1-0.5）**：更稳定、一致的语音
* **中（0.6-1.0）**：稳定性与自然度平衡
* **高（1.1-2.0）**：更有变化和表现力，但可能不稳定

### TopK

* **低（1-20）**：保守采样，更稳定
* **中（21-50）**：平衡选择
* **高（51-100）**：采样更广，变化更多

## 故障排查

### 常见问题

**Q：模型下载失败**
A：检查网络连接与 Hugging Face 的访问。尝试使用代理或镜像站点。

**Q：CUDA 内存不足**
A：

* 减少输入文本长度
* 降低批处理大小
* 在代码中设置 `device="cpu"` 以使用 CPU 模式

**Q：音频质量差**
A：

* 检查输入文本格式是否正确
* 调整温度参数（建议 0.7-1.0）
* 若使用声音克隆，确保参考音频质量良好

**Q：说话人标签无效**
A：

* 确保标签格式正确：\[S1]、\[S2] 等
* 检查标签前后是否有多余空格
* 确认文本中包含对应的说话人标签

**Q：节点加载失败**
A：

* 检查依赖项是否已正确安装
* 确认 ComfyUI 版本兼容
* 查看控制台中的错误信息

### 性能优化

**内存优化：**

* 长文本会自动拆分处理
* 模型实例缓存并复用
* 建议单段文本长度：不超过 500 字符

**速度优化：**

* 首次使用需要下载模型，之后会加快
* GPU 加速大幅提升生成速度
* 批量处理多个短文本效率高于处理单个长文本

### 系统要求

**最低配置：**

* Python 3.8+
* 4GB 内存
* 2GB 存储空间（用于模型）

**推荐配置：**

* Python 3.9+
* 8GB+ 内存
* NVIDIA 显卡（4GB+ 显存）
* SSD 存储

## 技术支持

若遇到问题，请确认以下几点：

1. 所有依赖项已正确安装
2. 模型已正确下载
3. 输入格式符合要求
4. 系统资源足够

如需了解更多技术细节，请参阅项目源码及 FireRedTTS2 官方文档。

---

如需将此内容导出为文档或进一步格式化为 Markdown、PDF 或其他形式，也可以告诉我！
