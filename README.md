# 纯本地中英混合 M4A 转文字

将 `.m4a` 录音转成 UTF-8 `.txt`，保留中文和英文，并区分讲话人。识别和说话人分析全部在本机完成，不调用转写 API，也不上传录音。

## 常用 Terminal Command

模型已经下载后，每次转录只需要：

```bash
cd /path/to/m4a_transcriber
source .venv/bin/activate
python transcribe_m4a.py "/完整路径/录音.m4a" --speakers 2 --offline
```

不知道讲话人数时，删除 `--speakers 2`：

```bash
python transcribe_m4a.py "/完整路径/录音.m4a" --offline
```

启动本地网页：

```bash
source .venv/bin/activate
python web_app.py
```

然后打开 [http://127.0.0.1:7860](http://127.0.0.1:7860)。

## 一、安装

当前电脑是 Apple Silicon Mac，使用 WhisperX 官方建议的 CPU `int8` 模式。

```bash
cd /path/to/m4a_transcriber
brew install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 二、首次下载说话人模型

只需做一次：

1. 登录 [Hugging Face](https://huggingface.co/) 并打开 [community-1 模型页面](https://huggingface.co/pyannote/speaker-diarization-community-1)，接受模型使用条件。
2. 在 [Access Tokens](https://huggingface.co/settings/tokens) 创建一个只读 Token。
3. 在终端临时设置 Token：

```bash
export HF_TOKEN="你的只读Token"
```

这个 Token 只用于下载本地模型，不是转写 API。音频不会上传。模型下载完成后可删除 Token，并使用 `--offline`。

程序默认关闭 pyannote 的匿名使用统计。

模型和 Matplotlib 缓存保存在本机；任何录音内容都不会发送给第三方。

## 三、运行

```bash
source .venv/bin/activate
python transcribe_m4a.py "/完整路径/录音.m4a"
```

首次运行会下载数 GB 模型，需要等待。默认在录音旁生成同名 `.txt`。

### 使用本地网页

```bash
source .venv/bin/activate
python web_app.py
```

然后在浏览器打开 [http://127.0.0.1:7860](http://127.0.0.1:7860)，选择 M4A 文件并点击“开始转写”。网页只监听本机地址，不会把音频上传到网络。首次下载模型时需要先设置 `HF_TOKEN`；模型已经缓存后不需要 Token，网页会自动使用离线模式。

已知有两位讲话人时，建议明确指定，可提高分人稳定性：

```bash
python transcribe_m4a.py "/完整路径/录音.m4a" --speakers 2
```

模型已下载后完全离线运行：

```bash
python transcribe_m4a.py "/完整路径/录音.m4a" --speakers 2 --offline
```

提高准确率（会明显变慢、占用更多内存）：

```bash
python transcribe_m4a.py "/完整路径/录音.m4a" --model medium
```

其他选项：

```bash
# 指定输出位置
python transcribe_m4a.py "录音.m4a" -o "结果.txt"

# 只知道人数范围
python transcribe_m4a.py "录音.m4a" --min-speakers 2 --max-speakers 4

# 不显示时间戳
python transcribe_m4a.py "录音.m4a" --no-timestamps
```

输出示例：

```text
[00:00.000 - 00:05.420] 讲话人1: 大家好，today we will discuss the new project.

[00:05.420 - 00:09.810] 讲话人2: 好的，我先介绍一下 background information。
```

## 说明

- `small` 是这台 Mac 上速度和准确率较平衡的默认模型；可改用 `medium` 或 `large-v3`。
- 讲话重叠、远距离录音、噪声或音量差异会降低分人准确率。
- 标签表示不同声音，不会自动知道真实姓名。
- 中英混合录音不应强制指定单一语言，本程序默认自动识别。
- 程序先用 WhisperX/FFmpeg 将音频载入内存，再交给 pyannote，不依赖 TorchCodec 的内置文件解码器。
