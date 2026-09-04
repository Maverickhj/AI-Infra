---
type: source
status: seed
created: 2026-09-04
updated: 2026-09-04
domains:
  - audio
  - audio-llm
  - speech-processing
  - log-mel
aliases:
  - Waveform 到 Log-Mel 的历史
  - Mel filterbank 历史
source:
  - https://doi.org/10.1121/1.1915893
  - https://dicklyon.com/hmh/Lyon_Hearing_book_01jan2018_smaller.pdf
  - https://doi.org/10.1017/9781139051699.008
  - https://doi.org/10.1109/TASSP.1980.1163420
ai_generated: true
reviewed: false
---

# 音频前端：Waveform 到 Log-Mel 的历史脉络

> [!todo]
> 待进一步核对原始文献全文，并整理为 Audio Encoder 的基础知识或音频前端演化笔记。

## 原始问题

把 Waveform 变为 Log-Mel Spectrogram 的过程，最早是哪篇论文提出的？

## 当前结论

严格来说，现代意义上的 `Waveform → Log-Mel Spectrogram` 并不是一篇论文一次性提出的，而是由以下几层工作逐步形成：

1. Mel 频率尺度：用心理声学实验描述人类对音高的感知。
2. 短时频谱：将 waveform 分帧并转换为局部频率能量。
3. Mel filterbank：将线性频率谱映射到非线性的 Mel 频率尺度。
4. Log compression：对 Mel filterbank 能量做对数压缩。
5. DCT：如果继续做离散余弦变换，则得到 MFCC，而不再是原始的 Log-Mel 特征。

如果追溯“完整思路”的最早来源，通常应追溯到 **Bridle & Brown, 1974** 的 JSRU 技术报告；如果限定为正式出版、并且希望引用一篇更容易检索的学术文献，则可引用 **Mermelstein, 1976**；而今天最经典、最广泛传播的 MFCC 版本来自 **Davis & Mermelstein, 1980**。

## 四组关键文献

### 1. Stevens、Volkmann、Newman，1937：Mel 尺度

**S. S. Stevens, J. Volkmann, and E. B. Newman, “A Scale for the Measurement of the Psychological Magnitude Pitch,” 1937.**

这篇工作通过心理声学实验建立了主观音高尺度，并引入以 Mel 为单位的表示：

- 1000 Hz 音调被定义为 1000 mels；
- Mel 不是物理频率，而是感知上的音高尺度；
- 后来的 Mel filterbank 使用这一思想，对频率轴进行非线性重采样或加权。

这篇论文提出的是 **Mel scale**，并不是完整的 Waveform-to-Log-Mel 工程流水线。

### 2. Bridle & Brown，1974：最早的完整思路来源

**J. S. Bridle and M. D. Brown, “An Experimental Automatic Word-Recognition System,” JSRU Report No. 1003, 1974.**

该工作使用实验性的 19-channel vocoder 对语音进行分析，得到：

- 逐帧的 short-term power spectrum；
- logarithmic amplitude / power；
- nonlinear frequency scale；
- 再通过 cosine transform 得到 spectrum-shape coefficients。

这已经非常接近今天的：

```text
Waveform
  → short-time analysis
  → power spectrum
  → nonlinear Mel-like filterbank
  → logarithmic compression
  → optional cosine transform
```

但需要注意：

- 它当时并没有使用现代术语 “Log-Mel Spectrogram”；
- 输出维度只有 19 个频带；
- 它本质上是低分辨率的 logarithmic short-time power spectrum；
- 它是 JSRU 技术报告，而不是今天通常所说的期刊论文。

Paul Mermelstein 后来将 Mel-based cepstral analysis 的思路归因于 Bridle 和 Brown。

### 3. Mermelstein，1976：Mel-based cepstral parameters

**P. Mermelstein, “Distance Measures for Speech Recognition, Psychological and Instrumental,” 1976.**

Mermelstein 将 Mel 频率尺度、对数频谱和余弦变换组合成了后来所称的 **mel-based cepstral parameters**。

从现代视角看：

```text
Log-Mel Spectrogram
  → DCT
  → MFCC
```

因此，这篇工作对于理解以下概念之间的关系非常关键：

- Mel spectrum；
- log-Mel filterbank energies；
- cepstrum；
- MFCC；
- 用较低维参数表示语音频谱包络。

### 4. Davis & Mermelstein，1980：经典 MFCC 版本

**S. B. Davis and P. Mermelstein, “Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences,” IEEE TASSP, 1980.**

这篇论文系统比较了多种语音参数化方法，并推广了今天最常见的 MFCC 流程：

```text
Waveform
  → framing + windowing
  → FFT / power spectrum
  → triangular Mel filterbank
  → log filterbank energies
  → DCT
  → MFCC
```

它的历史地位更准确地说是：

- 将 Mel filterbank + log energy + DCT 形成了经典工程方案；
- 系统评估了该方案对语音识别的效果；
- 使 MFCC 成为后来几十年语音识别系统的标准前端之一。

因此，不能简单地说 “Davis & Mermelstein 1980 首次提出了 Log-Mel”；更准确的说法是：它把此前已经出现的思路系统化并广泛推广成了经典 MFCC 特征。

## Log-Mel 与 MFCC 的区别

```text
Mel filterbank energies
  → log compression
  = Log-Mel Spectrogram / log-Mel filterbanks

Log-Mel filterbanks
  → DCT
  = MFCC
```

现代 Whisper、Qwen2.5-Omni 等模型通常直接使用 Log-Mel 特征，而不是传统 MFCC。原因之一是 Log-Mel 保留了更完整的时间-频率结构，适合交给后续的 Conv、Transformer 或 Conformer 继续学习。

## 与 Audio in LLM 的关系

对于 Audio LLM，可以把这段历史理解为输入端的固定前端：

```text
Waveform
  → STFT / power spectrum
  → Mel filterbank
  → log compression
  → Audio Encoder
  → Adapter / projector
  → LLM continuous tokens
```

它解决的是：

- 将高采样率 waveform 转换为较低速率的时间-频率序列；
- 用人类听觉相关的频率尺度压缩频率维度；
- 用对数压缩减小动态范围；
- 为后续 Audio Encoder 提供稳定、结构化的输入。

但 Log-Mel 本身并不等于“语言表示”。它仍然是声学前端特征。Whisper Encoder、Qwen AuT 以及后续 Adapter 还需要继续完成：

```text
acoustic representation
  → contextual audio representation
  → LLM-compatible representation
  → semantic understanding / reasoning
```

## 后续待研究问题

- Bridle & Brown 1974 原始报告中 19-channel vocoder 的精确滤波器定义是什么？
- Mermelstein 1976 与 Davis & Mermelstein 1980 在 filterbank、log compression 和 DCT 上分别有哪些具体差异？
- 为什么现代深度模型通常保留 Log-Mel，而不再使用低维 MFCC？
- Whisper 的 80/128-bin Log-Mel 与 Qwen Omni 的音频前端在采样率、帧移和频率范围上有什么区别？
- Log-Mel 的固定归纳偏置，对 Audio LLM 的长音频理解和高频环境声建模有哪些损失？
- 可学习 audio frontend（如 SincNet、LEAF、raw waveform convolution）试图替代 Log-Mel 的哪些固定步骤？

## 参考资料

- [Stevens, Volkmann, and Newman, 1937](https://doi.org/10.1121/1.1915893)
- [D. Lyon, *Human and Machine Hearing*，关于 Mel-frequency cepstrum 的历史说明](https://dicklyon.com/hmh/Lyon_Hearing_book_01jan2018_smaller.pdf)
- [R. F. Lyon, “Acoustic Approaches and Auditory Influence,” Cambridge University Press](https://doi.org/10.1017/9781139051699.008)
- [Davis and Mermelstein, 1980](https://doi.org/10.1109/TASSP.1980.1163420)

## 后续处理

<!-- 状态可选：seed、exploring、promoted、dropped。升格后在此链接正式项目。 -->

- **处理决定**：
- **正式项目**：
- **决定日期**：
- **原因**：
