# open-audio-pipeline

从多个开源音频数据集中提取优质的 TTS 训练语料。

## 设计

- **原子能力** (`audio_pipeline/ops/`): 每个能力一个 Stage 类,可自由组合。按处理成本从低到高排默认顺序,便宜的过滤器先跑,减少贵模型的计算量:

  | Stage | 能力 | 模型 | 过滤条件 |
  |---|---|---|---|
  | `DurationFilter` | 时长/文本预过滤 | 无(读元数据) | 时长超界、空文本 |
  | `AudioboxFilter` | 美学评分 | facebook/audiobox-aesthetics | PQ/CE/CU 低于阈值、PC 高于阈值 |
  | `MultiSpeakerFilter` | 多说话人检测 | nvidia/diar_streaming_sortformer_4spk-v2 + nvidia/Nemotron-3-Diarization(并集) | 任一模型判说话人数 > 1(任何时长的第二说话人都算, 短附和也过滤) |
  | `AsrWerFilter` | 双 ASR 一致性 | Qwen/Qwen3-ASR-1.7B | 拼音级 WER > 阈值即拒;通过后采用 Qwen 转写为最终文本(源文本是旧 ASR 输出,非 ground truth) |
  | `TextNormalizationStage` | 文本正则化(TN,默认关) | Qwen/Qwen3-4B | 兜底:ASR 已用 context 偏置强制中文数字输出,仅当残留书面形式时再启用 |
  | `ForcedAlignStage` | 强制对齐 | Qwen/Qwen3-ForcedAligner-0.6B | 只标注字/词级时间戳,不过滤 |
  | `EdgeSilenceTrimStage` | 首尾静音规整 | 无(复用对齐时间戳) | 只修复:句首保留 100ms、句尾 300ms,超出裁剪、不足补零,重编码 FLAC,时间戳/时长同步平移 |
  | `AbnormalPauseFilter` | 异常停顿检测 | 无(复用对齐时间戳) | 句内静音 / 首尾静音超阈值(trim 之后首尾基本不再触发) |
  | `PausePunctuationStage` | 停顿标点修正 | 无(复用对齐时间戳) | 只标注:≥300ms 无标点停顿补逗号,≥800ms 补句号 |

- **数据集入口** (`pipelines/run_*.py`): 每个数据集一个入口文件 + 一份 `configs/*.yaml` 阈值配置。
- **输出**: 每个输入 tar 分片产出一个 `{name}-clean.tar`(WebDataset 布局: `{key}.mp3` + `{key}.json`,json 内含原文本与全部 stage 标注)和一个 `{name}-drops.jsonl`(被拒样本的原因与评分,用于调阈值)。
- **断点续跑**: 输出先写 tmp 再原子改名,完成后落 `.done/` 标记,重跑自动跳过。
- **性能**: 所有模型每个 worker 进程只初始化一次,跨分片复用;后台线程预取下一批(tar 读取 + 线程池并行解码)与当前批 GPU 计算重叠;解码失败单样本记 `drop:input` 不中断;ASR 默认 vLLM 后端(稳态比 transformers 快 ~12 倍,引擎启动 ~55s 为一次性开销);多卡多 worker 用 `bash scripts/launch_workers.sh`(每卡 2 进程时 `asr.gpu_memory_utilization` ≤0.15)。共享 H200 单卡实测 ~48x 实时(含加载),稳态 ~75x。

## 使用

```bash
uv sync   # 首次安装依赖 (torch/NeMo/qwen-asr/audiobox 等)

# Emilia (tar 内 {id}.json + {id}.mp3)
python pipelines/run_emilia.py \
    --input "/data/Emilia/ZH/*.tar" \
    --output /data/emilia-clean \
    --config configs/emilia.yaml --device cuda:0

# WenetSpeech4TTS (tar.gz 内 wavs/*.wav + txts/*.txt)
python pipelines/run_wenetspeech.py \
    --input "/data/WenetSpeech4TTS/Basic/*.tar.gz" \
    --output /data/wenetspeech-clean \
    --config configs/wenetspeech.yaml
```

多卡并行:每卡一个进程,按分片取模切分:

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i nohup python pipelines/run_emilia.py \
      --input "/data/Emilia/ZH/*.tar" --output /data/emilia-clean \
      --worker-id $i --num-workers 8 > worker$i.log 2>&1 &
done
```

调试:`--limit-samples 64` 每分片只处理前 64 条。

### 多说话人模型选择

`diarization.model` 可以是单个模型或模型列表(HF 名或本地 `.nemo` 路径)。默认配置用 v2 + Nemotron-3 两个模型取并集:
每个模型都跑全部样本,任一判多人即过滤,drops 的 reason 形如 `multi_speaker:2@<判多人的模型>`,
各模型结果分别记在 `meta.diarization.models`。Nemotron-3 需 NeMo >= 3.1:

```bash
uv pip install "nemo_toolkit[asr] @ <nemo-speech 源码目录>" "lhotse==2.0.0a6"   # 不动 torch/vLLM
```

注意 `uv sync` 会把 NeMo 降回 lock 里的 2.7.3(同 vLLM),之后需重装。
2026-11-20 在 Emilia ZH 上对比(空闲 H200,同为 NeMo 3.1):Sortformer-v2 73 条/s,Nemotron-3 64 条/s;
在 v2 判多人的 188 条上 Nemotron-3 只判出 107 条,随机 1000 条中两者分别判出 2 / 6 条(min_speaker_sec=0.5)。
改为 min_speaker_sec=0(现行策略)后:188 条中 Nemotron-3 判出 115 条,随机 1000 条中两者分别判出 10 / 19 条。
试听标注(v2 判多人、Nemotron-3 判单人的 32 条):23 条是 v2 误判,9 条是 Nemotron-3 漏掉的真多人,其中多条是短附和。
Nemotron-3 的判定阈值和流式窗口(含模型卡推荐的 340/40/40/300)都扫过,救不回这些漏判,所以两个模型取并集。
代价:这个 stage 的耗时约翻倍;512 条实测 multi_speaker 丢 10 条(仅 v2 判 2、两者都判 3、仅 Nemotron-3 判 5)。

## 结果预览

零依赖网页服务,统计概览 + 通过/丢弃样本列表 + 在线试听 + 完整标注:

```bash
python tools/preview_server.py \
    --output-dir /data/emilia-clean \
    --source "/data/Emilia/ZH/*.tar" \    # 可选: 让被丢弃的样本也能试听
    --port 8791
# code-server 下访问 https://<host>/proxy/8791/
```

gz 源分片(WenetSpeech)会在启动时后台预提取被丢弃样本的音频到
`<output-dir>/.preview_cache/`,提取完成前丢弃样本试听可能较慢。

## 上传到 HuggingFace

按 ~100GB 一个包(repo 内目录,HF 单文件上限 50GB)上传清洗输出,断点续传:

```bash
python tools/upload_hf.py --data /data/emilia-clean --name emilia \
    --repo leeoxiang/open-audio-data --pack-size-gb 100
```

repo 结构:`{name}/pack_00000/{shard}-clean.tar` + `{name}/manifest.json`(每包的分片清单/大小/样本数)。已上传的包记录在 `<data-dir>/.hf_upload_state.json`,重跑自动跳过。

## 调阈值

先小规模跑一批,分析 `drops.jsonl` 中各 stage 的分数分布再调 `configs/*.yaml`:

```bash
jq -r '.stage' output/*-drops.jsonl | sort | uniq -c          # 各 stage 丢弃量
jq -r 'select(.stage=="audiobox") | .meta.audiobox.PQ' output/*-drops.jsonl | sort -n  # PQ 分布
```

## 新增数据集

1. `audio_pipeline/datasets/` 加一个 `iter_xxx_tar(path) -> Iterator[Sample]`;
2. 复制一份 config,按数据集特性开关 stage / 调阈值;
3. 复制一份 `pipelines/run_*.py`,替换 loader。

### VoxBox (SparkAudio/voxbox)

音频分片与元数据分离(`audios/{subset}/*.tar.gz` + `metadata/{subset}.jsonl`),35 个子源混合中英,带情感/性别/韵律标签(保留进输出 meta):

```bash
python pipelines/run_voxbox.py \
    --input "/path/voxbox/audios/*/*.tar.gz" \
    --metadata-dir /path/voxbox/metadata \
    --output /data/voxbox-clean --config configs/voxbox.yaml
# --skip-subsets 默认排除 emilia_zh,wenetspeech4tts(与独立处理重叠,即去重方案)
```
