# AGENTS.md

给在这个仓库里工作的 AI 助手 / 新同学的关键信息。使用方法与各 stage 说明见 `README.md`，这里只记录容易踩坑、代码里看不出来的约定。

## 项目是什么

从开源音频数据集（Emilia / Emilia-YODAS / WenetSpeech4TTS / VoxBox / ISCSLP 等）中清洗出 TTS 训练语料。

- 输出是 WebDataset tar：`{key}.flac` + `{key}.json`。
- 发布在 HF dataset `leeoxiang/open-audio-data`，FSx 上有一份完整副本：`/mnt/fsx/leolxliu/open-audio-data/`。
- 目录结构是 `{name}/pack_XXXXX/{shard}-clean.tar` 加 `{name}/manifest.json`，每个 pack 约 100GB。

代码结构：
- `audio_pipeline/ops/`：各 stage，每个 stage 一个类，按成本从低到高排序。
- `audio_pipeline/datasets/`：每个数据源一个 loader，产出 `Sample`。
- `audio_pipeline/config.py`：由 `configs/*.yaml` 组装 stage 列表。
- `pipelines/run_*.py`：各数据集的入口。
- `scripts/`：多卡启动与看护脚本。
- `tools/`：上传、预览、交叉检查、重过滤工具。

## 数据 meta 规范（新增数据源必须对齐 Emilia）

**后续接入的其他数据源，输出 json 的 meta 字段必须与 Emilia 对齐。** 下游训练与抽检脚本按 Emilia 的字段读取，字段缺失或改名会导致静默出错或报错。

### loader 层字段

由 `audio_pipeline/datasets/*.py` 的 `Sample.meta` 写入，以 `audio_pipeline/datasets/emilia.py` 为准：

| 字段 | 类型 | 说明 |
|---|---|---|
| `duration` | float，单位秒 | 原始时长 |
| `speaker` | str | **全局唯一**的说话人 ID，需带数据源或语种前缀，避免与其他数据源撞名。例如 `ZH_B00000_S00000`、`EN_tKvmUvxYZXI_SPEAKER_00`。源数据没有说话人信息时，用最细粒度的来源 ID 构造（如节目或视频 ID），实在没有才写 `null`，但 key 必须存在 |
| `dnsmos` | float 或 null | 源数据自带的 DNSMOS。没有就写 `null`，key 必须存在 |
| `source` | str | 数据源 slug（`emilia`、`emilia-yodas`、`wenetspeech4tts`……），小写连字符 |

另外两点：
- `Sample.text` 与 `Sample.language` 传入源文本和源语种，pipeline 会把源文本另存为 `orig_text`。
- 数据源特有字段（`subset`、`emotion`……）可以放在顶层，但**不得覆盖或复用上表字段及下面各 stage 写入的字段名**。新增字段要在 loader 的 docstring 里写明含义。

### stage 层字段

由各 stage 自动写入，loader 不要自己写：

| stage | 写入的字段 |
|---|---|
| bandwidth | `rolloff_hz` |
| audiobox | `audiobox{CE,CU,PC,PQ}` |
| multi_speaker | `diarization{num_speakers, speaker_seconds, models, flagged_by}` |
| asr_wer | `asr{text, language, wer, char_wer}`、`text_source` |
| align | `alignment[{text, start, end}]` |
| trim | `trim{orig_duration, cut_head, cut_tail, pad_head, pad_tail}`，同时更新 `duration` |
| pause | `pause{max_gap, head_silence, tail_silence}` |
| tier | `tier`（`hd` 或 `basic`） |
| punct | `punct_fix`（只在有改动时写） |
| 收尾 | 最终的 `text`（训练用，来自 Qwen3-ASR），以及 `language`（ISO 639-1 小写：`zh`、`en`、`ja`、`ko`） |

### 接入前自检

新数据源先小规模跑一个分片（`--limit-samples 64`），取一条输出 json，与 Emilia 的样本逐 key 比较。缺 `duration`、`speaker`、`dnsmos`、`source`、`text`、`language` 中任何一个都不能上线。可以拿这条现成样本做对照：

```bash
python -I -c "import tarfile,json,sys;tf=tarfile.open(sys.argv[1]);m=next(x for x in tf if x.name.endswith('.json'));print(sorted(json.load(tf.extractfile(m))))" \
  /mnt/fsx/leolxliu/open-audio-data/emilia/pack_00000/$(ls /mnt/fsx/leolxliu/open-audio-data/emilia/pack_00000 | head -1)
```

### 已知未对齐的历史数据

以后修 loader 时顺手补齐：
- `wenetspeech`：缺 `speaker` 和 `dnsmos`。已发布数据就是这样，loader `audio_pipeline/datasets/wenetspeech.py` 只写了 `source`。
- `voxbox` / `iscslp`：没有 `dnsmos`。voxbox 也没有 `speaker`。

## 关键约定与决策

- **多说话人过滤 = Sortformer-v2 ∪ Nemotron-3-Diarization**，`min_speaker_sec=0`。任一模型判出第二说话人就丢弃，短附和也要过滤掉，这是用户明确要求的。
  - 依据：试听验证过，单用任一模型都有明显误判或漏判；Nemotron-3 的阈值和窗口参数都扫过，没有改善。
  - 精度：用 fp32。bf16 实测会漏掉约 4% 的多人样本，不要为了提速改成 bf16。
- **最终文本以 Qwen3-ASR 转写为准**。源文本是旧 ASR 的输出，不是 ground truth，只用来算 WER 一致性。
- 音频统一输出 32kHz/16bit FLAC，已经是 FLAC 的不要再二次转码。
- **FSx 上 `/mnt/fsx/leolxliu/open-audio-data/` 的文件不要改动或删除**：oa-data 的训练 manifest 引用了这些路径。重处理结果另写到工作目录，再覆盖上传到 HF。

## 环境

- `.venv` 用 `uv sync` 建，但有两处在 lock 之外手动装过，**再跑 `uv sync` 会把它们冲掉，跑完要重装**：
  - vLLM：`uv pip install "qwen-asr[vllm]"`，会把 torch 降到 2.9.1+cu128。
  - NeMo 3.1：Nemotron-3 需要。安装命令是 `uv pip install "nemo_toolkit[asr] @ /opt/dlami/nvme/leolxliu/Nemotron/nemo-speech" "lhotse==2.0.0a6"`。
- 跑 pipeline 前先检查：`.venv/bin/python -c "import vllm, nemo"`。
- `HF_HOME=/opt/dlami/nvme/leolxliu/.cache/huggingface`。
- HF token 在 `/opt/dlami/nvme/leolxliu/.env`，用之前先 `set -a; . .env; set +a`。不要把 token 写进代码或日志。
- 机器是 8×H200，GPU 和 `/opt/dlami/nvme` 磁盘与他人共用：
  - **只用用户指定的卡**，脚本用 `GPU_LIST` 指定。
  - 大规模写盘前先看剩余空间。
- 从项目根目录运行 Python。cwd 在 nemo 包目录内时，`collections` 会被同名模块遮蔽；处理下载来的数据用 `python -I`。

## 运维经验

- **worker 运行期间不要改变输入目录里的文件集合**（删除、新增都不行）。
  - 早期按 glob 结果取模分配分片，删文件导致列表错位，两个 worker 写了同一个 tmp 文件。
  - 现在的新工具用 O_EXCL claim 动态领取分片（见 `tools/refilter_diar.py`）。
- 断点续跑依赖原子改名加完成标记，重跑会自动跳过已完成的分片。不要手工删除完成标记。
- 后台 worker 崩溃重启时会从工作区重新加载代码。**有长任务在跑时不要切换 git 分支**，也不要把代码改到不兼容的状态。
- HF：
  - 账号公开仓库有存储配额，超额返回 403 "exceeded your public storage space"。
  - 覆盖上传后旧的 LFS 文件仍然占配额，需要 `super_squash_history` 才能释放，释放有约 20 分钟延迟。
  - **squash 会不可逆地清空 repo 历史，必须先得到用户明确同意。**
- 多说话人过滤吞吐：fp32、双模型，每张 H200 约 90–100 条/s，瓶颈是 GPU。一张卡上超过 3 个 worker 没有收益。

## 协作习惯

- 与用户用中文交流。
- 删除数据、squash HF 历史、推送到 main 这类不可逆或对外的操作，先确认再做。
- 过滤策略的变更要用试听结果说话。可以做带标注功能的试听页面，让用户标注后汇总。
- `missevan-drama-downloader/` 与本项目无关，不要提交。
