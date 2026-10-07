# Unsloth 전수조사 분석 & 활용·수익화 가이드 (한국어)

> 이 문서는 Unsloth 저장소를 전수조사(Python 2,719개 / TypeScript 2,204개 / Rust 36개 파일)하여
> **"이게 뭔지 · 어떨 때 쓰는지 · 어떻게 활용·수익화할 수 있는지"**를 정리한 분석 문서입니다.

| 항목 | 내용 |
|---|---|
| **작성일** | 2026-10-07 |
| **분석 대상 (이 저장소)** | <https://github.com/bmshin94/unsloth> |
| **원본 저장소 (upstream)** | <https://github.com/unslothai/unsloth> |
| **공식 사이트 / 문서** | <https://unsloth.ai> · <https://unsloth.ai/docs> |
| **다운로드** | <https://unsloth.ai/download> · <https://github.com/unslothai/unsloth/releases> |
| **노트북 모음** | <https://github.com/unslothai/notebooks> |
| **Docker Hub** | <https://hub.docker.com/r/unsloth/unsloth> · <https://hub.docker.com/r/unsloth/unsloth-rocm> |
| **커뮤니티** | <https://discord.gg/unsloth> · <https://reddit.com/r/unsloth> · <https://x.com/UnslothAI> |
| **분석 기준 커밋** | `f766d0c` (Merge PR #1: docs: add CLAUDE.md project guide) |
| **작업 브랜치** | `claude/kind-goldberg-2o49x1` |

---

## 목차

1. [한 줄 요약](#1-한-줄-요약)
2. [규모와 구성](#2-규모와-구성)
3. [폴더 전수조사 결과](#3-폴더-전수조사-결과)
4. [아키텍처](#4-아키텍처)
5. [쉬운 설명 (비유)](#5-쉬운-설명-비유)
6. [언제 쓰고, 언제 안 쓰는가](#6-언제-쓰고-언제-안-쓰는가)
7. [설치 및 사용법](#7-설치-및-사용법)
8. [플러그인인가 / 스킬인가 / MCP인가](#8-플러그인인가--스킬인가--mcp인가)
9. [API 토큰이 필요한가](#9-api-토큰이-필요한가)
10. [AI 에이전트 구축에 도움이 되는가](#10-ai-에이전트-구축에-도움이-되는가)
11. [React / PHP로 만들 수 있는가](#11-react--php로-만들-수-있는가)
12. [라이선스 — 사업 전 필독](#12-라이선스--사업-전-필독)
13. [유튜브 강의 제작 가능성](#13-유튜브-강의-제작-가능성)
14. [수익화 아이디어 상세](#14-수익화-아이디어-상세)
15. [리스크 체크리스트](#15-리스크-체크리스트)
16. [90일 실행 로드맵](#16-90일-실행-로드맵)

---

## 1. 한 줄 요약

> **Unsloth = 내 PC에서 AI 모델을 "돌리고(추론) + 학습시키는(파인튜닝)" 올인원 데스크톱 앱 + 고속 학습 엔진**

클라우드 AI(OpenAI, Claude)에 토큰비를 내는 대신, **오픈소스 모델을 내 그래픽카드에 받아 직접 실행하고,
내 데이터로 재학습시켜 "나만의 전용 AI"를 만드는** 소프트웨어입니다.

핵심 성과: **같은 결과물, 2~5배 빠른 학습, VRAM 최대 70% 절감, 정확도 손실 없음.**
→ 수천만원짜리 서버 대신 **RTX 3090/4090급 게이밍 GPU로도 파인튜닝이 됩니다.**

---

## 2. 규모와 구성

| 항목 | 수치 |
|---|---|
| Python 파일 | 2,719개 (약 169만 줄) |
| TypeScript / TSX | 2,204개 (약 54만 줄) |
| Rust | 36개 |
| 테스트 파일 | 300개 이상 |

한 저장소에 **4개의 독립 제품**이 들어있는 구조입니다.

| 구성 | 경로 | 정체 | 라이선스 |
|---|---|---|---|
| **① Core (엔진)** | `unsloth/` | 고속 학습 Python 라이브러리 | **Apache 2.0** ✅ |
| **② Studio (제품)** | `studio/` | FastAPI 백엔드 + React 프론트 + Tauri 앱 | **AGPL-3.0** ⚠️ |
| **③ CLI (명령줄)** | `unsloth_cli/` | CLI + 코딩 에이전트 연결기 | **AGPL-3.0** ⚠️ |
| **④ 설치 인프라** | `install.sh`, `install.ps1`, `studio/install_*.py`, `docker/` | 환경지옥 자동해결 (약 100만 줄) | 혼합 |

> `LICENSE:190-191` 원문
> - `unsloth/*`, `tests/*`, `scripts/*` → Apache 2.0
> - `studio/*`, `unsloth_cli/*` (설치 선택사항) → AGPLv3

---

## 3. 폴더 전수조사 결과

### ① `unsloth/` — 엔진 (Unsloth Core)

**Unsloth의 심장.** 순수 Python 라이브러리이며 원래의 Unsloth입니다.

| 파일 / 폴더 | 역할 |
|---|---|
| `kernels/` | **핵심 비밀병기.** GPU 연산을 Triton으로 직접 재작성 — `cross_entropy_loss.py`, `fast_lora.py`, `rms_layernorm.py`, `rope_embedding.py`, `swiglu.py`, `geglu.py`, `flex_attention.py`, `fp8.py`, `layernorm.py`, `moe/` |
| `models/` | 모델별 최적화 패치 — `llama.py`, `llama4.py`, `qwen2/qwen3/qwen3_moe.py`, `gemma.py`, `gemma2.py`, `mistral.py`, `glm4_moe.py`, `cohere.py`, `granite.py`, `falcon_h1.py`, `vision.py`(멀티모달), `diffusion.py`(이미지), `rl.py`(강화학습), `dpo.py`, `sentence_transformer.py` |
| `loader.py` | 사용자 진입점 — `FastLanguageModel`, `FastVisionModel`, `FastTextModel`, `FastModel` |
| `save.py` | **약 33만 줄.** 내보내기 — GGUF / LoRA 병합 / Ollama / HuggingFace 업로드 |
| `chat_templates.py` | 약 13만 줄. 모델별 대화 포맷 수백 종 |
| `import_fixes.py` | **약 33만 줄.** transformers/peft/trl/torch 버전 충돌을 런타임 패치 |
| `trainer.py` | 학습 루프 |
| `optimizers/` | 메모리 절약 옵티마이저 (Q-GaLore) |
| `dataprep/` | 전처리 + **합성 데이터 생성** (`synthetic.py`) |
| `registry/` | 지원 모델 카탈로그 (`_llama`, `_qwen`, `_gemma`, `_deepseek`, `_mistral`, `_phi`) |
| `utils/packing.py`, `prefix_grouper.py` | 패딩 제거·패킹으로 속도 향상 |
| `_compressed_quantize.py`, `_gpu_init.py`, `device_type.py` | 양자화 / GPU 초기화 / 디바이스 추상화 |

**동작 원리:** 표준 HuggingFace 학습 경로의 비효율적 연산을 자기가 쓴 Triton 커널로 **런타임에 바꿔치기(monkey patching)** 합니다. 사용자 코드는 그대로인데 속도·메모리만 개선됩니다.

### ② `studio/` — 웹 UI + 데스크톱 앱 (Unsloth Studio)

```
studio/
├── backend/            FastAPI 서버
│   ├── routes/         REST API 27개
│   ├── core/           기능 구현 본체
│   ├── auth/           인증 (비밀번호 / API키 / 해싱 / 정책)
│   ├── mcp_server.py   ★ Unsloth를 MCP 서버로 노출
│   ├── cloudflare_tunnel.py  외부 HTTPS 접속
│   ├── lan_access.py   집 네트워크 공유
│   ├── integrations/blender/  Blender MCP 연동
│   └── plugins/        data-designer 시드 플러그인
├── frontend/           React 19 + TS + Vite + Tailwind 4
└── src-tauri/          Rust (Tauri 2) — 네이티브 앱 포장
```

**`routes/` (API 27개)**
`inference` `training` `training_history` `training_vram` `models` `datasets` `data_recipe`
`export` `rag` `research_runs` `chat_history` `chat_generation_runs` `mcp_servers` `skills`
`providers` `provider_credentials` `whisper` `video` `youtube` `preview` `prompts`
`profile_stats` `settings` `auth` `accounts` `llama` `llama_compat` `openai_codex_auth`

**`core/inference/` (130개+ 파일) — 가장 방대한 영역**

| 그룹 | 파일 |
|---|---|
| GGUF 실행 | `llama_cpp.py`, `llama_http.py`, `llama_server_args.py`, `llama_keepwarm.py`, `llama_admission.py`, `llama_tool_schema.py` |
| Apple 실리콘 | `mlx_inference.py` |
| 이미지 생성 (**40개 파일**) | `diffusion_*.py` — Flux, HiDream, Krea2, Ideogram4, ControlNet, 양자화, CUDA Graph, 컴파일 캐시 등 |
| 영상 생성 | `video.py`, `video_ltx2.py`, `video_minimax_h3*.py`, `video_families.py` |
| 음성 → 텍스트 | `stt_sidecar.py`, `stt_ggml_sidecar.py`, `stt_registry.py`, `stt_transformers_worker.py` |
| 이미지 생성 엔진 | `sd_cpp_engine.py`, `sd_cpp_server.py`, `sd_cpp_backend.py` |
| **도구 / 에이전트** | `tools.py` (**약 1.8만 줄**), `tool_call_parser.py`, `tool_loop_controller.py`, `tool_stream_exec.py`, `tool_confinement.py`, `tool_path_approval.py`, `studio_tool_loop.py`, `native_tool_tokens.py`, `sandbox_site` |
| **MCP** | `mcp_client.py`, `mcp_config_import.py` |
| **Skills** | `skills.py`, `bundled_skills/skill-creator` |
| 외부 공급자 | `providers.py`, `external_provider.py`, `pricing.py`, `api_monitor.py`, `provider_model_capabilities.py`, `key_exchange.py` |
| 호환 레이어 | `anthropic_compat.py`(Anthropic Messages API ↔ OpenAI 변환), `openai_responses_shared.py`, `openai_codex_*.py` |
| 컨텍스트 / 메모리 | `context_window.py`, `memory_contract.py`, `context_refusal.py` |
| 오프로딩 | `offload_planner.py`, `offload_cost_model.py`, `offload_layout.py` |

**`core/rag/` (19개 파일)** — `parsers.py`, `chunking.py`, `embeddings.py`, `retrieval.py`, `store.py`, `ingestion.py`, `pdf_ocr.py`, `folder_sync.py`(폴더 자동 동기화), `captioner.py`(이미지 설명), `web_rank.py`, `conversation_archive.py`, `job_leases.py`

**`core/training/`** — `lifecycle.py`, `resume.py`(중단 후 재개), `worker.py`, `trainer.py`, `account_jobs.py`, `s3_dataset.py`, `eval_dataset.py`, `provenance.py`(출처 추적), `dataset_bounds.py`, `diffusion_*_trainer.py`(이미지 모델 학습 7개), `fsdp2_design_notes.md`

**`core/data_recipe/`** — PDF/CSV/DOCX → 학습 데이터셋 자동 변환 (`service.py`, `huggingface.py`, `export.py`, `jobs/`, `oxc-validator`)

**`core/research/`** — Deep Research (`citations.py`, `parsing.py`, `prompts.py`, `redaction.py`)

**`core/export/`** — `export.py`, `orchestrator.py`, `worker.py`

**AI에게 주는 내장 도구 10종** (`tools.py`)
`web_search` · `python`(코드 실행) · `terminal`(쉘) · `read_file` / `edit_file` · `render_html` ·
`search_knowledge_base`(RAG) · `search_conversation` · `read_skill` · `create_skill` · `deep_research`
→ 각각 샌드박스 버전과 full-access 버전이 분리되어 있음

**외부 공급자 14종** (`core/inference/providers.py`)

| 유료 API | 로컬 / 셀프호스팅 |
|---|---|
| `openai`, `openai_codex`, `anthropic`, `gemini`, `deepseek`, `mistral`, `kimi`, `qwen`, `openrouter`, `huggingface` | `ollama`, `vllm`, `llama_cpp`, `custom` |

**프론트엔드 `src/features/` (30개 모듈)**
`chat` `training` `train-model-picker` `model-picker` `loaded-models` `rag` `images` `video` `audio`
`export` `datasets`(`dataset-picker`) `data-recipes` `recipe-studio` `deep-links` `api-monitor`
`credentials` `security` `settings` `tour` `hub` `auth` `hf-auth` `profile` `generation-presets`
`igpu-carveout`(내장GPU 메모리 분배) `native-intents` `find-in-page` `studio` `transformers-upgrade`

**프론트 스택:** React 19.2 · TypeScript · Vite · Tailwind CSS 4 · TanStack(Router/Table/Virtual) ·
Radix UI + shadcn + Base UI · Zustand · Dexie(IndexedDB) · `@assistant-ui/react` + `assistant-stream` ·
`streamdown` · `recharts` · `@xyflow/react` + dagre(노드 에디터) · Motion · react-pdf / unpdf / mammoth ·
Tauri 2 플러그인

### ③ `unsloth_cli/` — CLI + 코딩 에이전트 연결기

| 파일 | 역할 |
|---|---|
| `commands/start.py` | **약 26만 줄.** `unsloth start` — 코딩 에이전트를 로컬 모델에 연결 |
| `commands/studio.py` | 약 19만 줄. Studio 실행 / 설치 / 업데이트 / 비밀번호 |
| `claude_subagent_mcp.py` | **Claude Code용 서브에이전트 MCP 서버** |
| `codex_subagent_mcp.py` | Codex용 서브에이전트 MCP |
| `pi_subagent.ts` | Pi용 서브에이전트 (TypeScript) |
| `codex_fallback_prompt.md` | Codex 폴백 프롬프트 |
| `commands/chat.py` `train.py` `export.py` `inference.py` | 각 CLI 명령 |
| `_inference.py` `_model_catalog.py` `_studio_deps.py` | 추론 / 카탈로그 / 의존성 |
| `_tool_policy.py` `_system_dir_guard.py` `_studio_runtime_gate.py` | 보안 가드 |

**CLI 명령 전체**

```
unsloth studio              # 웹 UI 실행
unsloth train               # 학습
unsloth chat                # 터미널 채팅
unsloth inference           # 추론
unsloth export              # 모델 내보내기
unsloth run                 # llama-server 직접 실행 (= unsloth studio run)
unsloth list-checkpoints    # 체크포인트 목록
unsloth start <agent>       # 코딩 에이전트 연결
```

**`unsloth start` 지원 에이전트**

| 에이전트 | 명령 |
|---|---|
| Claude Code | `unsloth start claude` |
| OpenAI Codex | `unsloth start codex` |
| DeepSeek Harness | `unsloth start dsh` |
| Hermes | `unsloth start hermes` |
| OpenCode | `unsloth start opencode` |
| OpenClaw | `unsloth start openclaw` |
| Pi | `unsloth start pi` |

### ④ 설치 인프라 — 숨은 핵심 가치

| 파일 | 규모 | 역할 |
|---|---|---|
| `install.ps1` | 약 63만 줄 | Windows 설치 |
| `studio/setup.ps1` | 약 51만 줄 | Windows 셋업 |
| `studio/install_python_stack.py` | 약 52만 줄 | torch 백엔드 자동 선택 + 의존성 |
| `studio/install_llama_prebuilt.py` | 약 46만 줄 | llama.cpp 바이너리 (CUDA/ROCm/Vulkan/CPU) |
| `install.sh` | 약 37만 줄 | macOS/Linux/WSL 설치 |
| `studio/setup.sh` | 약 24만 줄 | 셋업 |
| `pyproject.toml` | 약 15만 줄 | **flash-attn 휠 URL을 Python×torch×ABI 조합별로 전부 하드코딩** |
| `studio/install_whisper_prebuilt.py` | 약 9만 줄 | whisper.cpp |
| `studio/install_sd_cpp_prebuilt.py` | 약 4만 줄 | stable-diffusion.cpp |
| `studio/install_node_prebuilt.py` | 약 5만 줄 | Node.js 자동 설치 |
| `docker/` | — | `Dockerfile`, `Dockerfile.rocm`, `Dockerfile.studio` |

**왜 이렇게 큰가:**
`GPU(NVIDIA/AMD/Intel/Apple/없음) × OS(Win/Linux/WSL/macOS) × Python(3.9~3.14) × torch 버전 × CUDA 버전`
= 수백 가지 조합. 직접 맞추면 며칠 걸리는 작업을 한 줄로 해결합니다.

> **결론: Unsloth의 진짜 상품은 "속도"가 아니라 "속도 + 안 깨짐"입니다.**

---

## 4. 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│  사용 방법 3가지                                              │
│  A. Unsloth Desktop  (Tauri 네이티브 앱 — 설치만 하면 끝)      │
│  B. Unsloth Studio   (웹 UI — 브라우저 접속, 서버/원격 가능)   │
│  C. Unsloth Core     (Python 코드 직접 — Colab/노트북)        │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  studio/frontend  ─ React 19 + TS + Tailwind 4 + Vite        │
│  studio/src-tauri ─ Rust (앱 껍데기)                          │
└──────────────────────────────────────────────────────────────┘
                   ↓ REST / SSE / WebSocket
┌──────────────────────────────────────────────────────────────┐
│  studio/backend ─ FastAPI                                     │
│  routes(27) → core(inference / training / rag / export /      │
│                    research / data_recipe)                    │
│  + auth + mcp_server + cloudflare_tunnel + lan_access        │
└──────────────────────────────────────────────────────────────┘
        ↓ 실행 엔진                      ↓ 학습 엔진
┌───────────────────────────┐  ┌──────────────────────────────┐
│ llama.cpp (GGUF)          │  │ unsloth/ (Core)              │
│ MLX (Apple)               │  │  kernels/ ← Triton 커스텀     │
│ stable-diffusion.cpp      │  │  models/  ← 모델별 패치       │
│ whisper.cpp               │  │  save.py  ← GGUF/LoRA 내보내기│
│ vLLM / 외부 API 14종       │  │  기반: PyTorch + HF + TRL/PEFT│
└───────────────────────────┘  └──────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  내 GPU — NVIDIA / AMD / Intel / Apple / Vulkan / CPU        │
│  (멀티 GPU 지원)                                              │
└──────────────────────────────────────────────────────────────┘
```

**표준 호환 API (언어 무관 연동의 핵심)**

```
POST /v1/chat/completions   ← OpenAI 호환
POST /v1/messages           ← Anthropic 호환 (anthropic_compat.py)
POST /v1/embeddings         ← 임베딩
GET  /v1/models
```

---

## 5. 쉬운 설명 (비유)

### 요리 비유

| 비유 | 실체 |
|---|---|
| 재료 | 오픈소스 AI 모델 (Qwen, Llama, Gemma, DeepSeek…) — 무료 |
| 주방 기구 | 내 그래픽카드(GPU) |
| 레시피 책 | 내 데이터 (회사 문서, 상담기록, 코드…) |
| **Unsloth Core** | **성능 개조한 가스레인지** — 같은 요리, 2~5배 빨리, 가스 70% 절약 |
| **Unsloth Studio** | **주방 전체 + 조리 모니터** — 버튼으로 조작 |
| **Unsloth Desktop** | **완성품 밀키트** — 설치만 하면 바로 |

### 두 가지 일

**① 돌리기 (추론)**

```
ChatGPT : 내 질문 → 인터넷 → OpenAI 서버 → 답   (돈 내고, 데이터 나감)
Unsloth : 내 질문 → 내 GPU → 답                  (공짜, 데이터 안 나감)
```

글만이 아니라 이미지 생성 · 영상 생성 · 음성인식까지 모두 로컬에서 동작합니다.

**② 학습시키기 (파인튜닝) — Unsloth의 본업**

> **파인튜닝 = 신입사원 교육**
> - 모델 다운로드 = 똑똑하지만 우리 회사를 모르는 신입 채용
> - 파인튜닝 = 우리 문서·매뉴얼·상담기록 1,000건으로 교육
> - 결과 = 우리 말투와 업무를 아는 전담 직원
>   (퇴사 없음 · 월급 없음 · 복제 무한)

### 유명해진 단 하나의 이유

```
일반 방식 : GPU 계산을 범용 라이브러리에 맡김         → 느리고 메모리 과다
Unsloth   : 자주 쓰는 연산 수십 개를 직접 재작성(Triton) → 같은 결과, 2~5배 빠름, 메모리 70%↓
```

`unsloth/kernels/`가 바로 그 "직접 다시 쓴 계산기"들입니다.

---

## 6. 언제 쓰고, 언제 안 쓰는가

### 쓰는 경우

| 상황 | 이유 |
|---|---|
| API 비용 부담 | 월 수십~수백만원 토큰비 → 전기요금 |
| 데이터를 외부로 못 보냄 | 의료·법률·금융·사내문서 — 전부 로컬 처리 |
| 도메인 특화 AI 필요 | 우리 말투·업계 용어로 파인튜닝 |
| GPU가 작음 | 8GB에서도 QLoRA 학습 가능 |
| 오프라인 환경 | 설치 후 인터넷 없이 동작 |
| 이미지/음성/영상까지 | diffusion · TTS · Whisper 지원 |
| 코딩 에이전트를 로컬로 | `unsloth start claude` |

### 안 쓰는 경우

| 상황 | 이유 |
|---|---|
| GPU가 아예 없음 | CPU/Vulkan 모드는 있으나 매우 느림 |
| 최고 성능이 필수 | 최신 Claude/GPT가 아직 우위 |
| 단순 챗봇만 필요 | 설치·운영 오버헤드 과다 |
| 대규모 동시 서빙 | vLLM / SGLang이 적합 |

### 하드웨어 요구사항

| VRAM | 가능한 것 |
|---|---|
| 없음 (CPU/Vulkan) | 작은 GGUF 실행만. 학습 사실상 불가 |
| 8GB (3060 Ti, 4060) | 1~4B 학습(QLoRA), 8B 실행 |
| 12~16GB (4070) | 7~8B 학습, 14B 실행 |
| **24GB (3090, 4090)** | **14B 학습, 32B 실행 — 가성비 최적점** |
| 48GB+ (A6000, 2×3090) | 32~70B 학습 |
| Apple M 시리즈 | 통합메모리 유리. MLX로 실행 + 학습 |

---

## 7. 설치 및 사용법

### A. 데스크톱 앱 (초보자 · 추천)

| OS | 파일 |
|---|---|
| Windows | `Unsloth-Desktop-Windows.exe` |
| macOS | `Unsloth-Desktop-MacOS.dmg` |
| Ubuntu | `Unsloth-Desktop-Ubuntu.deb` |
| Linux 기타 | `Unsloth-Desktop-Linux.AppImage` |

→ <https://unsloth.ai/download> · <https://github.com/unslothai/unsloth/releases/latest>

### B. Studio 웹 UI

```bash
# macOS / Linux / WSL
curl -fsSL https://unsloth.ai/install.sh | sh
unsloth studio                      # → http://127.0.0.1:8000
```

```powershell
# Windows
irm https://unsloth.ai/install.ps1 | iex
unsloth studio
```

**실행 옵션**

```bash
unsloth studio                      # 기본 (내 PC만)
unsloth studio -p 8888              # 포트 변경
unsloth studio -H 0.0.0.0 -p 8888   # LAN 공유 (폰/태블릿 접속)
unsloth studio --secure             # Cloudflare HTTPS 터널 → 전세계 접속
unsloth studio --disable-tools      # ⚠️ 외부 노출 시 필수
unsloth studio reset-password       # 비밀번호 재설정
unsloth studio stop                 # 종료
```

> ⚠️ **보안 경고:** 서버측 도구(`python` 실행, `terminal` 실행, 파일 편집)가 **기본 ON**입니다.
> 외부 노출 시 `--disable-tools`를 쓰거나 비밀번호를 철저히 관리하세요.
> 그렇지 않으면 **타인이 내 컴퓨터에서 임의 코드를 실행**할 수 있습니다.

```bash
# 비대화식 비밀번호 (서버/도커)
UNSLOTH_STUDIO_PASSWORD='강력한비번' unsloth studio --secure
```

### C. Docker

```bash
# Linux GPU 1회 설정
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/install_nvidia_toolkit.sh \
  -o install_nvidia_toolkit.sh && sudo -E bash install_nvidia_toolkit.sh

docker run -d --name unsloth --gpus all --ipc=host \
  -p 8000:8000 -p 8888:8888 \
  -v "$PWD":/workspace/host \
  -v "$HOME/.cache/huggingface":/workspace/.cache/huggingface \
  -v unsloth-studio:/opt/unsloth-studio \
  unsloth/unsloth && docker logs -f unsloth
```

| 이미지 | 용도 |
|---|---|
| `unsloth/unsloth` | 전체 (Studio + JupyterLab) |
| `unsloth/unsloth:core` | 노트북만 (가벼움) |
| `unsloth/unsloth-rocm` | AMD GPU (네이티브 Linux만 — WSL은 `/dev/kfd` 미노출로 불가) |

### D. Core (개발자)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv unsloth_env --python 3.13
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto
```

**전체 학습 코드**

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1) 모델 로드 (4bit 양자화 = VRAM 1/4)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen3-8B",
    max_seq_length = 2048,
    load_in_4bit   = True,
)

# 2) LoRA 어댑터 부착 (전체가 아니라 일부만 학습)
model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,          # 랭크 — 높이면 표현력↑ 메모리↑
    lora_alpha     = 16,
    lora_dropout   = 0,
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
)

# 3) 내 데이터
dataset = load_dataset("json", data_files="my_data.jsonl", split="train")

# 4) 학습
trainer = SFTTrainer(
    model = model, tokenizer = tokenizer, train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps                   = 60,
        learning_rate               = 2e-4,
        optim                       = "adamw_8bit",
        output_dir                  = "outputs",
    ),
)
trainer.train()

# 5) 내보내기 → Ollama / LM Studio / llama.cpp에서 바로 사용
model.save_pretrained_gguf("my-model", tokenizer, quantization_method="q4_k_m")
```

**GPU 없이 무료 테스트**
- Studio 체험: <https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb>
- 노트북 모음: <https://github.com/unslothai/notebooks>

### 실전 사용 순서 (Studio UI)

```
1. Models      모델 검색·다운로드 (HuggingFace 연동)
2. Chat        바로 대화 (도구 / MCP / RAG 활성화)
3. Datasets    내 데이터 업로드
   Data Recipe PDF/CSV/DOCX → 학습 데이터셋 자동 변환
4. Train       모델+데이터 선택 → VRAM 계산기 확인 → 학습 시작
               (중단해도 resume.py로 재개)
5. Export      GGUF / NVFP4 / FP8 / LoRA병합 / Ollama / HF업로드
6. Settings    API 키 발급, LAN 접속, MCP 서버, Skills 관리
```

### 삭제

```bash
# macOS / Linux / WSL
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.sh | sh
```
```powershell
# Windows
irm https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.ps1 | iex
```

모델 파일은 별도 삭제: `~/.cache/huggingface/hub/` (Windows: `%USERPROFILE%\.cache\huggingface\hub\`)

---

## 8. 플러그인인가 / 스킬인가 / MCP인가

> **결론: 셋 다 아닙니다. Unsloth는 "독립 애플리케이션 + 라이브러리"입니다.**
> 다만 **MCP를 양방향 지원**하고 **Agent Skills를 호스팅**하며, 자체 플러그인 구조도 있습니다.

| 분류 | 맞나 | 설명 |
|---|:---:|---|
| Claude Code 플러그인 | ❌ | 아님 (`plugin.json` 없음) |
| Agent Skill | ❌ | Unsloth 자체는 스킬이 아님 |
| **MCP 서버** | ⭕ 부분 | **Studio가 MCP 서버가 될 수 있음** (옵트인) |
| **MCP 클라이언트** | ⭕ | **외부 MCP 서버를 붙여 사용** |
| **Skills 호스트** | ⭕ | **Agent Skills를 읽고 실행** |
| **독립 앱 + 라이브러리** | ✅ | 정답 |

### ① Unsloth가 MCP 서버가 되는 경우 — 다른 AI가 Unsloth를 조종

`studio/backend/mcp_server.py` · 문서 `studio/MCP.md`

```bash
UNSLOTH_STUDIO_ENABLE_MCP=1 \
UNSLOTH_STUDIO_MCP_TOKEN='로컬-시크릿' \
unsloth studio
# 엔드포인트: http://127.0.0.1:8888/mcp/   (/mcp 요청은 /mcp/로 리다이렉트)
```

| 분류 | 노출 도구 |
|---|---|
| 탐색 | `studio_status`, `list_local_models` |
| 학습 | `get_training_status`, `start_training`, `stop_training`, `list_training_runs` |
| 레시피 | `validate_recipe`, `get_recipe_job_status`, `get_recipe_job_dataset` |
| 모델 | `load_checkpoint`, `export_gguf` |

> **의미: 이걸 켜면 Claude Code가 "모델 학습을 시작해"라고 말로 지시할 수 있습니다.**
> AI가 AI를 학습시키는 구조 → MLOps 자동화 에이전트 제작 가능.
>
> ⚠️ 기본 **OFF**, `UNSLOTH_STUDIO_MCP_TOKEN` **필수** (HTTP·WebSocket 모두 Bearer 토큰 정확 검증).
> 이유: 이 도구들이 GPU 메모리를 점유하고 모델 파일을 쓰고 진행 중인 작업을 중단시킬 수 있기 때문.

### ② Unsloth가 MCP 클라이언트가 되는 경우 — 내장 AI가 외부 도구 사용

`core/inference/mcp_client.py`, `mcp_config_import.py`, `routes/mcp_servers.py`

- UI의 **Manage MCP servers**로 외부 MCP 서버 등록
- 기존 Claude Desktop MCP 설정 **import 가능**
- 번들 예시: **Blender MCP** (`integrations/blender/`) — AI가 말로 3D 모델링.
  기본 비활성, 체크섬 검증된 런타임만 다운로드, 기본 포트 `9876`

### ③ Agent Skills 지원

`core/inference/skills.py` · `routes/skills.py` · `bundled_skills/skill-creator`

```python
source: Literal["agents", "claude", "bundled"]   # 3가지 출처
enabled, valid, shadowed, shadowed_by            # 상태 관리
license, compatibility, allowed_tools            # 메타데이터
```

→ `.claude/skills/` 의 스킬을 그대로 읽어 사용. 내장 AI에게 `read_skill` / `create_skill` 도구도 제공.
**즉 Claude Code용으로 만든 스킬을 로컬 모델에서 재사용할 수 있습니다.**

### ④ Unsloth 자체 플러그인

`studio/backend/plugins/` — `data-designer-github-repo-seed`, `data-designer-unstructured-seed`
데이터셋 생성기 확장용. 외부 공개 플러그인 API는 아직 아님.

### ⑤ 그리고 가장 중요한 것 — 표준 API 호환

```
POST /v1/chat/completions   OpenAI 호환
POST /v1/messages           Anthropic 호환
POST /v1/embeddings
GET  /v1/models
```

> **플러그인이냐 MCP냐 고민할 필요 없이, OpenAI SDK를 쓰는 모든 코드에서 `base_url`만 바꾸면 됩니다.**
> 언어 무관 (PHP · Node · Java · Go · C# · Python). 실무에서 가장 중요한 연동 포인트.

```bash
unsloth start claude --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

---

## 9. API 토큰이 필요한가

> **결론: Unsloth 자체는 토큰이 전혀 필요 없습니다. 100% 무료 오픈소스, 계정 가입도 없습니다.**

| # | 토큰 | 필수 | 언제 | 비용 |
|---|---|:---:|---|---|
| 1 | Unsloth 자체 | ❌ | — | 무료 |
| 2 | HuggingFace 토큰 | △ | **게이트 모델** 다운로드 (Llama, Gemma 등) | 무료 |
| 3 | Studio API 키 | △ | Studio를 외부/프로그램에서 호출 | 무료(자체 발급) |
| 4 | MCP 토큰 | ⭕ | MCP 서버 사용 시 (`UNSLOTH_STUDIO_MCP_TOKEN`) | 무료(자체 지정) |
| 5 | 외부 공급자 키 | ❌ | OpenAI/Claude/Gemini 혼용 시 | 💸 유료 |
| 6 | Studio 비밀번호 | ⭕ | 외부 노출(`--secure`, `0.0.0.0`) 시 강제 | 무료 |

**② HuggingFace** — `unsloth/` 접두사 모델(예: `unsloth/Qwen3-8B`)은 Unsloth 팀 재배포본이라 **토큰 없이 받아집니다.**
Meta Llama 원본처럼 게이트된 모델만 토큰 필요. Studio에 `hf-auth` 기능 내장.

**③ Studio API 키** — `Settings > API keys` (`POST /auth/api-keys`, SQLite `api_keys` 테이블, 해시 저장)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer unsl_xxxxx" \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"안녕"}]}'
```

> 설계상 **API 키로는 UI 전용 민감 작업이 차단**됩니다 (`require_ui_session(via_api_key)`).
> 키가 유출돼도 설정 변경·키 조회는 불가.

**⑤ 외부 공급자 14종** — `pricing.py` + `api_monitor.py`가 **토큰 비용을 실시간 추적**.
하이브리드 전략(간단 작업=로컬, 중요 작업=유료API) 운영에 유용.

### 보안 설계

| 메커니즘 | 파일 |
|---|---|
| **RSA 공개키 암호화** — 브라우저에서 암호화 전송, 평문 저장 안 함 | `core/inference/key_exchange.py` |
| API 키 해시 저장 (평문 미보관) | `auth/hashing.py`, `auth/storage.py` |
| API 키 ≠ UI 세션 권한 분리 | `routes/providers.py` |
| 도구 실행 샌드박스 + 경로 승인 | `tool_confinement.py`, `tool_path_approval.py`, `sandbox_site` |
| 외부 노출 시 비밀번호 강제 프롬프트 | `auth/terminal_prompt.py` |
| 부트스트랩 데드라인 (비번 없으면 자동 종료) | `auth/bootstrap_timeout.py` |

---

## 10. AI 에이전트 구축에 도움이 되는가

> **결론: ⭐⭐⭐⭐☆ 매우 큰 도움. 단, "에이전트 두뇌"가 아니라 "에이전트 인프라" 역할.**

### 이미 갖춰진 부품 (직접 만들 필요 없음)

| 에이전트 필수 요소 | Unsloth 제공 | 파일 |
|---|---|---|
| LLM 서빙 | llama.cpp / MLX / vLLM / 외부API 14종 | `llama_cpp.py`, `mlx_inference.py` |
| 도구 호출 | 파서·루프·치유 완비 | `tool_call_parser.py`, `tool_loop_controller.py`, `tool_healing.py` |
| 내장 도구 10종 | web_search, python, terminal, read/edit_file, render_html, search_knowledge_base, search_conversation, read_skill, create_skill, deep_research | `tools.py` |
| 샌드박싱 | 격리 실행 + 경로 승인 | `tool_confinement.py`, `sandbox_site` |
| MCP 클라이언트 | 외부 도구 무한 확장 | `mcp_client.py` |
| MCP 서버 | 다른 AI의 도구가 됨 | `mcp_server.py` |
| Agent Skills | 로드 / 검증 / shadowing | `skills.py` |
| 서브에이전트 | Claude/Codex/Pi용 MCP | `claude_subagent_mcp.py` 등 |
| RAG | 파싱→청킹→임베딩→검색 풀스택 | `core/rag/` (19파일) |
| Deep Research | 다단계 조사 + 인용 + 레닥션 | `core/research/` |
| 메모리/컨텍스트 | 자동 압축, 롤링 윈도우 | `memory_contract.py`, `context_window.py` |
| 멀티모달 | 이미지/비전/음성/영상 | `diffusion_*`, `stt_*`, `video_*` |
| 스트리밍 | SSE 제어 프레임 | `sse_control_frames.py` |
| 비용 추적 | 토큰·가격 모니터 | `pricing.py`, `api_monitor.py` |
| 웹 접근 정책 | 도메인 정책 제어 | `web_access_policy.py` |

### 4가지 활용 패턴

**패턴 A — 에이전트 런타임으로 사용 (가장 쉬움)**

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unsl_xxx")
# 기존 에이전트 코드 그대로. 모델만 로컬로.
```

→ LangChain, LlamaIndex, CrewAI, AutoGen, Vercel AI SDK 전부 그대로 연결됩니다.

**패턴 B — 에이전트가 Unsloth를 도구로 사용 (MCP 서버 모드)**

```
Claude Code ──MCP──> Unsloth ──> 모델 학습 / 내보내기
"어제 수집한 데이터로 Qwen3-8B 파인튜닝 시작하고 끝나면 GGUF로 내보내"
```

**패턴 C — 도구호출 전용 모델을 직접 학습 (가장 강력)**

```
문제: 로컬 소형 모델은 Tool Calling을 자주 틀림 → 에이전트로 못 씀
해결: 1) 큰 모델(Claude/GPT)로 "완벽한 도구호출 대화" 2,000~10,000건 합성
      2) Unsloth로 4B 모델에 그 패턴을 학습
      3) 결과: 우리 도메인 도구는 대형모델급으로 호출하는 초저비용 모델
```

재료가 이미 있음: `dataprep/synthetic.py`(합성 데이터), `tool_call_parser.py`, `native_tool_tokens.py`

**패턴 D — 하이브리드 라우팅 (실무 최적)**

```
간단/대량 작업 → 로컬 모델 (무료)
복잡/중요 작업 → Claude/GPT (유료)
→ 한 UI에서 혼용 (providers.py), api_monitor.py로 절감액 확인
```

### 솔직한 한계

| 한계 | 설명 | 대응 |
|---|---|---|
| 에이전트 프레임워크가 아님 | 플래너/상태머신/오케스트레이터 없음 | LangGraph 등과 조합 |
| 로컬 모델 추론력 격차 | 복잡한 다단계 추론은 Claude/GPT 우위 | 하이브리드 |
| GPU 종속 | GPU 없으면 실용성 없음 | 클라우드 GPU |
| 동시성 한계 | 개인/소팀용. 대규모 서빙은 vLLM/SGLang | 서빙 분리 |
| AGPL | Studio 기반 SaaS는 소스공개 의무 | Core(Apache)만 사용 |
| 빠른 변화 | 릴리스가 매우 잦음 | 버전 고정 |

---

## 11. React / PHP로 만들 수 있는가

| 만들려는 것 | React | PHP | 판정 |
|---|:---:|:---:|---|
| GPU 커널 / 학습 엔진 | ❌ | ❌ | **절대 불가** — Python + CUDA/Triton 필수 |
| 모델 추론 엔진 | ❌ | ❌ | 불가 — C++/CUDA 영역 |
| UI / 대시보드 | ✅ | ✅ | **가능 — Unsloth가 이미 React로 구현** |
| 백엔드 오케스트레이션 | ⭕ Node | ⭕ | 가능 (프로세스 실행 + HTTP 프록시) |
| Unsloth API 호출 앱 | ✅ | ✅ | **완전 가능 — 추천 경로** |
| 상용 래퍼 제품 | ⚠️ | ⚠️ | 가능하나 **AGPL 주의** |

### React — 이미 React입니다

`studio/frontend/`을 포크하면 React 프론트엔드를 그대로 재사용할 수 있습니다.
30개 feature 모듈이 대형 React 앱 설계 참고자료로도 훌륭합니다.

### PHP — 엔진은 불가, 앱은 완전 가능

PyTorch/CUDA/Triton 바인딩이 PHP에 없으므로 **PHP로 GPU 학습은 불가**합니다.
그러나 **OpenAI 호환 API**가 있으므로 앱 개발은 완전히 가능합니다.

```php
<?php
$ch = curl_init('http://127.0.0.1:8000/v1/chat/completions');
curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => [
        'Content-Type: application/json',
        'Authorization: Bearer ' . getenv('UNSLOTH_API_KEY'),
    ],
    CURLOPT_POSTFIELDS => json_encode([
        'model'    => 'my-finetuned-model',
        'messages' => [['role' => 'user', 'content' => $userInput]],
    ]),
]);
$res = json_decode(curl_exec($ch), true);
echo $res['choices'][0]['message']['content'];
```

공식 SDK로 더 간결하게:

```bash
composer require openai-php/client
```
```php
$client = OpenAI::factory()
    ->withApiKey(getenv('UNSLOTH_API_KEY'))
    ->withBaseUri('http://127.0.0.1:8000/v1')   // ← 이 한 줄이 전부
    ->make();
```

→ **Laravel · Symfony · WordPress 플러그인 전부 가능합니다.**

### 추천 아키텍처 (라이선스 안전)

```
┌────────────────────────────────────────────┐
│ 프론트: React / Next.js                     │ ← 우리 코드 (비공개 가능)
└────────────────────────────────────────────┘
                 ↓ REST
┌────────────────────────────────────────────┐
│ 앱 백엔드: PHP(Laravel) / Node / Python     │ ← 우리 코드 (비공개 가능)
│ 인증·결제·멀티테넌시·로그·과금·프롬프트·라우팅 │
└────────────────────────────────────────────┘
                 ↓ OpenAI 호환 HTTP  ← 라이선스 경계선
┌────────────────────────────────────────────┐
│ 추론: llama.cpp / vLLM / Ollama             │ ← Studio 대신 쓰면 AGPL 회피
└────────────────────────────────────────────┘
┌────────────────────────────────────────────┐
│ 학습: unsloth Core (Apache 2.0)             │ ← 상업 이용 자유
└────────────────────────────────────────────┘
```

> **핵심: 수익 제품에서는 학습만 Unsloth Core(Apache)로 하고, 서빙은 llama.cpp/vLLM으로.**
> Studio는 **내부 개발 도구로만** 사용하면 라이선스가 완전히 깔끔해집니다.

---

## 12. 라이선스 — 사업 전 필독

```
unsloth/, tests/, scripts/   → Apache 2.0  ✅ 상업 이용 자유, 소스공개 의무 없음
studio/, unsloth_cli/        → AGPL-3.0    ⚠️ 네트워크 서비스 제공 시 소스공개 의무
```

AGPL은 "**네트워크를 통해 사용자에게 서비스를 제공하면 수정한 소스를 사용자에게 공개**"해야 합니다
(GPL의 SaaS 구멍을 막은 조항).

| 시나리오 | 안전 여부 |
|---|:---:|
| Studio를 사내에서 사용 | ✅ 안전 |
| Studio를 수정 없이 설치해주고 구축비 수령 | ✅ 안전 |
| `unsloth/` Core로 학습 파이프라인 제작·판매 | ✅ 안전 (Apache) |
| 내 앱 → HTTP로 Studio 호출 (Studio 미수정) | 🟡 대체로 안전 (법률 검토 권장) |
| **Studio를 포크·수정해 SaaS 판매** | ❌ 소스공개 의무 |
| **Studio 프론트엔드를 내 제품에 복사** | ❌ 소스공개 의무 |

### 베이스 모델 라이선스도 별도로 따라붙습니다

| 모델 | 라이선스 | 상업화 |
|---|---|---|
| **Qwen / Mistral** | Apache 2.0 | ✅ **가장 안전 — 실무 권장** |
| Llama | Meta 커뮤니티 라이선스 | ⚠️ MAU 7억 제한 + 네이밍 의무 |
| Gemma | Google 사용약관 | ⚠️ 약관 확인 필수 |

> **상업화 전 베이스 모델 라이선스를 반드시 확인하세요.**

---

## 13. 유튜브 강의 제작 가능성

> **결론: ⭐⭐⭐⭐⭐ 가능하고, 가장 승산 높은 선택지입니다.**

### 왜 유리한가

| 근거 | 내용 |
|---|---|
| 라이선스 무관 | 영상·강의 제작은 제약 없음 (코드 재배포가 아님) |
| 공백 시장 | AI 파인튜닝 수요는 폭발, **한국어 영상은 거의 없음** |
| 시각적 임팩트 | "메모리 70% 절감", "2배 빠름"을 실시간 그래프로 증명 |
| 진입장벽 낮음 | Colab 무료 → **시청자가 GPU 없이 따라할 수 있음** |
| 결과물 명확 | "내 말투로 답하는 AI" 전후 비교 = 강한 Before/After |
| 갱신 수요 | 모델이 계속 출시 → 콘텐츠 무한 재생산 |
| CPM 최상위 | AI/개발 = 광고단가 높은 니치 |

### 시리즈 커리큘럼

**A. 입문 (조회수용 · 8~12분)**

| # | 제목 | 핵심 |
|---|---|---|
| 1 | 내 컴퓨터에서 ChatGPT 돌리기, 5분이면 끝 | 설치 + 첫 채팅 |
| 2 | GPU 없어도 됩니다 — Colab으로 AI 학습 | 무료 노트북 |
| 3 | AI가 내 말투를 흉내내게 만들기 | 100건 데이터 파인튜닝 |
| 4 | 내 PDF 1,000장을 AI에게 읽히기 | RAG |
| 5 | 로컬 AI로 이미지 생성 (무료 무제한) | diffusion |
| 6 | Claude Code를 공짜로 쓰는 방법 | `unsloth start claude` ← 클릭률 최상 |

**B. 실전 (15~25분)**

| # | 제목 |
|---|---|
| 7 | 학습 데이터 1,000건 만들기 — PDF/CSV → 데이터셋 (Data Recipe) |
| 8 | LoRA·QLoRA 완벽 이해 — 왜 메모리가 1/4이 되는가 |
| 9 | 하이퍼파라미터 실전 — r, alpha, lr, epoch 정하는 법 |
| 10 | GGUF 내보내기 → Ollama / LM Studio 배포 |
| 11 | VRAM 8GB에서 가능한 것 / 불가능한 것 (실측) |
| 12 | OpenAI API를 로컬로 교체하기 (PHP·Node·Python 예제) |

**C. 고급 (20~40분)**

| # | 제목 |
|---|---|
| 13 | GRPO로 추론 모델 직접 만들기 (DeepSeek-R1 방식) |
| 14 | MCP로 AI에게 도구 쥐어주기 |
| 15 | AI가 AI를 학습시킨다 — Unsloth MCP 서버 |
| 16 | Unsloth는 왜 빠른가 — Triton 커널 코드 리딩 |
| 17 | 사내 온프레미스 AI 구축 전과정 (보안·AGPL·운영) |

**D. 비즈니스 (구독 전환)**

| # | 제목 |
|---|---|
| 18 | AI API 비용 월 300만원 → 0원 만든 과정 |
| 19 | 파인튜닝 외주로 돈 버는 법 — 견적·범위·계약 |
| 20 | 오픈소스 라이선스 함정 — AGPL·Llama 라이선스 ← **희소성 최고** |

### 제작 팁

| 항목 | 권장 |
|---|---|
| 첫 15초 | "이거 무료입니다" + 결과물 먼저. 설명은 나중에 |
| 화면 구성 | 좌: 터미널/Studio · 우: `nvidia-smi` 실시간 VRAM 그래프 |
| 비교 샷 | 같은 작업 일반 vs Unsloth 병렬 실행 → 타이머 비교 |
| Before/After | 학습 전 답변 ↔ 학습 후 답변 나란히 |
| 재현성 | 모든 영상에 Colab 링크 + GitHub 레포 고정 댓글 |
| 에러 처리 | **설치 실패 장면을 일부러 넣고 해결** → 신뢰도·완주율 급상승 |
| 길이 | 입문 8~12분 / 실전 15~25분 |
| 업로드 | 주 1~2회, 입문 → 실전 → 고급 순 |

### 수익 경로

```
1. 애드센스            AI/개발 니치 = 고CPM
2. 멤버십 / Patreon    데이터셋·학습 스크립트·노트북 제공
3. 온라인 강의          인프런 / 클래스101 / Udemy
4. 전자책              "Unsloth 완전 가이드" PDF
5. 기업 교육·세미나  ★  단가 최고. 영상이 포트폴리오
6. 컨설팅·외주 유입  ★  실질 매출. 영상 본 기업이 직접 연락
7. 제휴                클라우드 GPU(Vast.ai, RunPod), GPU 하드웨어
```

> **핵심 전략: 유튜브 자체 수익보다 "기업 교육 + 컨설팅 유입 경로"로서의 가치가 10배 큽니다.**
> 영상 20개 = 살아있는 포트폴리오.

### 주의

| 항목 | 내용 |
|---|---|
| Unsloth 로고/이름 | 교육·리뷰 목적 사용은 문제 없음. 공식 제휴처럼 보이게 하지 말 것 |
| 모델 라이선스 | 영상에서 Llama 라이선스 제약을 반드시 언급 (신뢰도↑) |
| 버전 노후화 | 릴리스가 잦음 → 날짜/버전 명시, 설명란 업데이트 |
| 과장 금지 | "완전 무료"는 반쯤 틀림(전기·GPU·시간). 정직하게 가야 장기적으로 이김 |
| 보안 | `--secure` 노출 위험을 반드시 경고. 사고 시 책임 논란 |

---

## 14. 수익화 아이디어 상세

### 전략 지도

```
             투자(시간·돈) 적음 ──────────────────→ 많음
수익 빠름  ┌──────────────────┬──────────────────┐
    ↑      │ ① 교육·콘텐츠     │ ② 용역·구축       │
    │      │ (1~2개월)        │ (즉시~1개월)      │
    │      ├──────────────────┼──────────────────┤
    │      │ ③ 템플릿·자산     │ ④ 버티컬 SaaS     │
수익 느림  │ (2~3개월)        │ (6~12개월)       │
    ↓      └──────────────────┴──────────────────┘
```

### 1순위 — 용역 & 온프레미스 구축 (즉시 현금화)

**가장 빠르고, 라이선스가 완전히 안전하고, 단가가 높습니다.**

기업이 AI를 못 쓰는 1번 이유는 **"데이터를 외부에 보낼 수 없다"**입니다.
병원·법무법인·금융·제조·공공·국방 — 이들은 **"내부에만 있는 AI"에 돈을 아주 잘 씁니다.**

| 상품 | 범위 | 규모 |
|---|---|---|
| **A. 로컬 AI 구축 (기본)** | 서버 선정 · Unsloth 설치 · 모델 선택 · 사내 LAN 배포 · 교육 2시간 | 소형 |
| **B. 사내 문서 AI (RAG)** | A + 문서 수집·파싱·임베딩·RAG 구축·검색품질 튜닝 | 중형 |
| **C. 전용 모델 파인튜닝** | B + 데이터셋 설계·학습·평가·배포·재학습 1회 | 중~대형 |
| **D. 운영 유지보수** | 모델 업데이트·재학습·모니터링 | **월 구독 ★** |
| **E. API 비용 절감 진단** | 현 사용량 분석 → 로컬 전환 가능 작업 식별 → 절감액 리포트 | 진단 단건 → C 연결 |

> **E가 영업의 핵심 무기입니다.** "지금 월 OOO만원 쓰시는데 그중 OO%는 로컬로 옮겨도 품질 차이 없습니다"를
> **숫자로** 보여주면 계약이 쉽습니다. `api_monitor.py` + `pricing.py`가 그 데이터를 만들어 줍니다.

**라이선스 안전성:** Studio를 **수정 없이 고객사에 설치**하는 것은 단순 사용입니다.
AGPL은 "수정 후 서비스 제공"을 규제하며, **설치·구축·교육 용역비는 완전히 합법**입니다.

**영업 루트**
```
유튜브/블로그 콘텐츠 → 무료 진단(E) → 구축(A~C) → 유지보수(D, 월구독)
```
D가 쌓이면 현금흐름이 안정됩니다. 이것이 최종 목표.

### 2순위 — 교육 & 콘텐츠 (저위험 · 복리)

| # | 상품 | 특징 |
|---|---|---|
| 1 | 유튜브 채널 | 깔때기 입구 |
| 2 | 온라인 강의 | 한 번 만들면 계속 팔림. 한국어 Unsloth 강의 = 거의 공백 |
| 3 | 전자책 / 노션 가이드 | "Unsloth 실전 가이드 300p" |
| 4 | **기업 출강 ★** | **단가 최고.** "사내 AI 파인튜닝 워크숍 1일" |
| 5 | 유료 커뮤니티 | 월 구독. 질의응답 + 최신 모델 소식 |
| 6 | GPU 선택 가이드 + 제휴 | 하드웨어/클라우드 제휴 |

**가장 잘 팔릴 주제 (희소성 기준)**
1. **AGPL·Llama 라이선스 — 상업화 전 반드시 알아야 할 것** ← 쓸 사람이 거의 없고 기업은 절실
2. **VRAM별 실측 가능/불가능표** ← 모두 궁금해하는데 아무도 정리 안 함
3. **데이터셋 만들기** ← 파인튜닝 실패 원인 90%가 데이터. 여기가 진짜 노하우
4. GRPO/강화학습으로 추론 모델 만들기 ← 고급 권위
5. 에이전트를 로컬 모델로 돌리기 ← 트렌드 정면

### 3순위 — 버티컬 SaaS / 제품 (고수익 · 고난도)

설계 원칙: **Core(Apache)만 쓰고 Studio는 피하기** ([11장 아키텍처](#11-react--php로-만들-수-있는가) 참조)

| 버티컬 | 제품 | 왜 로컬이어야 하나 |
|---|---|---|
| **의료** | 진료기록 요약, 의무기록 코딩 보조 | 환자정보 외부전송 불가 (법) |
| **법률** | 판례·계약서 검토, 법률문서 초안 | 수임 비밀 |
| **금융** | 내부 리포트 분석, 컴플라이언스 체크 | 규제·망분리 |
| **제조** | 설비 매뉴얼 Q&A, 불량 리포트 분석 | 공정 기밀 |
| **공공** | 민원 분류·답변 초안, 내부규정 검색 | 망분리 의무 |
| **게임** | NPC 대화 생성, 로컬라이징 | 비용 (API로는 단가 불가) |
| **CS** | 상담 요약·분류·답변 추천 | 고객정보 + 대량호출 비용 |
| **교육** | 학생별 맞춤 피드백 | 학생정보 보호 |

**특별히 강력한 아이디어 2개**

**① 파인튜닝 as a Service (셀프서비스)**
```
고객: CSV/PDF 업로드 → 모델 선택 → 버튼 클릭
결과: 전용 모델 + GGUF 다운로드 + API 엔드포인트
과금: 학습 1회당 + 월 호스팅
```
Unsloth Core(Apache)로 백엔드 구성, UI는 자체 제작. 기술 리스크는 "GPU 큐 관리".
⚠️ **Studio 코드는 절대 쓰지 말고 Core만 사용** (AGPL 회피).

**② 도구호출 전용 소형 모델 (기술적으로 가장 뾰족함)**
```
문제: 로컬 소형 모델은 Tool Calling을 자주 틀림 → 에이전트로 못 씀
해결: 1) Claude/GPT로 완벽한 도구호출 대화 2,000~10,000건 합성
      2) Unsloth로 4B 모델에 학습
      3) 결과: 우리 도메인 도구는 대형모델급으로 호출하는 초저가 모델
상품: 특정 도메인(ERP/CRM/이커머스) 전용 "에이전트 두뇌" 라이선스
```
재료가 이미 갖춰져 있음: `dataprep/synthetic.py`, `tool_call_parser.py`, `native_tool_tokens.py`
**2026년 현재 수요는 크고 공급은 거의 없는 영역.**

### 4순위 — 템플릿 & 디지털 자산

| 상품 | 설명 | 주의 |
|---|---|---|
| **데이터셋** | 한국어 도메인별 instruction 데이터 (법률/의료/CS/금융) | ⚠️ 저작권·개인정보 — 적법 수집 필수 |
| LoRA 어댑터 | "한국어 존댓말", "법률문체", "마케팅 카피" | 베이스 모델 라이선스 확인 |
| 학습 레시피 | 검증된 하이퍼파라미터 + 스크립트 번들 | 안전 |
| 평가셋 | 한국어 모델 성능 벤치마크 | 안전 |
| Docker 이미지 | 원클릭 배포 구성 | Studio 수정 시 AGPL ⚠️ |
| Agent Skills 팩 | `skills.py`가 읽는 스킬 번들 | 안전 |

> **데이터셋이 가장 저평가된 기회입니다.** 모델은 무료로 쏟아지는데
> **양질의 한국어 학습 데이터는 돈을 내고도 구하기 어렵습니다.**
> 단, **수집 적법성(저작권·개인정보보호법) 확보가 전제**입니다.

---

## 15. 리스크 체크리스트

| # | 리스크 | 대응 |
|---|---|---|
| 1 | **AGPL 전염** | 수익 제품은 **Core(Apache)만** 사용. Studio는 내부 도구로만 |
| 2 | **베이스 모델 라이선스** | Llama=MAU 7억+네이밍 의무 / Gemma=Google 약관 / **Qwen·Mistral=Apache(최적)** |
| 3 | **학습 데이터 저작권** | 크롤링 데이터 학습 모델 판매는 위험. 고객 데이터는 **계약서에 소유권 명시** |
| 4 | **개인정보** | 상담·의무기록 학습 시 **비식별화 필수.** 모델이 학습 데이터를 토해낼 수 있음 |
| 5 | **성능 보장 함정** | "GPT-4급" 약속 금지. **계약서에 평가 기준·합격선을 수치로 명시** |
| 6 | **버전 변동** | 릴리스가 매우 잦음. 납품 시 **버전 고정 + Docker 동결** |
| 7 | **GPU 조달** | 고객이 GPU를 안 사면 프로젝트 정지. 클라우드 GPU 대안 준비 |
| 8 | **보안 사고** | `--secure` 노출 + 도구 ON = **원격 코드 실행 위험.** 납품 시 `--disable-tools` 기본 + 보안 체크리스트 서면화 |
| 9 | **기술 상향 평준화** | 1년 뒤 로컬 모델이 더 쉬워짐 → 단순 설치 용역 가치 하락. **도메인 전문성·데이터 자산으로 이동** |

---

## 16. 90일 실행 로드맵

```
0~30일   · Unsloth 직접 설치 · 내 데이터로 학습 1회 완주
         · 유튜브 입문 영상 3편 (①②③)
         · "API 비용 절감 진단" 서비스 정의 + 1페이지 제안서

30~60일  · 영상 6편 누적 → 첫 문의 유입 시작
         · 지인/소기업 1곳 무료 진단 → 레퍼런스 확보 ★
         · "AGPL/라이선스" 영상 1편 (차별화 핵심)

60~90일  · 첫 유료 구축 프로젝트 (A 또는 B 패키지)
         · 온라인 강의 촬영 시작
         · 유지보수(D) 월구독 제안 → 현금흐름 전환

90일 이후 · 버티컬 1개 선택 → 제품화 착수 (Core만 사용)
         · 또는 "도구호출 전용 모델" R&D
```

### 요점 3개

1. **용역부터** 시작하세요. 제품은 돈 벌면서 만드는 겁니다.
2. **콘텐츠를 영업 깔때기로** 쓰세요. 유튜브가 영업사원 역할을 합니다.
3. **Studio는 내부 도구, Core는 상품.** 이 선만 지키면 라이선스 걱정이 없습니다.

---

## 참고 링크 모음

| 구분 | URL |
|---|---|
| 이 저장소 | <https://github.com/bmshin94/unsloth> |
| 원본 저장소 | <https://github.com/unslothai/unsloth> |
| 공식 문서 | <https://unsloth.ai/docs> |
| 다운로드 | <https://unsloth.ai/download> |
| 릴리스 | <https://github.com/unslothai/unsloth/releases> |
| 무료 노트북 | <https://github.com/unslothai/notebooks> |
| Studio Colab | <https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb> |
| 파인튜닝 가이드 | <https://unsloth.ai/docs/get-started/fine-tuning-llms-guide> |
| 강화학습(RL) 가이드 | <https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide> |
| 모델 카탈로그 | <https://unsloth.ai/docs/get-started/unsloth-model-catalog> |
| MCP 가이드 | <https://unsloth.ai/docs/basics/mcp> |
| Claude Code 연동 | <https://unsloth.ai/docs/basics/claude-code> |
| API 가이드 | <https://unsloth.ai/docs/basics/api> |
| Unsloth Start | <https://unsloth.ai/docs/integrations/unsloth-start> |
| Docker Hub (NVIDIA) | <https://hub.docker.com/r/unsloth/unsloth> |
| Docker Hub (AMD) | <https://hub.docker.com/r/unsloth/unsloth-rocm> |
| Discord | <https://discord.gg/unsloth> |
| Reddit | <https://reddit.com/r/unsloth> |
| X (Twitter) | <https://x.com/UnslothAI> |
| 블로그 | <https://unsloth.ai/blog> |

### 저장소 내 참고 문서

| 파일 | 내용 |
|---|---|
| `README.md` | 설치·기능·뉴스 |
| `CLAUDE.md` | 프로젝트 가이드 (한국어) |
| `studio/MCP.md` | **MCP 서버/클라이언트 설정** |
| `studio/NVLINK_P2P.md` | 멀티 GPU NVLink P2P |
| `studio/PDF_OCR.md` | PDF OCR 설정 |
| `studio/ROCM_RDNA2_APU.md` | AMD RDNA2 APU |
| `unsloth/registry/REGISTRY.md` | 모델 레지스트리 |
| `studio/backend/core/training/fsdp2_design_notes.md` | FSDP2 설계 노트 |
| `CONTRIBUTING.md` | 기여 가이드 |
| `LICENSE` / `COPYING` / `studio/LICENSE.AGPL-3.0` | 라이선스 전문 |

---

*이 문서는 Claude Code를 통한 저장소 전수조사 결과를 정리한 것입니다. (2026-10-07)*
