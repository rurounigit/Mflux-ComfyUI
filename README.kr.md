<h1 align="center">Mflux-ComfyUI 2.1.0</h1>

<p align="center">
    <strong>mflux 0.13.1용 ComfyUI 노드 (Apple Silicon/MLX)</strong><br/>
    <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

## 개요

이 포크는 **mflux 0.13.1**을 사용하도록 노드를 업그레이드하면서 ComfyUI 워크플로우 호환성을 유지합니다. mflux 0.13.x의 새로운 통합 아키텍처를 활용하여 표준 FLUX 생성뿐만 아니라 Fill(채우기), Depth(깊이), Redux(재구성), Z-Image Turbo와 같은 특수 변형 모델도 지원합니다.

- **백엔드**: mflux 0.13.1 (macOS + Apple Silicon 필요).
- **그래프 호환성**: 레거시 입력을 내부적으로 마이그레이션하므로 기존 그래프가 계속 작동합니다.
- **통합 로딩**: 로컬 경로, HuggingFace 리포지토리 ID, 사전 정의된 별칭(예: `dev`, `schnell`)을 원활하게 처리합니다.

## mflux 0.13.1의 새로운 기능
이 버전은 중요한 백엔드 개선 사항을 제공합니다:
- **Z-Image Turbo 지원**: 속도에 최적화된 고속 증류(distilled) Z-Image 변형(6B 파라미터)을 지원합니다.
- **FIBO VLM 양자화**: 양자화된(3/4/5/6/8-bit) FIBO VLM 명령(`inspire`/`refine`)을 지원합니다.
- **통합 아키텍처**: 모델, LoRA 및 토크나이저의 해석 능력이 향상되었습니다.

## 주요 기능

- **핵심 생성**: 하나의 노드로 빠른 text2img 및 img2img 처리 (`QuickMfluxNode`).
- **Z-Image Turbo**: 새로운 고속 모델을 위한 전용 노드 (`MFlux Z-Image Turbo`).
- **FLUX 도구 지원**: **Fill** (인페인팅), **Depth** (구조 가이드), **Redux** (이미지 변형) 전용 노드 지원.
- **ControlNet**: Canny 미리보기 및 최선의 노력(best‑effort) 컨디셔닝; **Upscaler** ControlNet 지원 포함.
- **LoRA 지원**: 통합 LoRA 파이프라인 (LoRA 적용 시 양자화는 8 또는 None이어야 함).
- **양자화**: 메모리 효율성을 위한 다양한 옵션 (None, 3, 4, 5, 6, 8-bit).
- **메타데이터**: mflux CLI 도구와 호환되는 전체 생성 메타데이터(PNG + JSON) 저장.

## 설치

### ComfyUI-Manager 사용 (권장)
- “Mflux-ComfyUI”를 검색하여 설치합니다.

### 수동 설치
1. custom nodes 디렉토리로 이동합니다:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   ```
2. 저장소를 복제합니다:
   ```bash
   git clone https://github.com/rurounigit/Mflux-ComfyUI.git
   ```
3. ComfyUI 가상 환경을 활성화하고 의존성을 설치합니다:
   ```bash
   # 표준 venv 예시
   source /path/to/ComfyUI/venv/bin/activate

   pip install --upgrade pip wheel setuptools
   pip install 'mlx>=0.27.0' 'huggingface_hub>=0.26.0'
   pip install 'mflux==0.13.1'
   ```
4. ComfyUI를 재시작합니다.

**참고**: `mflux 0.13.1`은 `mlx >= 0.27.0`이 필요합니다. 구버전을 사용 중이라면 업그레이드해주세요.

## 노드 설명

### MFlux/Air (표준)
- **QuickMfluxNode**: 표준 FLUX txt2img, img2img, LoRA, ControlNet을 위한 올인원 노드.
- **MFlux Z-Image Turbo**: Z-Image 생성 전용 노드 (최적 기본값: 9 steps, no guidance).
- **Mflux Models Loader**: `models/Mflux`에서 로컬 모델 선택.
- **Mflux Models Downloader**: HuggingFace에서 양자화 또는 전체 모델 직접 다운로드.
- **Mflux Custom Models**: 커스텀 양자화 변형 모델 구성 및 저장.

### MFlux/Pro (고급)
- **Mflux Fill**: 인페인팅 및 아웃페인팅을 위한 FLUX.1-Fill 지원 (마스크 필요).
- **Mflux Depth**: 구조 가이드 생성을 위한 FLUX.1-Depth 지원.
- **Mflux Redux**: 이미지 스타일/구조 혼합을 위한 FLUX.1-Redux 지원.
- **Mflux Upscale**: Flux ControlNet Upscaler를 사용한 이미지 업스케일링.
- **Mflux Img2Img / Loras / ControlNet**: 커스텀 파이프라인 구축을 위한 모듈식 로더.

## 사용 팁

- **Z-Image Turbo**: 전용 노드를 사용하세요. 기본값은 **9 steps** 및 **0 guidance**입니다 (이 모델에 필수).
- **LoRA 호환성**: 현재 LoRA를 사용하려면 기본 모델을 `quantize=8` (또는 None)로 로드해야 합니다.
- **해상도**: 너비와 높이는 16의 배수여야 합니다 (필요 시 자동 조정됨).
- **Guidance (가이던스)**:
  - `dev` 모델은 guidance를 따릅니다 (기본값 약 3.5).
  - `schnell` 모델은 guidance를 무시합니다 (그대로 두어도 무방).
- **경로**:
  - 양자화 모델: `ComfyUI/models/Mflux`
  - LoRA: `ComfyUI/models/loras` (정리를 위해 `Mflux` 하위 폴더 생성 권장).

## 워크플로우

`workflows` 폴더에서 JSON 예제를 확인하세요:
- `Mflux text2img.json`
- `Mflux img2img.json`
- `Mflux ControlNet.json`
- `Mflux Fill/Redux/Depth` 예제 (가능한 경우)

ComfyUI에서 노드가 빨간색으로 표시되면 Manager의 "Install Missing Custom Nodes" 기능을 사용하세요.

## 감사

- **mflux**: [@filipstrand](https://github.com/filipstrand) 및 기여자분들.
- **raysers**: 초기 ComfyUI 통합 개념 제안.
- 일부 코드 구조는 **MFLUX-WEBUI**에서 영감을 받았습니다.

## 라이선스

MIT