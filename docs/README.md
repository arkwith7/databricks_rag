# Databricks RAG Application 문서

## 🚀 프로젝트 개요

이 프로젝트는 **Databricks 플랫폼**에서 구축되는 엔터프라이즈급 **RAG (Retrieval-Augmented Generation)** 시스템입니다. VS Code Databricks Extension을 활용하여 로컬 개발 환경에서 클라우드 리소스를 효율적으로 활용할 수 있습니다.

### ✨ 주요 특징

- **🔍 지능형 문서 검색**: Databricks Vector Search를 활용한 의미 기반 검색
- **🤖 자연어 답변 생성**: Foundation Model APIs를 통한 고품질 답변 생성
- **📊 완전한 메타데이터 추적**: 답변의 소스 문서 정보 완전 추적
- **🌐 환경 호환성**: Native Databricks와 VS Code Extension 모두 지원
- **⚡ 동적 리소스 발견**: 사용 가능한 LLM 모델 자동 탐지 및 연결

### 🎯 핵심 기능

- ✅ **PDF 문서 처리**: 자동 로드, 청킹, 메타데이터 관리
- ✅ **Vector Search 인덱스**: 실시간 동기화 및 고성능 검색
- ✅ **Foundation Models 통합**: Llama, DBRX, Mixtral 등 다양한 모델 지원
- ✅ **견고한 오류 처리**: 포괄적인 예외 처리 및 복구 시스템
- ✅ **확장 가능한 아키텍처**: 프로덕션 환경 배포 준비

---

## 📚 문서 구조

현재 문서는 다음과 같이 구성되어 있습니다:

### 🚀 시작하기
- **[README.md](README.md)** (이 파일): 프로젝트 개요 및 전체 가이드
- **[setup-guide.md](setup-guide.md)**: 상세한 환경 설정 가이드

### 🏗️ 시스템 이해하기
- **[architecture-guide.md](architecture-guide.md)**: 시스템 아키텍처 및 구성요소 설명
- **[implementation-guide.md](implementation-guide.md)**: 단계별 구현 과정

### 📖 사용하기
- **[usage-guide.md](usage-guide.md)**: 실제 사용법 및 예제
- **[api-reference.md](api-reference.md)**: API 레퍼런스

### 🔧 문제 해결하기
- **[troubleshooting-guide.md](troubleshooting-guide.md)**: 문제 해결 방법

### 📋 추가 계획
다음 문서들이 추후 추가될 예정입니다:
- **quick-start.md**: 빠른 시작 가이드
- **advanced-usage.md**: 고급 사용법 및 최적화
- **deployment-guide.md**: 프로덕션 배포 가이드

---

## ⚡ 빠른 시작

### 1. 전제 조건 확인

- **Databricks 워크스페이스**: 유료 또는 평가판 계정
- **VS Code**: 최신 버전 + Databricks Extension
- **클러스터**: DBR 13.0+ 런타임

### 2. 기본 설정

```bash
# VS Code에서 Databricks Extension 설치
code --install-extension databricks.databricks

# 워크스페이스 연결
# Ctrl+Shift+P → "Databricks: Configure Workspace"
```

### 3. 노트북 실행

1. **VS Code에서 `rag_app_on_vm.ipynb` 열기**
2. **클러스터 연결 확인**
3. **셀을 순서대로 실행**

### 4. 기본 사용법

```python
# 질문하기
response = qa_chain.invoke({"query": "AI 에이전트란 무엇인가요?"})
print(response['result'])

# 소스 정보 포함 답변
enhanced_response = enhance_rag_response(response)
formatted_sources = format_sources_for_display(enhanced_response['enhanced_sources'])
print(formatted_sources)
```

---

## 🛠️ 주요 구성 요소

### 📊 데이터 계층
- **PDF 로더**: LangChain PyPDFLoader
- **텍스트 청킹**: RecursiveCharacterTextSplitter
- **Delta 테이블**: Change Data Feed 활성화

### 🔍 검색 계층
- **Vector Search**: Databricks Vector Search
- **임베딩 모델**: databricks-bge-large-en
- **인덱스 타입**: HNSW (고성능 근사 검색)

### 🤖 생성 계층
- **Foundation Models**: Llama 3.1, DBRX, Mixtral
- **동적 발견**: 사용 가능한 엔드포인트 자동 탐지
- **RAG 체인**: LangChain RetrievalQA

---

## 📋 요구사항

### 기술 요구사항

#### Python 패키지
```
databricks-sdk
databricks-vectorsearch
databricks-connect
langchain
langchain-community
pandas
pyspark
PyPDF2
```

#### Databricks 권한
- **클러스터 접근**: DBR 13.0+ 런타임 사용 권한
- **Vector Search**: 인덱스 생성/읽기 권한
- **Model Serving**: Foundation Model API 사용 권한
- **Unity Catalog**: 카탈로그/스키마 접근 권한 (해당 시)

### 시스템 요구사항
- **메모리**: 최소 8GB RAM (권장: 16GB+)
- **저장공간**: 최소 5GB 여유 공간
- **네트워크**: Databricks 워크스페이스 접근

---

## 🚨 중요 참고사항

### 환경별 차이점

#### Native Databricks 환경
- 모든 기능 완전 지원
- 최적의 성능
- 프로덕션 배포에 적합

#### VS Code Extension 환경
- 로컬 개발 + 클라우드 리소스 활용
- 일부 기능 제한 가능
- 개발 및 테스트에 최적

### 알려진 제한사항

1. **Vector Search 메타데이터**: 일부 커스텀 컬럼은 인덱스에서 직접 사용 불가
2. **Foundation Model 가용성**: 워크스페이스별로 사용 가능한 모델이 다를 수 있음
3. **메모리 사용량**: 대용량 PDF 처리 시 클러스터 리소스 충분히 확보 필요

---

## 🔗 관련 리소스

### Databricks 공식 문서
- [Vector Search 가이드](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-models/index.html)
- [VS Code Extension](https://docs.databricks.com/en/dev-tools/vscode-ext/index.html)

### LangChain 문서
- [Databricks 통합](https://python.langchain.com/docs/integrations/platforms/databricks/)
- [RAG 튜토리얼](https://python.langchain.com/docs/use_cases/question_answering/)

### 커뮤니티
- [Databricks Community](https://community.databricks.com/)
- [LangChain Community](https://github.com/langchain-ai/langchain)

---

## 📞 지원 및 기여

### 문제 신고
문제가 발생했을 때는 다음 정보와 함께 신고해주세요:

1. **환경 정보**: Native vs VS Code Extension
2. **오류 메시지**: 전체 스택 트레이스
3. **재현 단계**: 문제 발생까지의 단계
4. **진단 정보**: `collect_diagnostic_info()` 결과

### 기여 방법
1. **문제 식별**: 버그 리포트 또는 기능 제안
2. **문서 개선**: 사용자 경험 향상을 위한 문서 업데이트
3. **코드 최적화**: 성능 향상 또는 새로운 기능 추가

---

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다. 상용 사용 시에는 관련 라이선스를 확인해주세요.

### 사용된 오픈소스 라이선스
- **LangChain**: MIT License
- **Databricks SDK**: Apache 2.0 License
- **pandas**: BSD License

---

## 🔄 버전 히스토리

### v1.0.0 (현재)
- ✅ 기본 RAG 시스템 구현
- ✅ Vector Search 통합
- ✅ Foundation Models 지원
- ✅ 메타데이터 관리 시스템
- ✅ VS Code Extension 호환성

### 향후 계획
- 🔄 멀티모달 지원 (이미지, 테이블)
- 🔄 실시간 스트리밍 응답
- 🔄 고급 검색 필터링
- 🔄 사용자 피드백 시스템

---

**🎉 Databricks RAG 시스템으로 지능형 문서 검색의 새로운 차원을 경험해보세요!**

더 자세한 정보는 각 가이드 문서를 참조하시거나, 노트북을 직접 실행해보시기 바랍니다.
