# RAG 시스템 아키텍처 가이드

## 📐 시스템 개요

이 문서는 Databricks 기반 RAG (Retrieval-Augmented Generation) 시스템의 전체 아키텍처와 구성 요소를 설명합니다.

## 🏗️ 아키텍처 다이어그램

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   사용자 쿼리    │───▶│   RAG 시스템      │───▶│   최종 답변      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │                RAG 파이프라인                           │
        │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
        │  │   문서 검색  │─▶│  컨텍스트   │─▶│  LLM 답변 생성  │ │
        │  │   (Vector   │  │   구성      │  │  (Foundation   │ │
        │  │   Search)   │  │             │  │   Models)      │ │
        │  └─────────────┘  └─────────────┘  └─────────────────┘ │
        └─────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │                 데이터 계층                             │
        │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
        │  │   PDF 문서  │─▶│ Delta Table │─▶│ Vector Search   │ │
        │  │    로드     │  │   (청크)    │  │    인덱스       │ │
        │  └─────────────┘  └─────────────┘  └─────────────────┘ │
        └─────────────────────────────────────────────────────────┘
```

## 🔧 핵심 구성 요소

### 1. 문서 처리 계층

#### **PDF 로더**
- **역할**: PDF 파일을 텍스트로 변환
- **기술**: LangChain PyPDFLoader
- **특징**: 페이지별 메타데이터 보존

#### **텍스트 청킹**
- **역할**: 긴 문서를 검색에 적합한 크기로 분할
- **기술**: RecursiveCharacterTextSplitter
- **설정**:
  - 청크 크기: 1,000 글자
  - 중복 영역: 200 글자 (컨텍스트 연속성 보장)

#### **메타데이터 관리**
```python
{
    "id": "chunk_0",
    "content": "실제 텍스트 내용",
    "source": "./data/pdf/document.pdf",
    "filename": "document.pdf",
    "page": 1,
    "chunk_index": 0,
    "chunk_size": 1000,
    "document_type": "pdf"
}
```

### 2. 저장 계층

#### **Delta Lake 테이블**
- **역할**: 문서 청크의 영구 저장
- **스키마**:
  ```sql
  CREATE TABLE rag_documents (
      id STRING,
      content STRING,
      source STRING,
      filename STRING,
      page INT,
      chunk_index INT,
      chunk_size INT,
      document_type STRING
  ) USING DELTA
  TBLPROPERTIES (
      'delta.enableChangeDataFeed' = 'true'
  )
  ```

#### **Change Data Feed**
- **목적**: Vector Search 인덱스 동기화
- **기능**: 데이터 변경 시 자동으로 벡터 인덱스 업데이트

### 3. 검색 계층

#### **Databricks Vector Search**
- **역할**: 의미 기반 문서 검색
- **임베딩 모델**: `databricks-bge-large-en`
- **벡터 차원**: 1,024차원
- **인덱스 타입**: HNSW (Hierarchical Navigable Small World)

#### **검색 프로세스**
1. 사용자 쿼리를 벡터로 변환
2. 코사인 유사도 계산
3. 상위 K개 관련 문서 반환
4. 메타데이터와 함께 결과 제공

### 4. 생성 계층

#### **Foundation Models**
지원되는 모델들:
- **Llama 3.1 405B**: 최고 성능
- **Llama 3.1 70B**: 균형잡힌 성능
- **DBRX**: Databricks 최적화 모델  
- **Mixtral 8x7B**: 효율적인 MoE 모델

#### **동적 엔드포인트 발견**
```python
def discover_available_llm_endpoints():
    candidate_endpoints = [
        "databricks-llama-3-1-405b-instruct",
        "databricks-llama-3-1-70b-instruct", 
        "databricks-dbrx-instruct",
        "databricks-mixtral-8x7b-instruct"
    ]
    # 각 엔드포인트 가용성 테스트
    # 첫 번째 응답하는 엔드포인트 선택
```

## 🔄 데이터 플로우

### 1. 문서 수집 단계
```
PDF 파일 → PyPDFLoader → 텍스트 추출 → 메타데이터 첨부
```

### 2. 전처리 단계  
```
긴 텍스트 → RecursiveCharacterTextSplitter → 청크 분할 → Delta 테이블 저장
```

### 3. 인덱싱 단계
```
Delta 테이블 → BGE-Large 임베딩 → 벡터 변환 → Vector Search 인덱스
```

### 4. 쿼리 처리 단계
```
사용자 쿼리 → 벡터 변환 → 유사도 검색 → 관련 문서 반환
```

### 5. 답변 생성 단계
```
관련 문서 + 사용자 쿼리 → Foundation Model → 컨텍스트 기반 답변
```

## 🛡️ 환경 호환성

### **Native Databricks 환경**
- **특징**: Databricks 클러스터에서 직접 실행
- **장점**: 모든 기능 완전 지원
- **사용 사례**: 프로덕션 배포

### **VS Code Extension 환경**  
- **특징**: 로컬 VS Code에서 Databricks 클러스터 연결
- **장점**: 로컬 개발 + 클라우드 리소스 활용
- **사용 사례**: 개발 및 테스트

### **환경 자동 감지**
```python
is_databricks_native = "DATABRICKS_RUNTIME_VERSION" in os.environ
is_vscode_databricks = # Spark 세션 및 Databricks 기능 테스트
```

## 🔧 설정 관리

### **Unity Catalog vs Hive Metastore**
- **Unity Catalog**: `catalog.schema.table` 형식
- **Hive Metastore**: `schema.table` 형식
- **자동 감지**: 현재 환경에 따라 동적 설정

### **리소스 명명 규칙**
```python
# Unity Catalog 환경
index_name = f"{catalog}.{schema}.rag_docs_index"
source_table_name = f"{catalog}.{schema}.rag_documents"

# Hive Metastore 환경  
index_name = f"{schema}.rag_docs_index"
source_table_name = f"{schema}.rag_documents"
```

## ⚡ 성능 최적화

### **Vector Search 최적화**
- **HNSW 인덱스**: 고속 근사 검색
- **배치 처리**: 대량 문서 효율적 처리
- **실시간 동기화**: Change Data Feed 활용

### **LLM 최적화**
- **엔드포인트 풀링**: 여러 모델 간 로드 밸런싱
- **캐싱**: 반복 쿼리 결과 캐시
- **스트리밍**: 실시간 응답 생성

### **메모리 관리**
- **청크 크기 조절**: 메모리 사용량 최적화
- **배치 크기 제한**: OOM 방지
- **가비지 컬렉션**: 주기적 메모리 정리

## 🔒 보안 및 권한

### **필요 권한**
- **클러스터 액세스**: DBR 13.0+ 런타임
- **Vector Search**: 인덱스 생성/읽기 권한
- **Model Serving**: Foundation Model API 사용 권한
- **Unity Catalog**: 카탈로그/스키마 접근 권한

### **데이터 보안**
- **전송 중 암호화**: HTTPS/TLS
- **저장 시 암호화**: Delta Lake 기본 암호화
- **액세스 제어**: Unity Catalog 기반 권한 관리

## 📊 모니터링 및 관찰성

### **시스템 메트릭**
- **검색 지연시간**: Vector Search 응답 시간
- **생성 지연시간**: LLM 응답 시간
- **처리량**: 초당 쿼리 수
- **정확도**: 검색 결과 관련성

### **로깅**
- **요청 로그**: 모든 사용자 쿼리 기록
- **오류 로그**: 시스템 오류 및 예외 추적
- **성능 로그**: 응답 시간 및 리소스 사용량

### **알림**
- **시스템 장애**: 구성 요소 실패 알림
- **성능 저하**: 응답 시간 임계값 초과
- **리소스 부족**: 클러스터 리소스 부족 경고

---

이 아키텍처는 확장 가능하고 유지보수가 용이한 엔터프라이즈급 RAG 시스템을 제공합니다. 각 구성 요소는 독립적으로 확장 및 최적화할 수 있으며, 다양한 배포 환경을 지원합니다.
