# Databricks RAG 시스템 설정 가이드

## 📋 개요

이 가이드는 Databricks RAG 시스템을 처음부터 설정하는 방법을 단계별로 안내합니다. VS Code Databricks Extension을 활용한 현대적인 개발 환경 구축에 중점을 둡니다.

## 🎯 전제 조건

### 필수 요구사항

#### 1. Databricks 계정 및 워크스페이스
- **Databricks 워크스페이스**: 유료 또는 14일 평가판 계정
- **클러스터**: DBR (Databricks Runtime) 13.0 이상
- **권한**: 클러스터 생성/관리, Vector Search, Model Serving 권한

#### 2. 로컬 개발 환경
- **VS Code**: 최신 버전 (1.80+)
- **Python**: 3.8 이상
- **Git**: 버전 관리용 (선택사항)

#### 3. 시스템 요구사항
- **RAM**: 최소 8GB, 권장 16GB+
- **저장공간**: 최소 5GB 여유 공간
- **네트워크**: 안정적인 인터넷 연결

---

## 🚀 1단계: VS Code 및 확장 설치

### VS Code 설치

```bash
# Windows - Chocolatey 사용 시
choco install vscode

# macOS - Homebrew 사용 시  
brew install --cask visual-studio-code

# Linux - Snap 사용 시
sudo snap install code --classic
```

### Databricks Extension 설치

#### 방법 1: VS Code 마켓플레이스
1. VS Code 실행
2. 좌측 확장(Extensions) 탭 클릭 (Ctrl+Shift+X)
3. "Databricks" 검색
4. **Microsoft** 제작 "Databricks" 확장 설치

#### 방법 2: 명령줄
```bash
code --install-extension databricks.databricks
```

#### 방법 3: 수동 설치
1. [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=databricks.databricks) 방문
2. "Install" 버튼 클릭
3. VS Code에서 설치 완료

### 설치 확인

확장이 정상적으로 설치되었는지 확인:

1. **Ctrl+Shift+P** 실행
2. "Databricks" 입력
3. Databricks 관련 명령어들이 표시되면 성공

---

## 🔗 2단계: Databricks 워크스페이스 연결

### Personal Access Token 생성

#### Databricks 워크스페이스에서:

1. **Databricks 워크스페이스** 로그인
2. 우상단 **사용자 아이콘** 클릭
3. **"User Settings"** 선택
4. **"Developer"** 탭 → **"Access tokens"**
5. **"Generate new token"** 클릭
6. **토큰 이름** 입력 (예: "VS Code Development")
7. **만료 기간** 설정 (90일 권장)
8. **생성된 토큰 복사** (⚠️ 한 번만 표시됩니다!)

### VS Code에서 워크스페이스 연결

#### 방법 1: 명령 팔레트 사용
1. **Ctrl+Shift+P** 실행
2. **"Databricks: Configure Workspace"** 선택
3. **Databricks URL** 입력
   ```
   https://your-workspace.cloud.databricks.com
   ```
4. **Personal Access Token** 붙여넣기

#### 방법 2: 설정 파일 직접 편집
VS Code에서 `settings.json` 열기:
```json
{
    "databricks.workspaceUri": "https://your-workspace.cloud.databricks.com",
    "databricks.authType": "pat",
    "databricks.personalAccessToken": "your-token-here"
}
```

### 연결 확인

1. **VS Code 좌측 패널**에 **Databricks 아이콘** 표시 확인
2. **클러스터 목록**이 표시되면 연결 성공
3. 오류 발생 시 **문제 해결** 섹션 참조

---

## ⚙️ 3단계: 클러스터 설정 및 연결

### 클러스터 요구사항

#### 권장 클러스터 설정:
- **Databricks Runtime**: 13.3 LTS 이상
- **Python Version**: 3.9+
- **Node Type**: 
  - **Driver**: i3.xlarge (4 cores, 30.5 GB)
  - **Workers**: i3.large (2 cores, 15.25 GB) × 2개 이상
- **Auto Termination**: 120분

### 클러스터 생성 (Databricks UI)

1. **Databricks 워크스페이스** → **"Compute"** 탭
2. **"Create Cluster"** 클릭
3. **클러스터 설정**:
   ```
   Cluster Name: rag-development
   Cluster Mode: Standard
   Databricks Runtime Version: 13.3 LTS (Scala 2.12, Spark 3.4.1)
   Node Type: i3.xlarge (Driver), i3.large (Workers)
   Workers: 2 (Min: 1, Max: 4)
   ```
4. **"Create Cluster"** 클릭

### VS Code에서 클러스터 연결

1. **VS Code 좌측 Databricks 패널** 확인
2. **사용 가능한 클러스터 목록**에서 클러스터 선택
3. **"Connect"** 버튼 클릭
4. **상태바**에 클러스터명이 표시되면 연결 완료

### 라이브러리 설치 (클러스터)

RAG 시스템에 필요한 라이브러리들을 클러스터에 설치:

#### Databricks UI에서:
1. **클러스터 페이지** → **"Libraries"** 탭
2. **"Install New"** → **"PyPI"**
3. 다음 패키지들을 하나씩 설치:
   ```
   langchain
   langchain-community
   databricks-vectorsearch
   pypdf
   ```

#### 또는 init script 사용:
```bash
#!/bin/bash
pip install langchain langchain-community databricks-vectorsearch pypdf
```

---

## 📁 4단계: 프로젝트 폴더 구성

### 권장 폴더 구조

```
databricks_rag/
├── rag_app_on_vm.ipynb           # 메인 RAG 노트북
├── text_to_sql_prompt_template.ipynb  # Text-to-SQL 모듈
├── requirements.txt              # Python 의존성
├── databricks.yml               # Databricks 설정 (선택)
├── data/                        # 데이터 폴더
│   ├── pdf/                     # PDF 문서들
│   │   └── *.pdf               # 분석할 PDF 파일들
│   └── processed_chunks.csv     # 처리된 청크 (자동 생성)
├── docs/                        # 문서화
│   ├── README.md
│   ├── setup-guide.md
│   └── *.md
└── .vscode/                     # VS Code 설정
    └── settings.json
```

### 프로젝트 초기화

#### 1. 작업 폴더 생성
```bash
mkdir -p ~/Projects/databricks_rag
cd ~/Projects/databricks_rag
```

#### 2. Git 초기화 (선택사항)
```bash
git init
echo "*.log" > .gitignore
echo "__pycache__/" >> .gitignore
echo ".env" >> .gitignore
```

#### 3. 데이터 폴더 생성
```bash
mkdir -p data/pdf
mkdir -p docs
```

#### 4. VS Code에서 폴더 열기
```bash
code .
```

### 노트북 파일 다운로드

필요한 노트북 파일들을 프로젝트 폴더에 배치:

1. **rag_app_on_vm.ipynb**: 메인 RAG 시스템
2. **text_to_sql_prompt_template.ipynb**: Text-to-SQL 유틸리티

---

## ✅ 5단계: 설정 검증

### 기본 연결 테스트

VS Code에서 새 노트북 생성 후 테스트:

```python
# 셀 1: 환경 확인
import os
print("Databricks Runtime:", os.environ.get('DATABRICKS_RUNTIME_VERSION', 'Not Found'))

# 셀 2: Spark 연결 테스트  
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("ConnectionTest").getOrCreate()
print("Spark Version:", spark.version)

# 셀 3: 간단한 SQL 테스트
result = spark.sql("SELECT 1 as test").collect()
print("SQL Test Result:", result[0]['test'])

# 셀 4: Databricks 기능 테스트
try:
    catalog = spark.sql("SELECT current_catalog()").collect()[0][0]
    schema = spark.sql("SELECT current_schema()").collect()[0][0]
    print(f"Current Catalog: {catalog}")
    print(f"Current Schema: {schema}")
except Exception as e:
    print(f"Catalog/Schema test failed: {e}")
```

### Vector Search 권한 확인

```python
# Vector Search 접근 권한 테스트
try:
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(disable_notice=True)
    endpoints = vsc.list_endpoints()
    print("✅ Vector Search 권한 있음")
    print(f"사용 가능한 엔드포인트: {len(endpoints)}개")
except Exception as e:
    print(f"❌ Vector Search 권한 없음: {e}")
```

### Model Serving 권한 확인

```python
# Foundation Model 접근 권한 테스트
try:
    from langchain_community.embeddings import DatabricksEmbeddings
    embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
    test_embedding = embedding_model.embed_query("test")
    
    if test_embedding and len(test_embedding) > 0:
        print("✅ Model Serving 권한 있음")
        print(f"임베딩 차원: {len(test_embedding)}")
    else:
        print("❌ 임베딩 응답 없음")
        
except Exception as e:
    print(f"❌ Model Serving 권한 없음: {e}")
```

---

## 🚨 문제 해결

### 일반적인 문제들

#### 1. "Authentication failed" 오류

**원인**: Personal Access Token 문제
**해결책**:
- 토큰 재생성 및 교체
- 토큰 만료 확인
- 워크스페이스 URL 정확성 확인

```bash
# 설정 초기화
# Ctrl+Shift+P → "Databricks: Configure Workspace"
```

#### 2. 클러스터 연결 실패

**원인**: 클러스터 상태 또는 권한 문제
**해결책**:
- Databricks UI에서 클러스터 상태 확인
- 클러스터 재시작
- 권한 설정 확인

#### 3. "Spark session not found" 오류

**원인**: 클러스터 연결 문제
**해결책**:
```python
# 수동으로 Spark 세션 재생성
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("ManualRestart").getOrCreate()
```

#### 4. 라이브러리 import 오류

**원인**: 필요한 패키지가 클러스터에 설치되지 않음
**해결책**:
- 클러스터 라이브러리 탭에서 패키지 설치 확인
- 클러스터 재시작 후 재시도

### 네트워크 관련 문제

#### 회사 방화벽 환경

```bash
# 프록시 설정이 필요한 경우
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

#### VPN 연결 문제

- VPN 연결 확인
- DNS 설정 확인
- 방화벽 예외 설정

### 성능 최적화

#### 클러스터 리소스 부족

**증상**: 메모리 부족, 느린 응답
**해결책**:
- 워커 노드 수 증가
- 노드 타입 업그레이드
- Auto Scaling 활성화

#### 네트워크 지연

**증상**: 느린 연결, 타임아웃
**해결책**:
- 지역별 워크스페이스 선택
- 클러스터 지역 확인
- 네트워크 대역폭 확인

---

## 📚 추가 리소스

### 공식 문서
- [Databricks VS Code Extension 가이드](https://docs.databricks.com/dev-tools/vscode-ext/index.html)
- [Vector Search 문서](https://docs.databricks.com/generative-ai/vector-search.html)
- [Foundation Model APIs](https://docs.databricks.com/machine-learning/foundation-models/)

### 커뮤니티
- [Databricks Community Forum](https://community.databricks.com/)
- [Stack Overflow - Databricks 태그](https://stackoverflow.com/questions/tagged/databricks)

### 학습 자료
- [Databricks Academy](https://academy.databricks.com/)
- [Generative AI 무료 코스](https://www.databricks.com/learn/training/generative-ai)

---

## ✅ 설정 완료 체크리스트

설정이 완료되면 다음 항목들을 확인하세요:

- [ ] VS Code 및 Databricks Extension 설치 완료
- [ ] Personal Access Token 생성 및 설정
- [ ] 워크스페이스 연결 성공
- [ ] 클러스터 생성 및 연결 완료
- [ ] 필요한 라이브러리 설치 완료
- [ ] 프로젝트 폴더 구조 생성
- [ ] 기본 연결 테스트 성공
- [ ] Vector Search 권한 확인
- [ ] Model Serving 권한 확인
- [ ] 샘플 노트북 실행 가능

모든 항목이 체크되면 **RAG 시스템 구축**을 시작할 준비가 완료된 것입니다! 🎉

다음 단계에서는 메인 노트북 `rag_app_on_vm.ipynb`를 실행하여 실제 RAG 시스템을 구축해보세요.
