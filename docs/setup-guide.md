# Databricks Text-to-SQL RAG 시스템 설치 가이드

이 가이드는 Databricks Text-to-SQL RAG 시스템을 설정하고 실행하는 단계별 방법을 설명합니다.

## 🎯 시스템 요구사항

### 필수 환경

- **Databricks Workspace** (Community Edition 또는 그 이상)
- **Python 3.8+** 환경
- **Spark 3.4+** (Databricks 런타임에 포함)
- **Databricks Runtime 13.0+** 권장

### 권한 및 액세스

- Databricks Foundation Models 액세스 권한
- Delta Lake 테이블 생성 권한
- 클러스터 생성 및 관리 권한

## 📋 단계별 설치 가이드

### 1단계: Databricks 환경 준비

#### 1.1 Databricks Workspace 접속

```bash
# Databricks CLI 설치 (로컬 환경)
pip install databricks-cli

# 인증 설정
databricks configure --token
```

#### 1.2 클러스터 생성

- **Runtime**: DBR 13.3 LTS ML 이상 권장
- **Node Type**: Standard_DS3_v2 이상 (최소 14GB RAM)
- **Workers**: 1-2개 (테스트용)

#### 1.3 Foundation Models 활성화

```python
# Databricks 워크스페이스에서 확인
# Settings → Admin Console → Feature enablement → Foundation Models
```

### 2단계: 프로젝트 설정

#### 2.1 저장소 클론

```bash
# GitHub에서 클론 (로컬 환경)
git clone <repository-url>
cd databricks_rag

# 또는 Databricks Repos 사용
# Workspace → Repos → Add Repo → GitHub URL
```

#### 2.2 필수 라이브러리 설치

```python
# Databricks 노트북에서 실행
%pip install -r requirements.txt
dbutils.library.restartPython()
```

### 3단계: 데이터 구축 및 환경 설정

#### 3.1 첫 번째 노트북 실행

```python
# 01_databricks_setup_northwind.ipynb 열기 및 실행
# 모든 셀을 순서대로 실행
```

**수행되는 작업:**
- ✅ Spark 세션 초기화
- ✅ Northwind 데이터베이스 스키마 생성
- ✅ 8개 테이블 구축 (customers, products, orders 등)
- ✅ 샘플 데이터 로드 및 검증
- ✅ 기본 SQL 테스트

#### 3.2 환경 검증

```python
# 노트북 마지막 셀에서 확인
print("✅ 모든 설정이 완료되었습니다!")
```

### 4단계: LangChain Agent 구현

#### 4.1 두 번째 노트북 실행

```python
# 02_langchain_agent_text_to_sql.ipynb 열기 및 실행
```

**구현되는 기능:**
- 🤖 LangChain Agent 초기화
- 🔧 Function Tools 구현 (4개)
- 🌐 Databricks Foundation Models 연동
- 🧪 테스트 및 데모 시스템

#### 4.2 모델 연결 확인

```python
# 노트북에서 모델 상태 확인
if model_manager.is_available:
    print("✅ Foundation Models 연결 성공")
else:
    print("❌ 모델 연결 실패")
```

### 5단계: 시스템 테스트

#### 5.1 자동 테스트 실행

```python
# 기본 테스트 스위트
test_agent_with_examples()
```

#### 5.2 대화형 데모

```python
# 실시간 질의응답 테스트
interactive_query_demo()
```

#### 5.3 고급 기능 테스트

```python
# 성능 분석 및 최적화 기능
demonstrate_advanced_features()
```

## 🔧 고급 설정

### 대안 AI 모델 설정

#### OpenAI API 사용 (Databricks Models 대신)

```python
# 환경 변수 설정
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

# 노트북에서 자동으로 대안 모델 사용
```

#### 로컬 모델 사용

```python
# Hugging Face Transformers 등 사용 가능
# 상세한 설정은 implementation-guide.md 참조
```

### 성능 최적화

#### 클러스터 설정 조정

- **Driver**: Standard_DS4_v2 (28GB RAM)
- **Workers**: 2-4개 (병렬 처리용)
- **Auto-scaling**: Enable

#### 캐싱 활성화

```python
# Spark SQL 결과 캐싱
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
```

## 🚨 문제 해결

### 자주 발생하는 문제들

#### 1. Foundation Models 연결 실패

```python
# 해결방법:
# 1. Workspace에서 Foundation Models 활성화 확인
# 2. 클러스터 재시작
# 3. OpenAI API 대안 사용
```

#### 2. 메모리 부족 오류

```python
# 해결방법:
# 1. 클러스터 노드 타입 업그레이드
# 2. 데이터 샘플링 크기 조정
# 3. 쿼리 LIMIT 추가
```

#### 3. 라이브러리 설치 실패

```python
# 해결방법:
%pip install --upgrade pip
%pip install -r requirements.txt --force-reinstall
dbutils.library.restartPython()
```

#### 4. SQL 실행 권한 오류

```python
# 해결방법:
# 1. Workspace 관리자에게 권한 요청
# 2. Unity Catalog 설정 확인
# 3. 클러스터 액세스 모드 확인
```

### 로그 및 디버깅

#### 로그 레벨 설정

```python
import logging
logging.basicConfig(level=logging.INFO)

# Spark 로그 레벨 조정
spark.sparkContext.setLogLevel("WARN")
```

#### 상세 오류 추적

```python
# 노트북에서 상세 오류 정보 활성화
import traceback

try:
    # 문제가 되는 코드
    pass
except Exception as e:
    print(f"오류: {str(e)}")
    traceback.print_exc()
```

## 📊 검증 체크리스트

설치가 완료되면 다음 항목들을 확인하세요:

### 환경 검증

- [ ] Databricks 클러스터 정상 실행
- [ ] Python 라이브러리 모두 설치됨
- [ ] Spark 세션 정상 초기화

### 데이터 검증

- [ ] northwind 데이터베이스 생성됨
- [ ] 8개 테이블 모두 생성됨 (customers, products, orders 등)
- [ ] 샘플 데이터 정상 로드됨
- [ ] 기본 SQL 쿼리 실행 가능

### AI 모델 검증

- [ ] Databricks Foundation Models 연결됨
- [ ] LLM 응답 테스트 통과
- [ ] 임베딩 모델 초기화됨

### Agent 기능 검증

- [ ] LangChain Agent 초기화됨
- [ ] 4개 Function Tools 모두 동작
- [ ] 자연어 → SQL 변환 성공
- [ ] SQL 실행 및 결과 분석 완료

### 테스트 검증

- [ ] 기본 테스트 케이스 통과
- [ ] 대화형 데모 정상 동작
- [ ] 고급 기능 테스트 완료

## � 다음 단계

설치가 완료되면:

1. **📚 사용법 학습**: [usage-guide.md](usage-guide.md) 참조
2. **🏗️ 아키텍처 이해**: [architecture-guide.md](architecture-guide.md) 참조
3. **💻 코드 분석**: [implementation-guide.md](implementation-guide.md) 참조
4. **🚀 프로덕션 배포**: 웹 API 및 UI 개발 고려

## 📞 지원 및 문의

설치 중 문제가 발생하면:

- 📖 **문서 참조**: [troubleshooting-guide.md](troubleshooting-guide.md)
- 🐛 **이슈 리포트**: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 **커뮤니티**: [GitHub Discussions](https://github.com/your-repo/discussions)

---

**🎉 설치 완료 후 첫 번째 질문을 시도해보세요!**

```python
# 예시: "가장 비싼 상품 5개를 보여주세요"
response = text_to_sql_agent.query("가장 비싼 상품 5개를 보여주세요")
print(response)
```