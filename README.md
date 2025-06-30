# Databricks Text-to-SQL RAG 시스템

이 프로젝트는 **Databricks 기반 Text-to-SQL RAG (Retrieval Augmented Generation) 시스템**을 구현합니다. LangChain Agent와 Foundation Models을 활용하여 자연어 질문을 SQL 쿼리로 변환하고 실행하는 지능형 시스템입니다.

## 🎯 프로젝트 목표

- **실제 Databricks Lakehouse** 환경에서 동작하는 Text-to-SQL 시스템
- **Northwind 샘플 데이터베이스**를 활용한 실용적인 데모
- **LangChain Agent + Function Tools** 아키텍처 구현
- **Databricks Foundation Models** 연동
- **안전하고 확장 가능한** 프로덕션 레디 시스템

## 🏗️ 시스템 아키텍처

```
사용자 자연어 질문
        ↓
LangChain Agent (추론 엔진)
        ↓
Function Tools 선택 및 실행
├── schema_search_tool     # 관련 테이블/컬럼 검색
├── sql_generation_tool    # SQL 쿼리 생성  
├── sql_execution_tool     # 안전한 SQL 실행
└── result_analysis_tool   # 결과 분석 및 요약
        ↓
Databricks Delta Lake (Northwind DB)
        ↓
최종 사용자 친화적 응답
```

## 📁 프로젝트 구조

```
databricks_rag/
├── 📓 01_databricks_setup_northwind.ipynb    # 데이터 구축 및 환경 설정
├── 📓 02_langchain_agent_text_to_sql.ipynb   # LangChain Agent 구현
├── 📄 README.md                              # 프로젝트 가이드 (본 파일)
├── 📄 requirements.txt                       # Python 의존성
├── 📄 databricks.yml                         # Databricks 프로젝트 설정
├── 📂 data/                                  # 샘플 데이터 
│   ├── processed_chunks.csv
│   └── pdf/
├── 📂 docs/                                  # 상세 문서들
│   ├── setup-guide.md                       # 설치 가이드
│   ├── architecture-guide.md                # 아키텍처 상세
│   ├── implementation-guide.md              # 구현 가이드
│   ├── usage-guide.md                       # 사용법 가이드
│   ├── api-reference.md                     # API 레퍼런스
│   └── troubleshooting-guide.md             # 트러블슈팅
└── 📂 src/                                   # 소스 코드 (향후 확장)
```

## 🚀 빠른 시작

### 1단계: 환경 설정 및 데이터 구축

```bash
# 1. 저장소 클론
git clone <repository-url>
cd databricks_rag

# 2. Databricks 환경에서 노트북 열기
# 01_databricks_setup_northwind.ipynb를 먼저 실행
```

**01_databricks_setup_northwind.ipynb**에서 수행하는 작업:
- ✅ Databricks 환경 및 Spark 세션 설정
- ✅ Northwind 샘플 데이터베이스 스키마 생성
- ✅ Delta Lake 테이블 구축 및 데이터 로드
- ✅ 데이터 검증 및 기본 SQL 테스트
- ✅ LangChain Agent 연동 준비

### 2단계: LangChain Agent 시스템 구현

```bash
# 02_langchain_agent_text_to_sql.ipynb 실행
```

**02_langchain_agent_text_to_sql.ipynb**에서 구현하는 기능:
- 🤖 **LangChain Agent** 아키텍처 설계 및 구현
- 🔧 **Function Tools** 개발 (4개 핵심 도구)
- 🌐 **Databricks Foundation Models** 연동
- 🧪 **종합 테스트** 및 데모 시나리오
- 🚀 **고급 기능** (성능 분석, 쿼리 최적화 등)

### 3단계: 시스템 테스트

```python
# 노트북에서 실행
test_agent_with_examples()        # 자동 테스트
interactive_query_demo()          # 대화형 데모  
demonstrate_advanced_features()   # 고급 기능
```

## 💡 사용 예시

### 기본 질문

```python
# 질문: "가장 비싼 상품 5개를 보여주세요"

# Agent 처리 과정:
# 1. 스키마 검색 → products 테이블 발견
# 2. SQL 생성 → SELECT * FROM northwind.products ORDER BY unitprice DESC LIMIT 5
# 3. SQL 실행 → 결과 데이터 반환
# 4. 결과 분석 → "가장 비싼 상품은 Côte de Blaye ($263.50)입니다..."
```

### 복잡한 비즈니스 질문

```python
# 질문: "1997년에 가장 많은 주문을 받은 직원의 이름과 주문 건수를 알려주세요"

# Agent 처리 과정:
# 1. 스키마 검색 → orders, employees 테이블 관련성 발견
# 2. SQL 생성 → 복잡한 JOIN 및 집계 쿼리 생성
# 3. SQL 실행 → 안전한 실행 및 결과 반환
# 4. 결과 분석 → "1997년에는 Margaret Peacock이 96건으로 최다 주문을 처리했습니다"
```

## 🛠️ 핵심 기능

### Function Tools

1. **schema_search_tool** 🔍
   - 자연어 질문에서 관련 테이블 및 컬럼 검색
   - 키워드 매칭 + 관련성 점수 계산
   - 샘플 데이터 포함하여 컨텍스트 제공
2. **sql_generation_tool** 🔧
   - Databricks Foundation Model 활용 SQL 생성
   - 스키마 컨텍스트 기반 정확한 쿼리 작성
   - Spark SQL 문법 준수
3. **sql_execution_tool** ⚡
   - 안전한 SQL 실행 (읽기 전용)
   - 위험 키워드 차단 (DROP, DELETE 등)
   - 성능 보호 (자동 LIMIT 추가)
   - 실행 시간 측정
4. **result_analysis_tool** 📊
   - AI 기반 결과 해석 및 요약
   - 사용자 친화적 자연어 응답 생성
   - 비즈니스 인사이트 제공

### 고급 기능

- 🎯 **다양한 SQL 접근법** 생성 및 비교
- 📈 **쿼리 성능 분석** 및 최적화 제안
- 📝 **쿼리 히스토리** 관리 및 통계
- 🛡️ **보안 검증** 및 오류 처리
- 🔄 **실시간 대화형** 인터페이스

## 📊 지원하는 질문 유형

### 기본 조회

- "모든 고객의 이름과 도시를 보여주세요"
- "상품 목록을 카테고리별로 정렬해주세요"

### 집계 및 통계  

- "가장 비싼 상품 10개는 무엇인가요?"
- "각 카테고리별 상품 개수와 평균 가격을 구해주세요"

### 복잡한 비즈니스 분석

- "1997년 월별 매출 추이를 알려주세요"
- "고객별 총 주문 금액이 가장 높은 10명을 보여주세요"
- "가장 인기 있는 상품 카테고리는 무엇인가요?"

### 자연어 질문

- "프랑스 고객들이 주로 어떤 상품을 주문하나요?"
- "배송이 가장 빠른 운송업체는 어디인가요?"
- "직원별 매출 성과를 비교해주세요"

## 🔧 기술 스택

- **데이터 플랫폼**: Databricks Lakehouse, Delta Lake
- **처리 엔진**: Apache Spark SQL
- **AI/ML**: Databricks Foundation Models (Llama-3.1-70B)
- **Agent 프레임워크**: LangChain ReAct Agent
- **개발 환경**: Jupyter Notebook, Python 3.8+
- **의존성**: langchain, pyspark, pandas 등

## 📚 상세 문서

- 📖 [설치 가이드](docs/setup-guide.md) - Databricks 환경 설정
- 🏗️ [아키텍처 가이드](docs/architecture-guide.md) - 시스템 설계 상세
- 💻 [구현 가이드](docs/implementation-guide.md) - 코드 구현 방법
- 📱 [사용법 가이드](docs/usage-guide.md) - 실제 사용 시나리오
- 📋 [API 레퍼런스](docs/api-reference.md) - 함수 및 클래스 설명
- 🔧 [트러블슈팅](docs/troubleshooting-guide.md) - 문제 해결 방법

## 🚀 프로덕션 배포

### 확장 가능한 아키텍처

1. **Web API 서버** (FastAPI/Flask)
2. **사용자 인터페이스** (Streamlit/Gradio)
3. **인증 및 권한 관리**
4. **로깅 및 모니터링**
5. **캐싱 및 성능 최적화**

### 보안 고려사항

- SQL Injection 방지 고도화
- 사용자별 데이터 접근 제어
- 쿼리 실행 로그 감사
- 민감 데이터 마스킹

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🙋‍♂️ 지원 및 문의

- 📧 이슈 및 버그 리포트: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 질문 및 토론: [GitHub Discussions](https://github.com/your-repo/discussions)
- 📖 문서 기여: `docs/` 폴더의 마크다운 파일 수정

---

## 🎉 다음 단계

1. **즉시 시작**: `01_databricks_setup_northwind.ipynb` 실행
2. **Agent 구현**: `02_langchain_agent_text_to_sql.ipynb` 실행
3. **테스트 및 데모**: 다양한 질문으로 시스템 검증
4. **프로덕션 확장**: Web API 및 UI 개발

**🚀 Happy Text-to-SQL RAG Development! 🚀**