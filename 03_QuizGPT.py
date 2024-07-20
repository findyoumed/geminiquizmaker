# 필요한 라이브러리들을 임포트합니다.
import streamlit as st  # Streamlit: 웹 애플리케이션을 쉽게 만들 수 있는 라이브러리
import google.generativeai as genai  # Google의 생성형 AI 모델을 사용하기 위한 라이브러리
from langchain.text_splitter import (
    CharacterTextSplitter,
)  # 긴 텍스트를 작은 청크로 나누는 도구
from PyPDF2 import PdfReader  # PDF 파일을 읽기 위한 라이브러리
from docx import Document  # Word 문서를 읽기 위한 라이브러리
import io  # 입출력 작업을 위한 코어 도구
import json  # JSON 데이터를 다루기 위한 라이브러리
import time  # 시간 관련 함수를 제공하는 라이브러리
from dotenv import load_dotenv  # .env 파일을 로드하기 위한 라이브러리
import os  # 운영 체제와 상호작용하기 위한 라이브러리

# .env 파일 로드
load_dotenv()

# Gemini 모델 설정
# Google의 Gemini AI 모델을 사용하기 위해 API 키를 설정합니다.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# 'gemini-pro' 모델을 사용하도록 설정합니다.
model = genai.GenerativeModel("gemini-pro")

# Streamlit 페이지 설정
# 웹 페이지의 제목과 아이콘을 설정합니다.
st.set_page_config(page_title="GeminiQuiz", page_icon="❓")
# 페이지 상단에 큰 제목을 표시합니다.
st.title("GeminiQuiz")


# 문서 포맷팅 헬퍼 함수
def format_docs(docs):
    """
    여러 문서를 하나의 문자열로 결합합니다.
    각 문서는 두 줄의 개행으로 구분됩니다.
    """
    return "\n\n".join(docs)


# JSON 출력 파서 클래스
class JsonOutputParser:
    """
    텍스트 형식의 JSON을 파싱하여 Python 객체로 변환합니다.
    """

    def parse(self, text):
        # JSON 형식의 문자열에서 불필요한 부분을 제거합니다.
        text = text.replace("```", "").replace("json", "")
        # 정제된 문자열을 JSON으로 파싱하여 반환합니다.
        return json.loads(text)


# JsonOutputParser 인스턴스를 생성합니다.
output_parser = JsonOutputParser()


# 질문 생성 함수
def generate_questions(context):
    """
    주어진 컨텍스트를 바탕으로 Gemini 모델을 사용하여 퀴즈 질문을 생성합니다.
    """
    # Gemini 모델에 전달할 프롬프트를 구성합니다.
    prompt = f"""
    당신은 선생님 역할을 하는 도우미입니다.
    주어진 컨텍스트만을 바탕으로 10개의 질문을 만들어 사용자의 지식을 테스트하세요.
    각 질문은 4개의 답변을 가져야 하며, 그 중 3개는 틀리고 1개는 맞아야 합니다.
    정답에는 (o)로 표시하세요.

    예시:
    질문: 바다의 색깔은 무엇인가요?
    답변: 빨강|노랑|초록|파랑(o)

    질문: 조지아의 수도는 어디인가요?
    답변: 바쿠|트빌리시(o)|마닐라|베이루트

    이제 당신 차례입니다!

    컨텍스트: {context}
    """

    # 최대 재시도 횟수를 설정합니다.
    max_retries = 3
    # 지정된 횟수만큼 API 호출을 시도합니다.
    for attempt in range(max_retries):
        try:
            # Gemini 모델을 사용하여 컨텐츠를 생성합니다.
            response = model.generate_content(prompt)
            # 응답이 비어있으면 오류를 발생시킵니다.
            if not response.parts:
                raise ValueError(
                    "Gemini API에서 응답을 받지 못했습니다. 프롬프트가 차단되었을 수 있습니다."
                )
            # 생성된 텍스트를 반환합니다.
            return response.text
        except genai.types.generation_types.BlockedPromptException:
            # 프롬프트가 차단된 경우 오류를 발생시킵니다.
            raise ValueError("Gemini 검열에 의해 차단되었습니다.")
        except Exception as e:
            # 429 오류(너무 많은 요청)가 발생하면 지수 백오프로 재시도합니다.
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(2**attempt)  # 지수 백오프
                continue
            # 500 오류(서버 내부 오류)가 발생하면 사용자에게 알립니다.
            if "500" in str(e):
                raise ValueError(
                    "Gemini 서버 내부 오류가 발생했습니다. 나중에 다시 시도해 주세요."
                )
            # 그 외의 오류는 상세 내용과 함께 예외를 발생시킵니다.
            raise ValueError(f"Gemini API 호출 중 오류 발생: {str(e)}")
    # 최대 재시도 횟수를 초과하면 오류를 발생시킵니다.
    raise ValueError("Gemini API 최대 재시도 횟수를 초과했습니다.")


# 질문 포맷팅 함수
def format_questions(questions):
    """
    생성된 질문들을 JSON 형식으로 포맷팅합니다.
    """
    # Gemini 모델에 전달할 프롬프트를 구성합니다.
    prompt = f"""
    당신은 강력한 포맷팅 알고리즘입니다.
    시험 질문을 JSON 형식으로 포맷팅합니다.
    (o)가 표시된 답변이 정답입니다.

    입력 예시:
    질문: 바다의 색깔은 무엇인가요?
    답변: 빨강|노랑|초록|파랑(o)

    질문: 조지아의 수도는 어디인가요?
    답변: 바쿠|트빌리시(o)|마닐라|베이루트

    출력 예시:
    ```json
    {{ "questions": [
            {{
                "question": "바다의 색깔은 무엇인가요?",
                "answers": [
                        {{ "answer": "빨강", "correct": false }},
                        {{ "answer": "노랑", "correct": false }},
                        {{ "answer": "초록", "correct": false }},
                        {{ "answer": "파랑", "correct": true }}
                ]
            }},
            {{
                "question": "조지아의 수도는 어디인가요?",
                "answers": [
                        {{ "answer": "바쿠", "correct": false }},
                        {{ "answer": "트빌리시", "correct": true }},
                        {{ "answer": "마닐라", "correct": false }},
                        {{ "answer": "베이루트", "correct": false }}
                ]
            }}
        ]
    }}
    ```
    이제 당신 차례입니다!

    질문들: {questions}
    """

    # 최대 재시도 횟수를 설정합니다.
    max_retries = 3
    # 지정된 횟수만큼 API 호출을 시도합니다.
    for attempt in range(max_retries):
        try:
            # Gemini 모델을 사용하여 컨텐츠를 생성합니다.
            response = model.generate_content(prompt)
            # 응답이 비어있으면 오류를 발생시킵니다.
            if not response.parts:
                raise ValueError(
                    "Gemini API에서 응답을 받지 못했습니다. 프롬프트가 차단되었을 수 있습니다."
                )
            # 생성된 JSON 텍스트를 파싱하여 반환합니다.
            return output_parser.parse(response.text)
        except genai.types.generation_types.BlockedPromptException:
            # 프롬프트가 차단된 경우 오류를 발생시킵니다.
            raise ValueError("Gemini 검열에 의해 차단되었습니다.")
        except Exception as e:
            # 429 오류(너무 많은 요청)가 발생하면 지수 백오프로 재시도합니다.
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(2**attempt)  # 지수 백오프
                continue
            # 500 오류(서버 내부 오류)가 발생하면 사용자에게 알립니다.
            if "500" in str(e):
                raise ValueError(
                    "Gemini 서버 내부 오류가 발생했습니다. 나중에 다시 시도해 주세요."
                )
            # 그 외의 오류는 상세 내용과 함께 예외를 발생시킵니다.
            raise ValueError(f"Gemini API 호출 중 오류 발생: {str(e)}")
    # 최대 재시도 횟수를 초과하면 오류를 발생시킵니다.
    raise ValueError("Gemini API 최대 재시도 횟수를 초과했습니다.")


# 파일 처리 함수
@st.cache_data(show_spinner="파일 로딩 중...")
def split_file(file):
    """
    업로드된 파일을 읽고 텍스트를 추출한 후, 작은 청크로 분할합니다.
    """
    # 텍스트를 저장할 변수를 초기화합니다.
    text = ""
    # 파일 확장자를 확인합니다.
    file_extension = file.name.split(".")[-1].lower()

    # PDF 파일 처리
    if file_extension == "pdf":
        try:
            # PDF 파일을 읽습니다.
            pdf_reader = PdfReader(io.BytesIO(file.getvalue()))
            # 각 페이지의 텍스트를 추출하여 결합합니다.
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            # 추출된 텍스트가 없으면 오류를 발생시킵니다.
            if not text.strip():
                raise ValueError("PDF에 추출 가능한 텍스트가 없습니다.")
        except Exception as e:
            # PDF 처리 중 오류가 발생하면 예외를 발생시킵니다.
            raise ValueError(f"PDF 파일 처리 중 오류 발생: {str(e)}")
    # 텍스트 파일 처리
    elif file_extension == "txt":
        # 텍스트 파일의 내용을 읽어 디코딩합니다.
        text = file.getvalue().decode("utf-8")
    # Word 문서 처리
    elif file_extension in ["docx", "doc"]:
        try:
            # Word 문서를 읽습니다.
            doc = Document(io.BytesIO(file.getvalue()))
            # 각 문단의 텍스트를 추출하여 결합합니다.
            for para in doc.paragraphs:
                text += para.text + "\n"
            # 추출된 텍스트가 없으면 오류를 발생시킵니다.
            if not text.strip():
                raise ValueError("Word 문서에 추출 가능한 텍스트가 없습니다.")
        except Exception as e:
            # Word 문서 처리 중 오류가 발생하면 예외를 발생시킵니다.
            raise ValueError(f"Word 문서 처리 중 오류 발생: {str(e)}")
    else:
        # 지원하지 않는 파일 형식에 대해 오류를 발생시킵니다.
        raise ValueError("지원하지 않는 파일 형식입니다.")

    # 추출된 텍스트가 없으면 오류를 발생시킵니다.
    if not text.strip():
        raise ValueError("파일에 추출 가능한 텍스트가 없습니다.")

    # 텍스트를 작은 청크로 분할합니다.
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",  # 줄바꿈을 기준으로 텍스트를 나눕니다.
        chunk_size=800,  # 각 청크의 최대 크기를 800자로 설정합니다.
        chunk_overlap=100,  # 청크 간 100자 겹침을 허용하여 문맥을 유지합니다.
    )
    # 분할된 텍스트 청크를 반환합니다.
    return splitter.split_text(text)


# 간단한 텍스트 검색 함수
def text_search(term):
    """
    입력된 텍스트를 그대로 리스트로 반환합니다.
    실제 검색 기능은 구현되어 있지 않습니다.
    """
    return [term]


# 퀴즈 생성 함수
@st.cache_data(
    show_spinner="퀴즈 생성 중..."
)  # 이 함수의 결과를 캐시하여 성능을 향상시킵니다.
def run_quiz_chain(docs, topic):
    """
    문서나 주제를 바탕으로 퀴즈를 생성합니다.
    """
    # 문서가 있으면 문서를, 없으면 주제를 컨텍스트로 사용합니다.
    context = format_docs(docs) if docs else topic
    # 컨텍스트를 바탕으로 질문을 생성합니다.
    questions = generate_questions(context)
    # 생성된 질문을 JSON 형식으로 포맷팅합니다.
    formatted_questions = format_questions(questions)
    # 포맷팅된 질문을 반환합니다.
    return formatted_questions


# 메인 애플리케이션 로직
def main():
    """
    Streamlit 애플리케이션의 주요 로직을 정의합니다.
    """
    # 사이드바 구성
    with st.sidebar:
        docs = None
        topic = None
        # 사용자가 파일 업로드와 텍스트 입력 중 선택할 수 있게 합니다.
        choice = st.selectbox(
            "사용할 옵션을 선택하세요.",
            ("파일", "텍스트 입력"),
        )
        if choice == "파일":
            # 파일 업로더를 표시합니다.
            file = st.file_uploader(
                ".docx, .txt 또는 .pdf 파일을 업로드하세요",
                type=["pdf", "txt", "docx"],
            )
            if file:
                try:
                    # 업로드된 파일을 처리합니다.
                    docs = split_file(file)
                except Exception as e:
                    # 파일 처리 중 오류가 발생하면 에러 메시지를 표시합니다.
                    st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
        else:
            # 텍스트 입력 필드를 표시합니다.
            topic = st.text_input("주제를 입력하거나 텍스트를 붙여넣으세요...")
            if topic:
                # 입력된 텍스트를 처리합니다.
                docs = text_search(topic)

    # 문서나 주제가 없으면 시작 메시지를 표시합니다.
    if not docs and not topic:
        st.markdown(
            """
        GeminiQuiz에 오신 것을 환영합니다.

        업로드한 파일이나 입력한 텍스트를 바탕으로 퀴즈를 만들어 지식을 테스트하고 공부를 도와드립니다.

        사이드바에서 파일을 업로드하거나 텍스트를 입력하여 시작하세요.
        """
        )
    else:
        try:
            # 퀴즈를 생성합니다.
            response = run_quiz_chain(docs, topic if topic else file.name)
            # 생성된 각 질문에 대해 UI를 구성합니다.
            for question in response["questions"]:
                # 질문을 표시합니다.
                st.subheader(question["question"])
                # 답변 선택을 위한 라디오 버튼을 표시합니다.
                answer = st.radio(
                    question["question"],  # 라벨을 질문으로 설정
                    [answer["answer"] for answer in question["answers"]],
                    key=question["question"],
                    label_visibility="collapsed",
                )
                # 정답 확인 버튼을 표시합니다.
                if st.button("정답 확인", key=f"check_{question['question']}"):
                    # 정답을 찾습니다.
                    correct_answer = next(
                        a["answer"] for a in question["answers"] if a["correct"]
                    )
                    # 사용자의 답변이 정답인지 확인하고 결과를 표시합니다.
                    if answer == correct_answer:
                        st.success("정답입니다!")
                    else:
                        st.error(f"틀렸습니다. 정답은 {correct_answer}입니다.")
                # 질문 사이에 구분선을 추가합니다.
                st.write("---")
        except Exception as e:
            # 퀴즈 생성 중 오류가 발생하면 에러 메시지를 표시합니다.
            st.error(f"퀴즈 생성 중 오류가 발생했습니다: {str(e)}")


# 스크립트가 직접 실행될 때 main 함수를 호출합니다.
if __name__ == "__main__":
    main()
