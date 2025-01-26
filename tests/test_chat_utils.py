import sys
import pytest
from pathlib import Path
from langchain_community.chat_models.openai import ChatOpenAI

sys.path.append(Path(__file__).parents[1].as_posix())
from chat_utils import (
    create_llm,
    filter_stopwords,
    hypothetical_document_embeddings,
    qa_from_db,
    results_from_db,
    load_chat_embeddings,
)
from streamlit_app_utils import load_db_from_file_and_create_if_not_exists


@pytest.fixture
def selected_file_path():
    return Path(__file__).parents[1] / "documents" / "sparksofagi.pdf"


@pytest.fixture
def db(selected_file_path):
    return load_db_from_file_and_create_if_not_exists(selected_file_path.as_posix())


@pytest.fixture()
def llm():
    return create_llm("gpt-4o-mini")


def test_filter_stopwords(question="I am wondering, what is your name?"):
    filtered = filter_stopwords(question.lower())
    assert filtered == "wondering , name ?"


def test_create_llm(llm):
    assert isinstance(llm, ChatOpenAI), "Output should be an instance of ChatOpenAI"


@pytest.mark.parametrize("hypothetical", [True, False])
def test_qa_from_db(db, llm, hypothetical):
    sys_message, sources = qa_from_db(
        "Can GPT draw a unicorn?", db, llm, hypothetical=hypothetical
    )
    assert isinstance(sys_message, str), (
        f"Output should be a string, {type(sys_message)}"
    )
    assert isinstance(sources, str), f"Output should be a string, {type(sources)}"


@pytest.mark.filterwarnings("error")
def test_hypothetical_document_embeddings(llm):
    output = hypothetical_document_embeddings(
        "What is a pirate's favorite letter?", llm
    )
    assert isinstance(output, str), "Output should be a string?"


def test_results_from_db(db):
    results_from_db(db, "Is GPT good at cooking?", num_results=2)


def test_load_chat_embeddings(selected_file_path):
    load_chat_embeddings(selected_file_path.as_posix())
