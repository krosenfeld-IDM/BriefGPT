import os
import sys
import pytest
from pathlib import Path

sys.path.append(Path(__file__).parents[1].as_posix())
from streamlit_app_utils import load_db_from_file_and_create_if_not_exists, process_summarize_button, validate_api_key

@pytest.fixture
def selected_file_path():
    return Path(__file__).parents[1] / 'documents' / 'sparksofagi.pdf'

def test_load_load_db_from_file_and_create_if_not_exists(selected_file_path):
    load_db_from_file_and_create_if_not_exists(selected_file_path.as_posix())
    assert Path('embeddings/sparksofagi.faiss').exists()
    assert Path('embeddings/sparksofagi.pkl').exists()

@pytest.mark.skip(reason="Ignoring this test")
def test_validate_api_key():
    validate_api_key()

@pytest.mark.parametrize('find_clusters', [False])
@pytest.mark.skip(reason="Ignoring this test")
def test_process_summarize_button(find_clusters):
    file = 'example.txt'
    process_summarize_button(file, use_gpt_4=False, find_clusters=find_clusters, file=True)
    assert Path(f"summaries/{file.split('.')[0]}_summary.txt").exists(), f"File not found: {file.split('.')[0]}_summary.txt"
