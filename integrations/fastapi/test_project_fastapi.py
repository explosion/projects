import pytest
from spacy.cli.project.run import project_run
from spacy.cli.project.assets import project_assets
from pathlib import Path


@pytest.mark.skip(reason="Import currently fails")
def test_fastapi_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "install")

    # This is ugly, but we only have the dependency here
    from fastapi.testclient import TestClient
    from scripts.main import app, ModelName

    model_names = [model.value for model in ModelName]
    assert model_names
    client = TestClient(app)
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == model_names
    articles = [{"text": "This is a text"}, {"text": "This is another text"}]
    data = {"articles": articles, "model": model_names[0]}
    response = client.post("/process/", json=data)
    assert response.status_code == 200
    result = response.json()["result"]
    assert len(result) == len(articles)
    assert [{"text": entry["text"]} for entry in result] == articles
    assert all("ents" in entry for entry in result)
