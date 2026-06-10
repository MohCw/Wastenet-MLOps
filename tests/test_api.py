import io

from PIL import Image
import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "model_loaded" in r.json()


def test_predict_no_file(client):
    r = client.post("/predict")
    assert r.status_code == 422


def test_predict_wrong_content_type(client):
    r = client.post("/predict", files={"file": ("data.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_predict_success(client, monkeypatch):
    import api.main as m

    monkeypatch.setattr(
        m,
        "_predict_one",
        lambda img, fname: {
            "filename": fname,
            "predicted_class": "cardboard",
            "confidence": 0.95,
            "scores": {
                "cardboard": 0.95,
                "glass": 0.01,
                "metal": 0.01,
                "paper": 0.01,
                "plastic": 0.01,
                "trash": 0.01,
            },
            "inference_ms": 10.0,
        },
    )
    img_bytes = io.BytesIO()
    Image.new("RGB", (64, 64)).save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    r = client.post("/predict", files={"file": ("img.jpg", img_bytes, "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_class"] == "cardboard"
    assert data["confidence"] == 0.95
    assert "scores" in data
    assert "inference_ms" in data
    assert "filename" not in data


def test_predict_invalid_image(client):
    r = client.post(
        "/predict",
        files={"file": ("fake.jpg", b"not-an-image", "image/jpeg")},
    )
    assert r.status_code == 400


def test_predict_model_not_loaded(client, monkeypatch):
    import api.main as m

    monkeypatch.setattr(m, "_model", None)
    img_bytes = io.BytesIO()
    Image.new("RGB", (64, 64)).save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    r = client.post("/predict", files={"file": ("img.jpg", img_bytes, "image/jpeg")})
    assert r.status_code == 503


@pytest.mark.parametrize(
    "fmt,mime",
    [
        ("JPEG", "image/jpeg"),
        ("PNG", "image/png"),
        ("WEBP", "image/webp"),
    ],
)
def test_predict_image_formats(client, monkeypatch, fmt, mime):
    import api.main as m

    monkeypatch.setattr(
        m,
        "_predict_one",
        lambda img, fname: {
            "filename": fname,
            "predicted_class": "glass",
            "confidence": 0.9,
            "scores": {
                "cardboard": 0.01,
                "glass": 0.9,
                "metal": 0.02,
                "paper": 0.02,
                "plastic": 0.02,
                "trash": 0.03,
            },
            "inference_ms": 5.0,
        },
    )
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format=fmt)
    buf.seek(0)
    r = client.post("/predict", files={"file": (f"img.{fmt.lower()}", buf, mime)})
    assert r.status_code == 200


def test_predict_response_schema(client, monkeypatch):
    import api.main as m

    scores = {
        "cardboard": 0.8,
        "glass": 0.05,
        "metal": 0.05,
        "paper": 0.05,
        "plastic": 0.03,
        "trash": 0.02,
    }
    monkeypatch.setattr(
        m,
        "_predict_one",
        lambda img, fname: {
            "filename": fname,
            "predicted_class": "cardboard",
            "confidence": 0.8,
            "scores": scores,
            "inference_ms": 7.5,
        },
    )
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="JPEG")
    buf.seek(0)
    r = client.post("/predict", files={"file": ("x.jpg", buf, "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"predicted_class", "confidence", "scores", "inference_ms"}
    assert data["confidence"] == max(data["scores"].values())
    assert data["inference_ms"] > 0


def test_monitoring_documented_in_openapi(client):
    schema = client.get("/openapi.json").json()
    assert "/monitoring" in schema["paths"]


def test_monitoring_redirects_when_available(client, monkeypatch, tmp_path):
    import api.main as m

    monkeypatch.setattr(m, "MONITORING_STATIC_DIR", tmp_path)  # existing dir
    r = client.get("/monitoring", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"].endswith("/monitoring/")


def test_monitoring_404_when_not_generated(client, monkeypatch, tmp_path):
    import api.main as m

    monkeypatch.setattr(m, "MONITORING_STATIC_DIR", tmp_path / "missing")  # absent dir
    r = client.get("/monitoring", follow_redirects=False)
    assert r.status_code == 404
    assert "run_drift" in r.json()["detail"]
