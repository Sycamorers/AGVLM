from agri_vlm.data.transforms import parse_ip102_label
from agri_vlm.data.normalizers import normalize_classification_records_dataset


def test_rice_disease_normalizer_prefers_english_parenthetical(tmp_path):
    from PIL import Image
    import json

    image_path = tmp_path / "images" / "train" / "000001.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(10, 120, 30)).save(image_path)
    (tmp_path / "records.jsonl").write_text(
        json.dumps(
            {
                "id": "train-000001",
                "image": "images/train/000001.jpg",
                "label": "Bệnh Gạch Nâu ( Narrow Brown Spot )",
                "split": "train",
                "crop": "rice",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = normalize_classification_records_dataset(
        raw_dir=tmp_path,
        repo_root=tmp_path,
        dataset_name="rice_disease",
    )

    assert rows[0]["target"]["canonical_label"] == "narrow brown spot"
    assert rows[0]["metadata"]["original_label"] == "Narrow Brown Spot"


def test_rice_disease_normalizer_expands_nutrient_aliases(tmp_path):
    from PIL import Image
    import json

    image_path = tmp_path / "images" / "train" / "000001.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(10, 120, 30)).save(image_path)
    (tmp_path / "records.jsonl").write_text(
        json.dumps(
            {
                "id": "train-000001",
                "image": "images/train/000001.jpg",
                "label": "K",
                "split": "train",
                "crop": "rice",
                "disease": "K",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = normalize_classification_records_dataset(
        raw_dir=tmp_path,
        repo_root=tmp_path,
        dataset_name="rice_disease",
    )

    assert rows[0]["target"]["canonical_label"] == "potassium deficiency"
    assert rows[0]["metadata"]["disease"] == "potassium deficiency"
    assert rows[0]["metadata"]["source_label"] == "K"


def test_parse_ip102_label_strips_leading_class_id() -> None:
    assert parse_ip102_label("45_alfalfa_weevil") == "alfalfa weevil"
    assert parse_ip102_label("102 Cicadellidae") == "Cicadellidae"
