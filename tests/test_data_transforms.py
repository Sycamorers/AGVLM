from agri_vlm.data.transforms import parse_ip102_label


def test_parse_ip102_label_strips_leading_class_id() -> None:
    assert parse_ip102_label("45_alfalfa_weevil") == "alfalfa weevil"
    assert parse_ip102_label("102 Cicadellidae") == "Cicadellidae"
