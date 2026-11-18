from pipeline.checks.box_aspect import run


def test_aspect_uses_pixel_dimensions(make_ds):
    ds = make_ds(images={"a": 1}, size=(800, 400), labels={"a": [
        "0 .5 .5 .40 .08",  # 320/32 = 10, đúng ngưỡng
        "0 .5 .5 .44 .08",  # 352/32 = 11, cảnh báo
        "1 .5 .5 .04 .88",  # 352/32 = 11, box theo chiều dọc
    ]})
    issues = run(ds, max_ratio=10)
    assert [(i.line, i.severity, i.extra["ratio"]) for i in issues] == [
        (2, "warn", 11), (3, "warn", 11),
    ]


def test_aspect_skips_bad_lines_and_keeps_original_line_number(make_ds):
    ds = make_ds(images={"a": 1, "b": 2}, size=(800, 400), labels={"a": [
        "garbage", "", "93 .5 .5 .44 .08", "1 .5 .5 0 .88", "1 .5 .5 .04 .88",
    ]})
    assert [(i.image, i.line, i.code) for i in run(ds, max_ratio=10)] == [
        ("a", 5, "BOX_ASPECT_EXTREME"),
    ]
