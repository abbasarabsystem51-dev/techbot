"""
tests/test_part1_data_and_configs.py — Tests for configuration and data files.
"""
import json
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_config_yaml_structure():
    config_path = ROOT / "config.yaml"
    assert config_path.exists(), "config.yaml must exist in root"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert isinstance(cfg, dict)
    assert "run_plan" in cfg
    run_plan = cfg["run_plan"]

    assert run_plan["total_daily_runs"] == 5
    assert run_plan["total_daily_posts"] == 10
    assert run_plan["ai_posts_count"] == 7
    assert run_plan["tech_posts_count"] == 3

    runs = run_plan["runs"]
    assert len(runs) == 5

    expected_times = ["09:00", "12:00", "15:00", "18:00", "21:00"]
    for idx, (run, expected_time) in enumerate(zip(runs, expected_times), start=1):
        assert run["run_index"] == idx
        assert run["time"] == expected_time
        assert len(run["slots"]) == 2
        for slot in run["slots"]:
            assert "slot_id" in slot
            assert "basket" in slot
            assert "category" in slot

    # Verify feeds/sources
    assert "sources" in cfg
    sources = cfg["sources"]
    assert len(sources) >= 6
    source_names = [s["name"] for s in sources]
    assert "TechCrunch AI" in source_names
    assert "OpenAI Blog" in source_names
    assert "Google AI Blog" in source_names
    assert "Ars Technica" in source_names

    # Verify models
    assert "gemini" in cfg
    assert cfg["gemini"]["max_requests_per_day"] == 1
    assert "temperature" in cfg["gemini"]


def test_curriculum_json_complete_30_lessons():
    curriculum_path = ROOT / "curriculum.json"
    assert curriculum_path.exists(), "curriculum.json must exist in root"

    with open(curriculum_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert data["total_lessons"] == 30
    lessons = data["lessons"]
    assert len(lessons) == 30

    for idx, lesson in enumerate(lessons, start=1):
        assert lesson["lesson_id"] == idx
        assert lesson["title"], f"Lesson {idx} title missing"
        assert lesson["level"] in ["مقدماتی", "متوسط", "پیشرفته"]

        # Check 3 parts: concept, example, practice
        assert "concept" in lesson, f"Lesson {idx} missing concept part"
        concept = lesson["concept"]
        assert len(concept["summary"]) > 10
        assert len(concept["everyday_example"]) > 5
        assert len(concept["allowed_facts"]) >= 1

        assert "example" in lesson, f"Lesson {idx} missing example part"
        example = lesson["example"]
        assert example["scenario"]
        assert example["bad_case"]
        assert example["good_case"]

        assert "practice" in lesson, f"Lesson {idx} missing practice part"
        practice = lesson["practice"]
        assert practice["try_now"]
        assert "quiz" in practice
        quiz = practice["quiz"]
        assert quiz["question"]
        assert len(quiz["options"]) >= 3
        assert 0 <= quiz["correct_index"] < len(quiz["options"])
        assert quiz["explanation"]


def test_glossary_json_content():
    glossary_path = ROOT / "glossary.json"
    assert glossary_path.exists(), "glossary.json must exist in root"

    with open(glossary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    terms = data["terms"]
    assert len(terms) >= 25

    term_keys = [t["term"] for t in terms]
    assert "LLM" in term_keys
    assert "RAG" in term_keys
    assert "Token" in term_keys
    assert "Context Window" in term_keys
    assert "Fine-tuning" in term_keys
    assert "LoRA" in term_keys
    assert "Prompt Injection" in term_keys

    for t in terms:
        assert t["term"]
        assert t["fa_name"]
        assert t["short_definition"]
        assert t["everyday_analogy"]


def test_hashtags_json_content():
    hashtags_path = ROOT / "hashtags.json"
    assert hashtags_path.exists(), "hashtags.json must exist in root"

    with open(hashtags_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert "baskets" in data
    assert "news_ai" in data["baskets"]
    assert "news_tech" in data["baskets"]
    assert "tool_ai" in data["baskets"]
    assert "lesson" in data["baskets"]
    assert "channel_tags" in data
    assert "#تک_نیوز" in data["channel_tags"]


def test_scoring_weights_json_content():
    scoring_path = ROOT / "scoring_weights.json"
    assert scoring_path.exists(), "scoring_weights.json must exist in root"

    with open(scoring_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert "item_selection" in data
    criteria = data["item_selection"]["criteria"]
    assert criteria["freshness"]["weight"] == 3.0
    assert criteria["topic_relevance"]["weight"] == 3.0
    assert criteria["source_authority"]["weight"] == 2.0
    assert criteria["recency_uniqueness"]["weight"] == 2.0

    assert "post_quality" in data
    quality = data["post_quality"]["criteria"]
    assert quality["clarity"]["weight"] == 3.0
    assert quality["technical_accuracy"]["weight"] == 3.0
    assert quality["reading_value"]["weight"] == 2.0
    assert quality["length_compliance"]["weight"] == 2.0
    assert data["post_quality"]["passing_threshold"] == 7.0
