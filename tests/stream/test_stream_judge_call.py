"""The judge's model call: what is sent, what comes back, and how failures end.

No test here reaches the model service. Every call goes to a fake `post`.
"""

import ast
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from streamclip.judge import call

KEY = "stand-in"
WINDOW1, WINDOW2 = 540.0, 2000.0


def _word(i, text, start):
    return {"i": i, "text": text, "start": start, "end": start + 0.3, "prob": 1.0}


def _jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (1, 1), "white").save(out, "JPEG")
    return out.getvalue()


def _hotspot(n, start, fired):
    return {"n": n, "peak": start + 60.0, "start": start, "end": start + 90.0, "score": 0.9,
            "fired": fired}


def write_job(root: Path, ready: list[int], hotspots: list[dict], transcripts: dict) -> Path:
    (root / "candidates").mkdir(parents=True)
    (root / "stream.json").write_text(json.dumps({
        "source": "sample.mp4", "kind": "file", "video_id": "", "title": "Ranked till I drop",
        "creator_login": "somebody", "creator_name": "Somebody", "duration": 3600.0,
        "started_at": "", "width": 1280, "height": 720, "fps": 30.0,
        "chapters": [{"start": 0.0, "end": 3600.0, "title": "Fortnite"}], "extras": {}}))
    (root / "hotspots.json").write_text(json.dumps({"wanted": 6, "hotspots": hotspots}))
    (root / "candidates" / "index.json").write_text(json.dumps(
        {"language": "en", "captions": True, "ready": ready, "notes": []}))
    for n in ready:
        folder = root / "candidates" / f"{n:02d}"
        folder.mkdir()
        words = transcripts.get(n, [])
        (folder / "transcript.json").write_text(json.dumps({
            "words": words, "duration": 90.0,
            "segments": [{"start": words[0]["start"], "end": words[-1]["end"], "text": "",
                          "first_word": 0, "last_word": len(words) - 1}] if words else []}))
        (folder / "sheet.jpg").write_bytes(_jpeg())
    return root


@pytest.fixture
def judge_job(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(call, "RETRY_WAIT", 0.0)
    monkeypatch.delenv("JUDGE_ANTHROPIC_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spoken = [_word(0, "wait", WINDOW1 + 55.0), _word(1, "what", WINDOW1 + 55.5),
              _word(2, "is", WINDOW1 + 56.0), _word(3, "that.", WINDOW1 + 56.5),
              _word(4, "OH", WINDOW1 + 61.0), _word(5, "MY", WINDOW1 + 61.4),
              _word(6, "GOD", WINDOW1 + 61.8)]
    fired = [{"signal": "chat", "t": WINDOW1 + 62.0, "strength": 0.8, "detail": "chat ×4"},
             {"signal": "voice", "t": WINDOW1 + 61.0, "strength": 0.9, "detail": "Screaming"}]
    return write_job(tmp_path / "job", [1, 2],
                     [_hotspot(1, WINDOW1, fired), _hotspot(2, WINDOW2, [])], {1: spoken})


def verdict(n, **changes) -> dict:
    return {"n": n, "keep": True, "score": 8, "reason": "A clean scare with a payoff.",
            "start": 30.0, "end": 65.0, "title": "He did not see that coming",
            "caption": "wait for it", "hashtags": ["#gaming"],
            "beats": [{"kind": "payoff", "t": 61.0}], "protect": []} | changes


def message(answer, stop="end_turn") -> bytes:
    text = answer if isinstance(answer, str) else json.dumps(answer)
    return json.dumps({"type": "message", "model": "claude-opus-5-5", "stop_reason": stop,
                       "content": [{"type": "thinking", "thinking": ""},
                                   {"type": "text", "text": text}]}).encode()


GOOD = message({"candidates": [verdict(1), verdict(2, keep=False, score=2)]})
good_answer = GOOD
REFUSAL = json.dumps({"type": "message", "model": "claude-opus-5-5", "stop_reason": "refusal",
                      "content": []}).encode()


def fake_post(*answers):
    """A stand-in for call._post that hands back each (status, body) in turn."""
    if len(answers) == 2 and isinstance(answers[0], int):
        answers = (answers,)
    queue, calls = list(answers), []

    def post(url, headers, body):
        calls.append((url, headers, body))
        return queue.pop(0)
    post.calls = calls
    return post


def test_call_py_imports_only_the_standard_library():
    tree = ast.parse(Path(call.__file__).read_text())
    names = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert names <= set(sys.stdlib_module_names)


def test_request_holds_every_ready_candidate_in_order(judge_job):
    body = call.build_request(judge_job)
    blocks = body["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "text", "image", "text", "image", "text"]
    assert "Candidate 1" in blocks[1]["text"] and blocks[2]["source"]["media_type"] == "image/jpeg"
    assert "Candidate 2" in blocks[3]["text"]
    assert body["system"] == (Path(call.__file__).parent / "prompt.md").read_text()


def test_the_prompt_has_the_model_mark_content_drops(judge_job):
    prompt = call.build_request(judge_job)["system"]
    assert "begin its `reason` with `content:`" in prompt
    for subject in ("slur", "hate", "sexual content", "really getting hurt"):
        assert subject in prompt


def test_request_asks_for_an_answer_held_to_the_schema(judge_job):
    body = call.build_request(judge_job)
    schema = json.loads((Path(call.__file__).parent / "answer_schema.json").read_text())
    assert body["model"] == call.MODEL == "claude-opus-5-5"
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": schema}
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(body)
    json.dumps(body)                                    # everything in it is plain JSON


def test_transcript_lines_are_in_window_seconds(judge_job):
    text = call.build_request(judge_job)["messages"][0]["content"][1]["text"]
    assert "[61.0] OH MY GOD" in text and "What fired: Screaming at 61 s, chat ×4 at 62 s" in text
    assert "[55.0] wait what is that." in text


def test_a_candidate_with_no_speech_says_so(judge_job):            # Review Focus 1
    assert "(no speech in this window)" in call.build_request(judge_job)["messages"][0]["content"][3]["text"]


def test_a_candidate_with_no_sheet_is_still_sent(judge_job):
    (judge_job / "candidates" / "02" / "sheet.jpg").unlink()
    blocks = call.build_request(judge_job)["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "text", "image", "text", "text"]
    assert "no frames" in blocks[3]["text"]


def test_main_writes_the_raw_answer(judge_job, tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_ANTHROPIC_KEY", "stand-in")
    monkeypatch.setattr(call, "_post", fake_post(200, good_answer))
    assert call.main([str(judge_job), str(tmp_path)]) == 0
    raw = json.loads((tmp_path / "judge_raw.json").read_text())
    assert raw["candidates"][0]["n"] == 1 and raw["model"] == "claude-opus-5-5"


def test_the_key_goes_in_x_api_key_and_nowhere_else(judge_job):
    seen = {}
    call.judge(judge_job, "stand-in", post=lambda url, headers, body: (seen.update(headers) or (200, good_answer)))
    assert seen["x-api-key"] == "stand-in" and "stand-in" not in json.dumps(call.build_request(judge_job))
    assert [name for name, value in seen.items() if "stand-in" in value] == ["x-api-key"]


def test_main_never_prints_the_key_or_the_request(judge_job, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JUDGE_ANTHROPIC_KEY", "stand-in")
    monkeypatch.setattr(call, "_post", fake_post((500, b"the key stand-in was odd"), (200, GOOD)))
    assert call.main([str(judge_job), str(tmp_path)]) == 0
    printed = "".join(capsys.readouterr())
    assert printed and "stand-in" not in printed and "OH MY GOD" not in printed


def test_the_judge_key_is_preferred_over_the_general_one(judge_job, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "general")
    monkeypatch.setenv("JUDGE_ANTHROPIC_KEY", "judge-only")
    post = fake_post(200, GOOD)
    monkeypatch.setattr(call, "_post", post)
    assert call.main([str(judge_job), str(tmp_path)]) == 0
    assert post.calls[0][1]["x-api-key"] == "judge-only"


@pytest.mark.parametrize("answers, code", [
    ([(529, b"busy"), (200, GOOD)], 0),          # one retry on a busy or failed call
    ([(200, b"not json"), (200, GOOD)], 0),      # one retry on an answer that is not JSON
    ([(200, message("Sorry, no.")), (200, GOOD)], 0),
    ([(200, message({"candidates": [verdict(99)]})), (200, GOOD)], 0),   # nobody we asked about
    ([(200, message(json.dumps({"candidates": [verdict(1)]})[:40], stop="max_tokens")), (200, GOOD)], 0),
    ([(200, REFUSAL), (200, REFUSAL)], 3),       # Review Focus 2: unusable twice -> exit 3, no file
    ([(529, b"busy"), (500, b"down")], 3),
    ([(401, b"bad key")], 4)])                   # a rejected key is not retried
def test_retries_and_exit_codes(judge_job, tmp_path, monkeypatch, answers, code):
    monkeypatch.setenv("JUDGE_ANTHROPIC_KEY", KEY)
    post = fake_post(*answers)
    monkeypatch.setattr(call, "_post", post)
    assert call.main([str(judge_job), str(tmp_path / "out")]) == code
    assert len(post.calls) == len(answers)
    assert (tmp_path / "out" / "judge_raw.json").exists() == (code == 0)


def test_a_dropped_connection_is_retried_like_a_failed_call(judge_job):
    answers = [OSError("connection reset"), (200, GOOD)]

    def post(url, headers, body):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer
    assert [c["n"] for c in call.judge(judge_job, KEY, post=post)["candidates"]] == [1, 2]


def test_usage_errors_exit_2_and_a_missing_key_exits_4(judge_job, tmp_path):
    assert call.main([]) == 2
    assert call.main([str(tmp_path / "nowhere"), str(tmp_path)]) == 2
    assert call.main([str(judge_job), str(tmp_path)]) == 4          # no key in the environment
    assert not (tmp_path / "judge_raw.json").exists()


def test_a_streamed_answer_is_read_the_same_way(judge_job):
    answer = json.dumps({"candidates": [verdict(1), verdict(2)]})
    events = [
        {"type": "message_start", "message": {"model": "claude-opus-5-5", "content": []}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": answer[:50]}},
        {"type": "ping"},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": answer[50:]}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"}]
    stream = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    raw = call.judge(judge_job, KEY, post=fake_post(200, stream))
    assert [c["n"] for c in raw["candidates"]] == [1, 2]
    cut_short = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events[:4]).encode()
    with pytest.raises(call.JudgeError):
        call.judge(judge_job, KEY, post=fake_post((200, cut_short), (200, cut_short)))


def test_more_than_24_candidates_go_in_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(call, "RETRY_WAIT", 0.0)
    ready = list(range(1, 31))
    job = write_job(tmp_path / "job", ready, [_hotspot(n, 200.0 * n, []) for n in ready], {})
    post = fake_post((200, message({"candidates": [verdict(n) for n in ready[:24]]})),
                     (200, message({"candidates": [verdict(n) for n in ready[24:]]})))
    raw = call.judge(job, KEY, post=post)
    assert [c["n"] for c in raw["candidates"]] == ready
    sent = [json.loads(body)["messages"][0]["content"] for _, _, body in post.calls]
    assert [sum(b["type"] == "image" for b in blocks) for blocks in sent] == [24, 6]
    assert "Candidate 25" in sent[1][1]["text"]


def test_no_ready_candidates_asks_nothing(tmp_path):
    job = write_job(tmp_path / "job", [], [], {})
    assert call.judge(job, KEY, post=fake_post())["candidates"] == []


def test_the_tokens_used_are_added_up_and_printed(judge_job, tmp_path, monkeypatch, capsys):
    # Yuki prices the agent from what each job costs, and the judge is its biggest model bill.
    answer = json.dumps({"candidates": [verdict(1), verdict(2)]})
    events = [
        {"type": "message_start", "message": {"model": "claude-opus-5-5", "content": [], "usage": {
            "input_tokens": 1200, "cache_read_input_tokens": 0, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": answer}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 300}},
        {"type": "message_stop"}]
    stream = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    whole = json.loads(message({"candidates": [verdict(1)]}))
    whole["usage"] = {"input_tokens": 1000, "output_tokens": 50}
    raw = call.judge(judge_job, KEY, post=fake_post(200, stream))
    assert raw["usage"] == {"input_tokens": 1200, "output_tokens": 300, "requests": 1}

    monkeypatch.setattr(call, "RETRY_WAIT", 0.0)
    monkeypatch.setenv("JUDGE_ANTHROPIC_KEY", "stand-in")
    monkeypatch.setattr(call, "_post", fake_post((200, message("not json")), (200, json.dumps(whole).encode())))
    assert call.main([str(judge_job), str(tmp_path)]) == 0
    assert json.loads((tmp_path / "judge_raw.json").read_text())["usage"]["requests"] == 2
    assert "judge: 2 requests used 1000 input tokens and 50 output tokens" in capsys.readouterr().out
