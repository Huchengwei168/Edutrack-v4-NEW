"""EduTrack - empty-first Streamlit learning feedback platform.

The app starts with an empty SQLite database. It contains no seeded students,
questions, scores, reports, or demo answers. Teachers create the data by
uploading an exam and a scorebook; DeepSeek extracts the exam structure and
students can then ask questions about published exam items.
"""

from __future__ import annotations

import base64
import ast
import hashlib
from html import escape as html_escape
import io
import json
import mimetypes
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional chart enhancement
    go = None


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "edutrack.db"

st.set_page_config(
    page_title="EduTrack · 学习反馈中台",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create an empty database schema. No sample rows are inserted."""
    with db_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                account TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('学生端', '教师端')),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS exams (
                exam_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                subject TEXT NOT NULL,
                class_name TEXT NOT NULL,
                teacher_account TEXT,
                published INTEGER NOT NULL DEFAULT 0,
                verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS questions (
                exam_id TEXT NOT NULL,
                question_no INTEGER NOT NULL,
                question_type TEXT NOT NULL,
                question_text TEXT NOT NULL,
                options_json TEXT NOT NULL DEFAULT '[]',
                answer TEXT,
                full_score REAL,
                topic TEXT,
                analysis TEXT,
                PRIMARY KEY (exam_id, question_no),
                FOREIGN KEY (exam_id) REFERENCES exams(exam_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS scores (
                exam_id TEXT NOT NULL,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                question_no INTEGER NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY (exam_id, student_id, question_no),
                FOREIGN KEY (exam_id) REFERENCES exams(exam_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS answer_sheets (
                exam_id TEXT NOT NULL,
                student_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                data BLOB NOT NULL,
                uploaded_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (exam_id, student_id),
                FOREIGN KEY (exam_id) REFERENCES exams(exam_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS interventions (
                intervention_id TEXT PRIMARY KEY,
                teacher_account TEXT NOT NULL,
                exam_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                target_group TEXT NOT NULL,
                action TEXT NOT NULL,
                planned_date TEXT,
                evaluation_date TEXT,
                before_rate REAL,
                after_rate REAL,
                status TEXT NOT NULL DEFAULT '计划中',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (exam_id) REFERENCES exams(exam_id) ON DELETE CASCADE
            );
            """
        )
        exam_columns = {row["name"] for row in conn.execute("PRAGMA table_info(exams)").fetchall()}
        if "teacher_account" not in exam_columns:
            conn.execute("ALTER TABLE exams ADD COLUMN teacher_account TEXT")
        if "verified" not in exam_columns:
            conn.execute("ALTER TABLE exams ADD COLUMN verified INTEGER NOT NULL DEFAULT 0")


def hash_password(account: str, password: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        account.encode("utf-8"),
        120_000,
    ).hex()


def create_user(account: str, password: str, role: str) -> tuple[bool, str]:
    account = account.strip()
    if len(account) < 2:
        return False, "账号至少需要 2 个字符。"
    if len(password) < 6:
        return False, "密码至少需要 6 个字符。"
    if role not in {"学生端", "教师端"}:
        return False, "请选择正确的账号角色。"
    try:
        with db_connect() as conn:
            conn.execute(
                "INSERT INTO users(account, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (account, hash_password(account, password), role, now_iso()),
            )
        return True, "注册成功，请使用新账号登录。"
    except sqlite3.IntegrityError:
        return False, "该账号已经注册，请直接登录。"


def authenticate_user(account: str, password: str) -> sqlite3.Row | None:
    with db_connect() as conn:
        user = conn.execute("SELECT * FROM users WHERE account = ?", (account.strip(),)).fetchone()
    if user is None:
        return None
    if user["password_hash"] != hash_password(user["account"], password):
        return None
    return user


def configured_secret(name: str, default: str | None = None) -> str | None:
    """Use the on-page value first, then environment variables/Streamlit Secrets."""
    session_key = {
        "DEEPSEEK_API_KEY": "deepseek_api_key_value",
        "DEEPSEEK_MODEL": "deepseek_model_value",
    }.get(name)
    if session_key:
        session_value = str(st.session_state.get(session_key, "")).strip()
        if session_value:
            return session_value
    value = os.getenv(name)
    if value:
        return value
    try:
        secret_value = st.secrets.get(name)
    except Exception:  # Streamlit Secrets may not exist during local runs.
        secret_value = None
    return str(secret_value).strip() if secret_value else default


def get_ai_key() -> str | None:
    return configured_secret("DEEPSEEK_API_KEY")


def get_ai_model() -> str:
    return configured_secret("DEEPSEEK_MODEL", "deepseek-flash") or "deepseek-flash"


RETRYABLE_DEEPSEEK_STATUS = {408, 429, 500, 502, 503, 504}


def _deepseek_retry_delay(error: urllib.error.HTTPError, attempt: int) -> float:
    """Return a short bounded delay, honoring DeepSeek's Retry-After when present."""
    retry_after = ""
    try:
        retry_after = str(error.headers.get("Retry-After", "")).strip()
    except Exception:
        retry_after = ""
    try:
        return min(max(float(retry_after), 0.5), 20.0) if retry_after else min(2**attempt, 8.0)
    except ValueError:
        return min(2**attempt, 8.0)


def deepseek_chat(messages: list[dict[str, Any]], max_tokens: int = 1200, json_mode: bool = False) -> str:
    api_key = get_ai_key()
    if not api_key:
        raise RuntimeError("请先返回教师工作台或学生首页，在 AI 配置卡片中输入 DeepSeek API Key。")

    payload = {
        "model": get_ai_model(),
        "thinking": {"type": "disabled"},
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    body: dict[str, Any] | None = None
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            if exc.code in RETRYABLE_DEEPSEEK_STATUS and attempt < max_attempts - 1:
                time.sleep(_deepseek_retry_delay(exc, attempt))
                continue
            if exc.code == 503:
                raise RuntimeError("DeepSeek 当前服务繁忙（503），系统已自动重试 4 次，请稍后再上传或更换时间再试。") from exc
            if exc.code == 429:
                raise RuntimeError("DeepSeek API 请求过于频繁（429），请稍等片刻后再试。") from exc
            raise RuntimeError(f"DeepSeek API 返回错误（{exc.code}），请检查 API Key、模型名称或稍后再试。") from exc
        except urllib.error.URLError as exc:
            if attempt < max_attempts - 1:
                time.sleep(min(2**attempt, 8.0))
                continue
            raise RuntimeError(f"暂时无法连接 DeepSeek（{exc.reason}），请检查网络后再试。") from exc
        except (TimeoutError, OSError) as exc:
            if attempt < max_attempts - 1:
                time.sleep(min(2**attempt, 8.0))
                continue
            raise RuntimeError("连接 DeepSeek 超时，请稍后再试。") from exc
        except Exception as exc:
            raise RuntimeError(f"调用 DeepSeek API 失败：{exc}") from exc

    if body is None:
        raise RuntimeError("DeepSeek 暂时没有返回结果，请稍后再试。")

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("DeepSeek 返回内容格式异常。") from exc
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content).strip()


def parse_json_response(content: str) -> Any:
    cleaned = content.strip()
    cleaned = cleaned.replace("\ufeff", "")
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates = [cleaned]
    object_match = re.search(r"\{[\s\S]*\}", cleaned)
    array_match = re.search(r"\[[\s\S]*\]", cleaned)
    if object_match:
        candidates.append(object_match.group(0))
    if array_match:
        candidates.append(array_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                # Some model gateways return Python-style dicts with single
                # quotes or trailing commas despite being asked for JSON.
                return ast.literal_eval(candidate)
            except (SyntaxError, ValueError, TypeError):
                continue
    raise ValueError("AI 返回内容不是完整 JSON，可能是输出过长或接口返回了额外文字。请重试一次。")


def normalize_questions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get("questions", [])
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError("AI 返回中没有 questions 数组。")

    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        number_value = item.get("question_number", item.get("number", item.get("id")))
        try:
            number = int(number_value)
        except (TypeError, ValueError):
            continue
        text = str(item.get("text", item.get("question_text", "")) or "").strip()
        if not text or number in seen:
            continue
        seen.add(number)

        options = item.get("options", [])
        if isinstance(options, dict):
            options = [f"{key}. {value}" for key, value in options.items()]
        elif isinstance(options, str):
            options = [part.strip() for part in re.split(r"\n|(?=\b[A-D][.、])", options) if part.strip()]
        elif not isinstance(options, list):
            options = []

        full_score = item.get("full_score", item.get("score"))
        try:
            full_score = float(full_score) if full_score not in (None, "") else None
        except (TypeError, ValueError):
            full_score = None

        answer = item.get("answer")
        answer = str(answer).strip() if answer not in (None, "") else None
        normalized.append(
            {
                "question_no": number,
                "question_type": str(item.get("type", item.get("question_type", "待确认")) or "待确认"),
                "question_text": text,
                "options": [str(option).strip() for option in options if str(option).strip()],
                "answer": answer,
                "full_score": full_score,
                "topic": str(item.get("topic", "") or "").strip(),
                "analysis": str(item.get("analysis", "") or "").strip(),
            }
        )
    normalized.sort(key=lambda item: item["question_no"])
    if not normalized:
        raise ValueError("AI 没有识别出有效题目，请检查文件清晰度或调整试卷格式。")
    return normalized


EXAM_ANALYSIS_PROMPT = """
你是 EduTrack 的试卷结构识别器。请把教师上传的试卷内容识别为 JSON，不要输出 Markdown，不要输出解释文字。
返回格式必须是：
{
  "questions": [
    {
      "question_number": 1,
      "type": "选择题",
      "text": "完整题干",
      "options": ["A. ...", "B. ..."],
      "answer": null,
      "full_score": null,
      "topic": "",
      "analysis": ""
    }
  ]
}

规则：
1. 只识别上传内容中真实存在的题目，不能补写、猜测或生成示例题。
2. 无法确认标准答案、满分、知识点或解析时使用 null 或空字符串，不要猜测。
3. 保留完整题干和选项；题号必须是整数。
4. 允许把同一大题下的小题分别列出。
""".strip()


def extract_exam_upload(uploaded_file: Any) -> tuple[str, str]:
    """Return (kind, content), where content is text or an image data URL."""
    filename = getattr(uploaded_file, "name", "upload")
    suffix = Path(filename).suffix.lower()
    raw = uploaded_file.getvalue()

    if suffix in {".txt", ".md", ".csv"}:
        text = raw.decode("utf-8", errors="ignore").strip()
        if not text:
            raise ValueError("上传的文本文件为空。")
        return "text", text

    if suffix == ".docx":
        # DOCX is a ZIP of XML files. Reading document.xml directly keeps the
        # deployed app independent of python-docx and avoids a common Cloud
        # deployment failure when dependency installation is stale.
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                document_xml = archive.read("word/document.xml")
            root = ET.fromstring(document_xml)
        except (KeyError, zipfile.BadZipFile, ET.ParseError) as exc:
            raise ValueError("DOCX 文件结构损坏，无法读取。") from exc
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        parts = []
        for paragraph in root.findall(".//w:p", namespace):
            text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace)).strip()
            if text:
                parts.append(text)
        text = "\n".join(parts).strip()
        if not text:
            raise ValueError("Word 文件没有提取到文字。")
        return "text", text

    if suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError as exc:
            raise ValueError("缺少 pdfplumber，请先安装 requirements.txt 中的依赖。") from exc
        pages: list[str] = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text.strip())
        text = "\n\n".join(pages).strip()
        if not text:
            raise ValueError("该 PDF 没有可提取文字。请将扫描页转为清晰 PNG/JPG 后上传。")
        return "text", text

    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        mime = getattr(uploaded_file, "type", None) or mimetypes.guess_type(filename)[0] or "image/png"
        encoded = base64.b64encode(raw).decode("ascii")
        return "image", f"data:{mime};base64,{encoded}"

    raise ValueError("暂不支持该文件类型，请上传 docx、pdf、txt、md、csv 或 png/jpg。")


def analyze_exam_upload(uploaded_file: Any) -> list[dict[str, Any]]:
    kind, content = extract_exam_upload(uploaded_file)
    if kind == "text":
        user_content: Any = f"{EXAM_ANALYSIS_PROMPT}\n\n试卷原文：\n{content[:160_000]}"
    else:
        user_content = [
            {"type": "text", "text": EXAM_ANALYSIS_PROMPT},
            {"type": "image_url", "image_url": {"url": content}},
        ]
    response = deepseek_chat(
        [
            {"role": "system", "content": "你负责严谨的教育试卷结构识别。"},
            {"role": "user", "content": user_content},
        ],
        max_tokens=12000,
        json_mode=True,
    )
    return normalize_questions(parse_json_response(response))


def save_exam(
    name: str,
    subject: str,
    class_name: str,
    questions: list[dict[str, Any]],
    teacher_account: str,
) -> str:
    exam_id = uuid.uuid4().hex
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO exams(
                exam_id, name, subject, class_name, teacher_account, published, verified, created_at
            ) VALUES (?, ?, ?, ?, ?, 0, 0, ?)
            """,
            (exam_id, name.strip(), subject.strip(), class_name.strip(), teacher_account, now_iso()),
        )
        conn.executemany(
            """
            INSERT INTO questions(
                exam_id, question_no, question_type, question_text, options_json,
                answer, full_score, topic, analysis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    exam_id,
                    item["question_no"],
                    item["question_type"],
                    item["question_text"],
                    json.dumps(item["options"], ensure_ascii=False),
                    item["answer"],
                    item["full_score"],
                    item["topic"],
                    item["analysis"],
                )
                for item in questions
            ],
        )
    return exam_id


def claim_legacy_exams(teacher_account: str) -> None:
    """Assign exams created before teacher ownership existed to the first teacher who opens the app."""
    with db_connect() as conn:
        conn.execute(
            "UPDATE exams SET teacher_account = ? WHERE teacher_account IS NULL",
            (teacher_account,),
        )


def get_exams(teacher_account: str | None = None, published_only: bool = False) -> list[sqlite3.Row]:
    query = "SELECT * FROM exams"
    params: list[Any] = []
    conditions: list[str] = []
    if teacher_account is not None:
        conditions.append("teacher_account = ?")
        params.append(teacher_account)
    if published_only:
        conditions.append("published = 1")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC"
    with db_connect() as conn:
        return conn.execute(query, params).fetchall()


def get_questions(exam_id: str) -> list[dict[str, Any]]:
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT * FROM questions WHERE exam_id = ? ORDER BY question_no",
            (exam_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["options"] = json.loads(item.pop("options_json") or "[]")
        except json.JSONDecodeError:
            item["options"] = []
        result.append(item)
    return result


def update_question(exam_id: str, question_no: int, values: dict[str, Any]) -> None:
    with db_connect() as conn:
        conn.execute(
            """
            UPDATE questions
            SET question_type = ?, question_text = ?, options_json = ?,
                answer = ?, full_score = ?, topic = ?, analysis = ?
            WHERE exam_id = ? AND question_no = ?
            """,
            (
                values["question_type"],
                values["question_text"],
                json.dumps(values["options"], ensure_ascii=False),
                values["answer"],
                values["full_score"],
                values["topic"],
                values["analysis"],
                exam_id,
                question_no,
            ),
        )


def update_questions_batch(exam_id: str, items: list[tuple[int, dict[str, Any]]]) -> None:
    """Save all edited questions for one exam in a single database transaction."""
    with db_connect() as conn:
        conn.executemany(
            """
            UPDATE questions
            SET question_type = ?, question_text = ?, options_json = ?,
                answer = ?, full_score = ?, topic = ?, analysis = ?
            WHERE exam_id = ? AND question_no = ?
            """,
            [
                (
                    values["question_type"],
                    values["question_text"],
                    json.dumps(values["options"], ensure_ascii=False),
                    values["answer"],
                    values["full_score"],
                    values["topic"],
                    values["analysis"],
                    exam_id,
                    question_no,
                )
                for question_no, values in items
            ],
        )


def mark_exam_verified(exam_id: str) -> None:
    with db_connect() as conn:
        conn.execute("UPDATE exams SET verified = 1 WHERE exam_id = ?", (exam_id,))


def get_scores(exam_id: str, student_id: str | None = None) -> pd.DataFrame:
    query = "SELECT student_id AS 学号, student_name AS 姓名, question_no AS 题号, score AS 得分 FROM scores WHERE exam_id = ?"
    params: list[Any] = [exam_id]
    if student_id is not None:
        query += " AND student_id = ?"
        params.append(student_id)
    query += " ORDER BY student_id, question_no"
    with db_connect() as conn:
        rows = conn.execute(query, params).fetchall()
    if not rows:
        return pd.DataFrame(columns=["学号", "姓名", "题号", "得分"])
    return pd.DataFrame([dict(row) for row in rows])


def get_score_count(exam_id: str) -> int:
    with db_connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM scores WHERE exam_id = ?", (exam_id,)).fetchone()[0])


def save_answer_sheet(exam_id: str, student_id: str, uploaded_file: Any, teacher_account: str) -> None:
    filename = str(getattr(uploaded_file, "name", "答题卡"))
    mime_type = str(getattr(uploaded_file, "type", "") or mimetypes.guess_type(filename)[0] or "application/octet-stream")
    data = uploaded_file.getvalue()
    if not data:
        raise ValueError("答题卡文件为空，请重新选择文件。")
    with db_connect() as conn:
        owner = conn.execute("SELECT teacher_account FROM exams WHERE exam_id = ?", (exam_id,)).fetchone()
        if owner is None or owner["teacher_account"] != teacher_account:
            raise ValueError("你没有权限向这场考试上传答题卡。")
        conn.execute(
            """
            INSERT INTO answer_sheets(
                exam_id, student_id, filename, mime_type, data, uploaded_by, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(exam_id, student_id) DO UPDATE SET
                filename = excluded.filename,
                mime_type = excluded.mime_type,
                data = excluded.data,
                uploaded_by = excluded.uploaded_by,
                created_at = excluded.created_at
            """,
            (exam_id, student_id, filename, mime_type, sqlite3.Binary(data), teacher_account, now_iso()),
        )


def get_answer_sheets(exam_id: str, student_id: str | None = None) -> list[sqlite3.Row]:
    query = "SELECT exam_id, student_id, filename, mime_type, data, uploaded_by, created_at FROM answer_sheets WHERE exam_id = ?"
    params: list[Any] = [exam_id]
    if student_id is not None:
        query += " AND student_id = ?"
        params.append(student_id)
    query += " ORDER BY student_id"
    with db_connect() as conn:
        return conn.execute(query, params).fetchall()


def get_answer_sheet_count(exam_ids: list[str]) -> int:
    if not exam_ids:
        return 0
    placeholders = ",".join("?" for _ in exam_ids)
    with db_connect() as conn:
        return int(
            conn.execute(
                f"SELECT COUNT(*) FROM answer_sheets WHERE exam_id IN ({placeholders})",
                exam_ids,
            ).fetchone()[0]
        )


def get_topic_stats(exam_id: str) -> pd.DataFrame:
    """Aggregate class performance by the teacher-confirmed knowledge point."""
    scores = get_scores(exam_id)
    questions = get_questions(exam_id)
    if scores.empty or not questions:
        return pd.DataFrame(columns=["知识点", "得分率", "题目数", "有效记录数"])
    question_meta = {question["question_no"]: question for question in questions}
    topic_stats: dict[str, dict[str, float]] = {}
    for question_no, group in scores.groupby("题号"):
        question = question_meta.get(int(question_no), {})
        try:
            full_score = float(question.get("full_score"))
        except (TypeError, ValueError):
            full_score = 0.0
        if full_score <= 0:
            continue
        topic = question.get("topic") or "待教师核对"
        item = topic_stats.setdefault(topic, {"得分": 0.0, "满分": 0.0, "题目数": 0.0, "记录数": 0.0})
        item["得分"] += float(group["得分"].sum())
        item["满分"] += full_score * len(group)
        item["题目数"] += 1
        item["记录数"] += len(group)
    rows = []
    for topic, values in topic_stats.items():
        rows.append(
            {
                "知识点": topic,
                "得分率": round(values["得分"] / values["满分"] * 100, 1) if values["满分"] else 0.0,
                "题目数": int(values["题目数"]),
                "有效记录数": int(values["记录数"]),
            }
        )
    return pd.DataFrame(rows, columns=["知识点", "得分率", "题目数", "有效记录数"]).sort_values("得分率", ascending=True)


def save_intervention(
    teacher_account: str,
    exam_id: str,
    topic: str,
    target_group: str,
    action: str,
    planned_date: str,
    before_rate: float | None,
    notes: str,
) -> None:
    with db_connect() as conn:
        owner = conn.execute("SELECT teacher_account FROM exams WHERE exam_id = ?", (exam_id,)).fetchone()
        if owner is None or owner["teacher_account"] != teacher_account:
            raise ValueError("你没有权限为这场考试创建教学干预记录。")
        conn.execute(
            """
            INSERT INTO interventions(
                intervention_id, teacher_account, exam_id, topic, target_group, action,
                planned_date, before_rate, status, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '计划中', ?, ?)
            """,
            (
                uuid.uuid4().hex,
                teacher_account,
                exam_id,
                topic,
                target_group,
                action,
                planned_date or None,
                before_rate,
                notes,
                now_iso(),
            ),
        )


def get_interventions(teacher_account: str, exam_id: str | None = None) -> list[sqlite3.Row]:
    query = "SELECT i.*, e.name AS exam_name, e.subject AS subject FROM interventions i JOIN exams e ON e.exam_id = i.exam_id WHERE i.teacher_account = ?"
    params: list[Any] = [teacher_account]
    if exam_id:
        query += " AND i.exam_id = ?"
        params.append(exam_id)
    query += " ORDER BY i.created_at DESC"
    with db_connect() as conn:
        return conn.execute(query, params).fetchall()


def update_intervention(
    intervention_id: str,
    teacher_account: str,
    status: str,
    after_rate: float | None,
    evaluation_date: str,
    notes: str,
) -> None:
    with db_connect() as conn:
        conn.execute(
            """
            UPDATE interventions
            SET status = ?, after_rate = ?, evaluation_date = ?, notes = ?
            WHERE intervention_id = ? AND teacher_account = ?
            """,
            (status, after_rate, evaluation_date or None, notes, intervention_id, teacher_account),
        )


def get_student_exams(student_id: str) -> list[sqlite3.Row]:
    with db_connect() as conn:
        return conn.execute(
            """
            SELECT e.* FROM exams e
            WHERE e.published = 1
              AND (
                  EXISTS (
                      SELECT 1 FROM scores s
                      WHERE s.exam_id = e.exam_id AND s.student_id = ?
                  )
                  OR EXISTS (
                      SELECT 1 FROM answer_sheets a
                      WHERE a.exam_id = e.exam_id AND a.student_id = ?
                  )
              )
            ORDER BY e.created_at DESC
            """,
            (student_id, student_id),
        ).fetchall()


def get_student_wrong_questions(student_id: str) -> pd.DataFrame:
    """Build the student's wrong-question book from published exams only."""
    rows: list[dict[str, Any]] = []
    for exam in get_student_exams(student_id):
        scores = get_scores(exam["exam_id"], student_id)
        if scores.empty:
            continue
        question_map = {question["question_no"]: question for question in get_questions(exam["exam_id"])}
        for _, score_row in scores.iterrows():
            question_no = int(score_row["题号"])
            question = question_map.get(question_no, {})
            score = float(score_row["得分"])
            full_score = question.get("full_score")
            try:
                full_score_value = float(full_score) if full_score is not None else None
            except (TypeError, ValueError):
                full_score_value = None
            is_wrong = score < full_score_value if full_score_value and full_score_value > 0 else score <= 0
            if not is_wrong:
                continue
            rate = score / full_score_value * 100 if full_score_value and full_score_value > 0 else None
            rows.append(
                {
                    "学科": exam["subject"],
                    "考试": exam["name"],
                    "题号": question_no,
                    "题型": question.get("question_type") or "待核对",
                    "题干": question.get("question_text") or "成绩表中存在，试卷未识别",
                    "我的得分": score,
                    "满分": full_score_value if full_score_value is not None else "未识别",
                    "得分率": f"{rate:.1f}%" if rate is not None else "未计算",
                    "知识点": question.get("topic") or "待教师核对",
                    "标准答案": question.get("answer") or "待教师核对",
                    "解析": question.get("analysis") or "待教师补充",
                }
            )
    columns = ["学科", "考试", "题号", "题型", "题干", "我的得分", "满分", "得分率", "知识点", "标准答案", "解析"]
    return pd.DataFrame(rows, columns=columns)


def build_wrong_book_docx(rows: list[dict[str, Any]], title: str, include_answers: bool = True) -> bytes:
    """Create a lightweight DOCX export without adding a new runtime dependency."""
    body_parts = [
        '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="32"/></w:rPr><w:t xml:space="preserve">'
        + xml_escape(title)
        + "</w:t></w:r></w:p>",
        '<w:p><w:r><w:t xml:space="preserve">EduTrack 学生错题汇总</w:t></w:r></w:p>',
    ]
    for index, row in enumerate(rows, start=1):
        body_parts.append(
            '<w:p><w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">'
            + xml_escape(f"{index}. {row['学科']} · {row['考试']} · 第 {row['题号']} 题")
            + "</w:t></w:r></w:p>"
        )
        details = [
            f"题型：{row['题型']}",
            f"知识点：{row['知识点']}",
            f"我的得分：{row['我的得分']} / {row['满分']}（得分率：{row['得分率']}）",
            f"题干：{row['题干']}",
        ]
        if include_answers:
            details.extend([f"标准答案：{row['标准答案']}", f"解析：{row['解析']}"])
        for detail in details:
            body_parts.append('<w:p><w:r><w:t xml:space="preserve">' + xml_escape(detail) + "</w:t></w:r></w:p>")
        body_parts.append('<w:p><w:r><w:br/></w:r></w:p>')

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body>'
        + "".join(body_parts)
        + '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/></w:sectPr>'
        + "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


def parse_question_column(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(?:第\s*)?(\d+)(?:\.0+)?(?:\s*题)?", text)
    return str(int(match.group(1))) if match else None


def clean_identifier(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def excel_column_index(reference: str) -> int:
    letters = re.match(r"[A-Za-z]+", reference or "")
    if not letters:
        return 0
    index = 0
    for char in letters.group(0).upper():
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def read_xlsx_without_openpyxl(raw_bytes: bytes) -> pd.DataFrame:
    """Read the first worksheet from a basic .xlsx using only stdlib XML."""
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns = {"x": main_ns}
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            names = set(archive.namelist())
            sheet_name = "xl/worksheets/sheet1.xml"
            if sheet_name not in names:
                sheet_candidates = sorted(name for name in names if name.startswith("xl/worksheets/") and name.endswith(".xml"))
                if not sheet_candidates:
                    raise ValueError("xlsx 文件中没有找到工作表。")
                sheet_name = sheet_candidates[0]

            shared_strings: list[str] = []
            if "xl/sharedStrings.xml" in names:
                shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                for item in shared_root.findall("x:si", ns):
                    shared_strings.append("".join(node.text or "" for node in item.findall(".//x:t", ns)))

            root = ET.fromstring(archive.read(sheet_name))
            rows: list[list[Any]] = []
            max_column = 0
            for row_node in root.findall(".//x:sheetData/x:row", ns):
                values: dict[int, Any] = {}
                for cell in row_node.findall("x:c", ns):
                    column = excel_column_index(cell.get("r", "A1"))
                    cell_type = cell.get("t", "")
                    value_node = cell.find("x:v", ns)
                    inline_node = cell.find("x:is", ns)
                    value: Any = ""
                    if cell_type == "s" and value_node is not None:
                        try:
                            value = shared_strings[int(value_node.text or "0")]
                        except (ValueError, IndexError):
                            value = ""
                    elif cell_type == "inlineStr" and inline_node is not None:
                        value = "".join(node.text or "" for node in inline_node.findall(".//x:t", ns))
                    elif value_node is not None:
                        value = value_node.text or ""
                        if cell_type == "b":
                            value = value == "1"
                    values[column] = value
                    max_column = max(max_column, column)
                rows.append([values.get(column, "") for column in range(max_column + 1)])
    except (zipfile.BadZipFile, ET.ParseError, KeyError) as exc:
        raise ValueError("xlsx 文件结构损坏，无法读取。") from exc
    return pd.DataFrame(rows)


def parse_scorebook(uploaded_file: Any) -> tuple[pd.DataFrame, list[str]]:
    """Parse a real scorebook without inventing IDs, names, or scores."""
    try:
        raw = pd.read_excel(uploaded_file, sheet_name=0, header=None)
    except ImportError as exc:
        filename = str(getattr(uploaded_file, "name", "")).lower()
        if filename.endswith(".xlsx"):
            raw = read_xlsx_without_openpyxl(uploaded_file.getvalue())
        else:
            raise ValueError("解析 .xls 需要安装 xlrd；建议将成绩表另存为 .xlsx 后重试。") from exc
    header_row = None
    for idx in range(min(20, len(raw))):
        values = raw.iloc[idx].tolist()
        labels = {str(value).strip() for value in values if not pd.isna(value)}
        question_count = sum(parse_question_column(value) is not None for value in values)
        if question_count >= 1 and ({"学号", "学生学号", "考号", "学籍号"} & labels) and ({"姓名", "学生姓名", "名字"} & labels):
            header_row = idx
            break
    if header_row is None:
        raise ValueError("成绩表必须包含“学号”和“姓名”，并至少包含一列题号得分。")

    header = [str(value).strip() if not pd.isna(value) else "" for value in raw.iloc[header_row].tolist()]
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = header
    data = data.dropna(how="all")
    qcols = {column: parse_question_column(column) for column in data.columns if parse_question_column(column) is not None}
    id_col = next((column for column in data.columns if column in {"学号", "学生学号", "考号", "学籍号"}), None)
    name_col = next((column for column in data.columns if column in {"姓名", "学生姓名", "名字"}), None)
    if id_col is None or name_col is None:
        raise ValueError("成绩表必须包含学号列和姓名列。")

    records: list[dict[str, Any]] = []
    valid_rows = 0
    for _, row in data.iterrows():
        student_id = clean_identifier(row[id_col])
        student_name = clean_identifier(row[name_col])
        if not student_id or not student_name:
            continue
        valid_rows += 1
        for original, question_id in qcols.items():
            score = pd.to_numeric(row[original], errors="coerce")
            if pd.isna(score):
                continue
            records.append(
                {
                    "学号": student_id,
                    "姓名": student_name,
                    "题号": int(question_id),
                    "得分": float(score),
                }
            )
    if not records:
        raise ValueError("没有从成绩表中解析到有效成绩。")
    warnings: list[str] = []
    source_rows = len(data)
    if source_rows > valid_rows:
        warnings.append(f"有 {source_rows - valid_rows} 行缺少有效学号/姓名，已跳过。")
    return pd.DataFrame(records), warnings


def save_scores(exam_id: str, scores: pd.DataFrame) -> None:
    with db_connect() as conn:
        conn.execute("DELETE FROM scores WHERE exam_id = ?", (exam_id,))
        conn.executemany(
            "INSERT INTO scores(exam_id, student_id, student_name, question_no, score) VALUES (?, ?, ?, ?, ?)",
            [
                (exam_id, str(row["学号"]), str(row["姓名"]), int(row["题号"]), float(row["得分"]))
                for _, row in scores.iterrows()
            ],
        )


def publish_exam(exam_id: str) -> None:
    with db_connect() as conn:
        conn.execute("UPDATE exams SET published = 1 WHERE exam_id = ?", (exam_id,))


def inject_css() -> None:
    css = (APP_DIR / "edutrack.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header(title: str, subtitle: str) -> None:
    role = st.session_state.get("role", "EduTrack")
    crumb = "教师端 · EduTrack" if role == "教师端" else "学生端 · EduTrack" if role == "学生端" else "EduTrack · 学习反馈平台"
    st.markdown(
        f'<header class="page-head"><div class="crumb">{html_escape(crumb)}</div><h1>{html_escape(title)}</h1><p>{html_escape(subtitle)}</p></header>',
        unsafe_allow_html=True,
    )


def metric_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for index, (col, (label, value, hint)) in enumerate(zip(cols, items)):
        with col:
            st.markdown(
                f'<div class="metric" style="animation-delay:{index * 55}ms"><div class="label">{html_escape(label)}</div><div class="value">{html_escape(value)}</div><div class="muted">{html_escape(hint)}</div></div>',
                unsafe_allow_html=True,
            )


def render_teacher_ai_config_card() -> None:
    """Keep API configuration visible on the teacher dashboard."""
    with st.container(border=True):
        st.markdown('<div class="settings-card-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
        left, right = st.columns([3, 1.15])
        with left:
            st.markdown("### DeepSeek AI 配置")
            st.caption("上传试卷识别、AI 解析和教学建议都会使用此配置。Key 仅保存在当前会话。")
            key_col, model_col = st.columns([2, 1])
            with key_col:
                st.text_input(
                    "DeepSeek API Key",
                    type="password",
                    key="deepseek_api_key_input",
                    placeholder="sk-...",
                    help="密钥不会写入项目代码或 SQLite 数据库。",
                )
            with model_col:
                st.text_input("模型名称", key="deepseek_model_input")
        with right:
            if st.button("保存 AI 配置", type="primary", use_container_width=True, key="teacher_dashboard_save_ai"):
                st.session_state["deepseek_api_key_value"] = str(st.session_state.get("deepseek_api_key_input", "")).strip()
                st.session_state["deepseek_model_value"] = str(st.session_state.get("deepseek_model_input", "")).strip() or "deepseek-flash"
                st.success("AI 配置已保存。")
            if st.button("测试连接", use_container_width=True, key="teacher_dashboard_test_ai"):
                if not get_ai_key():
                    st.warning("请先输入并保存 API Key。")
                else:
                    with st.spinner("正在测试 DeepSeek……"):
                        try:
                            deepseek_chat(
                                [
                                    {"role": "system", "content": "你是连接测试助手。"},
                                    {"role": "user", "content": "只回复：连接成功"},
                                ],
                                max_tokens=16,
                            )
                            st.success(f"DeepSeek 已连接：{get_ai_model()}")
                        except RuntimeError as exc:
                            st.error(str(exc))
            if get_ai_key():
                st.caption(f"当前已配置：{get_ai_model()}")
            else:
                st.warning("尚未配置")


BRAND_MARK = (
    '<div class="brand"><span class="brand-icon" aria-hidden="true">'
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M20 4C9 4 4 9 4 20c11 0 16-5 16-16ZM4 20l10-10"/>'
    '</svg></span>EduTrack</div>'
)


def render_login_page() -> None:
    left, right = st.columns([1.12, 0.88], gap="large")
    with left:
        # The trail is decorative and never receives pointer or keyboard input.
        squares = []
        for index in range(16):
            progress = index / 15
            x = 4 + 105 * progress
            y = 96 - 45 * progress + 10 * progress * (1 - progress)
            size = 12 + 73 * progress
            opacity = .08 + .47 * progress
            rotation = (index * 137.508) % 90 - 45
            delay = .08 * index
            squares.append(
                f'<span class="login-square" style="--x:{x:.2f}%;--y:{y:.2f}%;'
                f'--size:{size:.1f}px;--opacity:{opacity:.2f};'
                f'--rotation:{rotation:.1f}deg;--delay:{delay:.2f}s"></span>'
            )
        st.markdown(
            f"""
            <section class="login-brand">
              {BRAND_MARK}
              <div class="eyebrow">阅卷系统的下游分析层</div>
              <h1>一步<span>一个脚印</span></h1>
              <p>教师上传真实试卷与成绩，学生查看自己的答题情况，并围绕具体题目获得学习引导。</p>
              <ol class="login-points">
                <li><span class="login-step" aria-hidden="true">01</span>教师创建考试并核对题目</li>
                <li><span class="login-step" aria-hidden="true">02</span>成绩与答题卡按学号对应</li>
                <li><span class="login-step" aria-hidden="true">03</span>学生查看分析、错题与建议</li>
              </ol>
              <div class="login-trail" aria-hidden="true">{"".join(squares)}</div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    with right:
        with st.container(border=True):
            st.markdown('<div class="auth-panel-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
            st.markdown('<div class="eyebrow">SECURE WORKSPACE</div>', unsafe_allow_html=True)
            st.markdown("## 登录或创建账号")
            st.caption("教师端与学生端使用各自注册的账号。平台不会预置演示学生或成绩。")
            student_tab, teacher_tab, register_tab = st.tabs(["学生端登录", "教师端登录", "注册账号"])

        def login_panel(tab: Any, role: str, key_prefix: str) -> None:
            with tab:
                with st.form(f"{key_prefix}_login_form"):
                    account = st.text_input("账号 / 学号", key=f"{key_prefix}_account")
                    password = st.text_input("密码", type="password", key=f"{key_prefix}_password")
                    submitted = st.form_submit_button("登录", type="primary", use_container_width=True)
                if submitted:
                    user = authenticate_user(account, password)
                    if user is None or user["role"] != role:
                        st.error("账号、密码或账号角色不正确。")
                    else:
                        st.session_state["authenticated"] = True
                        st.session_state["account"] = user["account"]
                        st.session_state["role"] = user["role"]
                        st.session_state["ai_messages"] = {}
                        st.rerun()

        login_panel(student_tab, "学生端", "student")
        login_panel(teacher_tab, "教师端", "teacher")

        with register_tab:
            with st.form("register_form"):
                role = st.selectbox("注册角色", ["学生端", "教师端"])
                account = st.text_input("新账号 / 学号")
                password = st.text_input("设置密码（至少 6 位）", type="password")
                confirm = st.text_input("确认密码", type="password")
                submitted = st.form_submit_button("注册", type="primary", use_container_width=True)
            if submitted:
                if password != confirm:
                    st.error("两次输入的密码不一致。")
                else:
                    ok, message = create_user(account, password, role)
                    if not ok:
                        st.error(message)
                    else:
                        user = authenticate_user(account, password)
                        st.session_state["authenticated"] = True
                        st.session_state["account"] = user["account"]
                        st.session_state["role"] = user["role"]
                        st.session_state["ai_messages"] = {}
                        st.rerun()

    st.caption("学生账号建议直接使用成绩表中的学号注册；教师上传成绩后，学生才能看到属于自己的已发布考试。")


def render_ai_box(exam_id: str, question: dict[str, Any], student_score: float | None = None) -> None:
    st.markdown(
        '<div class="ai-panel-marker" aria-hidden="true"></div>'
        '<div class="section-title ai-heading"><span class="ai-icon" aria-hidden="true">✦</span>AI 题目答疑</div>',
        unsafe_allow_html=True,
    )
    st.caption("DeepSeek 会着重引导学生；当你明确追问标准答案或选项时，也会在解释思路后直接核对。")
    prompt = st.text_area(
        "输入你的问题或解题思路",
        key=f"ai_question_{exam_id}_{question['question_no']}",
        height=120,
        placeholder="例如：我选了 B，但不知道错在哪里……",
    )
    if st.button("发送给 DeepSeek", key=f"send_ai_{exam_id}_{question['question_no']}", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("请先输入问题或解题思路。")
        else:
            options = "\n".join(question.get("options", [])) or "（题目没有识别到选项）"
            score_text = "未上传该题得分" if student_score is None else str(student_score)
            user_prompt = (
                "请着重引导学生理解思路、知识点和错误原因，不要编造题目中不存在的信息。"
                "如果学生明确要求标准答案、正确选项或最后结论，可以在简短讲解后直接给出，不要故意回避；"
                "如果学生没有明确要求答案，则先给提示、追问和解题步骤。\n"
                f"题号：{question['question_no']}\n"
                f"题型：{question['question_type']}\n"
                f"题干：{question['question_text']}\n"
                f"选项：{options}\n"
                f"学生得分：{score_text}\n"
                f"学生问题：{prompt.strip()}"
            )
            try:
                reply = deepseek_chat(
                    [
                        {"role": "system", "content": "你是 EduTrack 的耐心学习辅导老师。着重引导学生：先判断卡点，解释知识点和思路；学生明确要求核对标准答案时，可以给出答案并说明原因。不得编造题干、选项或不存在的评分信息。"},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=1200,
                )
                messages = st.session_state.setdefault("ai_messages", {})
                conversation = messages.setdefault(f"{exam_id}:{question['question_no']}", [])
                conversation.extend([("你", prompt.strip()), ("DeepSeek", reply)])
            except RuntimeError as exc:
                st.error(str(exc))

    for author, message in st.session_state.get("ai_messages", {}).get(f"{exam_id}:{question['question_no']}", [])[-6:]:
        with st.chat_message("user" if author == "你" else "assistant"):
            st.write(message)


def render_ai_result(message: str) -> None:
    with st.container(border=True):
        st.markdown('<div class="ai-result-marker" aria-hidden="true"></div>', unsafe_allow_html=True)
        st.markdown(message)


def render_teacher_dashboard() -> None:
    render_header("教师工作台", "上传试卷、导入成绩、确认 AI 识别结果后，再向学生发布。")
    teacher_account = st.session_state["account"]
    claim_legacy_exams(teacher_account)
    exams = get_exams(teacher_account)
    published = [exam for exam in exams if exam["published"]]
    exam_ids = [exam["exam_id"] for exam in exams]
    if exam_ids:
        placeholders = ",".join("?" for _ in exam_ids)
        with db_connect() as conn:
            student_count = conn.execute(
                f"SELECT COUNT(DISTINCT student_id) FROM scores WHERE exam_id IN ({placeholders})",
                exam_ids,
            ).fetchone()[0]
            question_count = conn.execute(
                f"SELECT COUNT(*) FROM questions WHERE exam_id IN ({placeholders})",
                exam_ids,
            ).fetchone()[0]
    else:
        student_count = 0
        question_count = 0
    metric_row(
        [
            ("已创建考试", str(len(exams)), "来自教师上传"),
            ("已发布考试", str(len(published)), "教师确认后可见"),
            ("学生记录", str(student_count), "来自成绩表"),
            ("AI 识别题目", str(question_count), "来自上传试卷"),
            ("答题卡", str(get_answer_sheet_count(exam_ids)), "按学号上传"),
        ]
    )
    st.markdown('<div class="section-title">AI 服务</div>', unsafe_allow_html=True)
    render_teacher_ai_config_card()
    st.markdown('<div class="section-title">考试列表</div>', unsafe_allow_html=True)
    if not exams:
        st.info("目前没有考试数据。请进入“上传试卷”页面，并配置 DeepSeek API Key。")
        return
    table = pd.DataFrame(
        [
            {
                "考试名称": exam["name"],
                "学科": exam["subject"],
                "班级": exam["class_name"],
                "状态": "已发布" if exam["published"] else "草稿",
                "题目核对": "已确认" if exam["verified"] else "待核对",
                "创建时间": exam["created_at"],
            }
            for exam in exams
        ]
    )
    status_styles = {
        "已发布": "color: #06684a; background-color: #e7f7f0",
        "已确认": "color: #06684a; background-color: #e7f7f0",
        "待核对": "color: #8a4900; background-color: #fef4e6",
        "草稿": "color: #545765; background-color: #f0f0f3",
    }
    styled_table = table.style.map(status_styles.get, subset=["状态", "题目核对"])
    st.dataframe(styled_table, use_container_width=True, hide_index=True)


def render_create_exam() -> None:
    render_header("上传试卷", "教师上传真实试卷，DeepSeek 识别题号、题干、选项和可确认的结构字段。")
    if not get_ai_key():
        st.warning("请先返回教师工作台，在顶部的 AI 配置卡片中输入 DeepSeek API Key。")
    with st.form("create_exam_form"):
        form_col, upload_col = st.columns([1, 1], gap="large")
        with form_col:
            name = st.text_input("考试名称", placeholder="填写考试名称")
            subject = st.text_input("学科", placeholder="填写学科")
            class_name = st.text_input("班级", placeholder="填写班级")
        with upload_col:
            uploaded = st.file_uploader(
                "上传试卷文件",
                type=["docx", "pdf", "txt", "md", "csv", "png", "jpg", "jpeg", "webp"],
                help="支持可提取文字的 docx/pdf/txt/md/csv，以及清晰图片。",
            )
            st.markdown("#### 识别流程")
            st.caption("上传教师的原始试卷 → DeepSeek 提取题目字段 → 教师在“题目核对”中确认后再导入成绩。")
        submitted = st.form_submit_button("上传并使用 DeepSeek 识别", type="primary", use_container_width=True)

    if submitted:
        if not name.strip() or not subject.strip() or not class_name.strip():
            st.error("请完整填写考试名称、学科和班级。")
        elif uploaded is None:
            st.error("请先选择试卷文件。")
        elif not get_ai_key():
            st.error("请先返回教师工作台，在顶部的 AI 配置卡片中输入 DeepSeek API Key。")
        else:
            with st.spinner("正在读取文件并请求 DeepSeek 识别，请稍候……"):
                try:
                    questions = analyze_exam_upload(uploaded)
                    exam_id = save_exam(name, subject, class_name, questions, st.session_state["account"])
                    st.session_state["last_exam_id"] = exam_id
                    st.success(f"已创建考试并保存 {len(questions)} 道 AI 识别题目。请在下方核对后，再导入成绩。")
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))

    exam_id = st.session_state.get("last_exam_id")
    if exam_id:
        questions = get_questions(exam_id)
        if questions:
            st.markdown('<div class="section-title">最近一次识别结果摘要</div>', unsafe_allow_html=True)
            table = pd.DataFrame(
                [
                    {
                        "题号": question["question_no"],
                        "题型": question["question_type"],
                        "题干": question["question_text"],
                        "满分": question["full_score"] if question["full_score"] is not None else "未识别",
                        "标准答案": question["answer"] or "待教师核对",
                        "知识点": question["topic"] or "待教师核对",
                    }
                    for question in questions
                ]
            )
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.info("请进入左侧“题目核对”页面检查 AI 识别结果，可逐题保存，也可一键解析和保存全卷；确认整卷后才能导入成绩。")


def render_question_review() -> None:
    render_header("题目核对", "教师逐题检查 AI 识别结果；保存并确认整卷后，才允许导入成绩。")
    teacher_account = st.session_state["account"]
    claim_legacy_exams(teacher_account)
    exams = get_exams(teacher_account)
    if not exams:
        st.info("目前没有待核对的考试，请先上传试卷。")
        return

    exam_ids = [exam["exam_id"] for exam in exams]
    exam_labels = {
        exam["exam_id"]: f"{exam['name']} · {'已核对' if exam['verified'] else '待核对'} · {'已发布' if exam['published'] else '未发布'}"
        for exam in exams
    }
    selected_id = st.selectbox("选择要核对的考试", exam_ids, format_func=lambda exam_id: exam_labels[exam_id])
    exam = next(exam for exam in exams if exam["exam_id"] == selected_id)
    questions = get_questions(selected_id)
    if not questions:
        st.info("该考试没有 AI 识别到题目。")
        return

    if exam["published"]:
        st.warning("该考试已经发布。修改题目后，学生端会读取最新内容，请谨慎操作。")
    elif not exam["verified"]:
        st.warning("当前考试尚未核对。可以逐题保存，也可以在页面底部一键保存全部修改，最后点击“确认本卷题目”。")
    else:
        st.success("本卷题目已确认，可以进入“导入成绩”。")

    def question_widget_keys(question_no: int) -> dict[str, str]:
        return {
            "question_type": f"review_type_{selected_id}_{question_no}",
            "question_text": f"review_text_{selected_id}_{question_no}",
            "options": f"review_options_{selected_id}_{question_no}",
            "full_score": f"review_full_{selected_id}_{question_no}",
            "answer": f"review_answer_{selected_id}_{question_no}",
            "topic": f"review_topic_{selected_id}_{question_no}",
            "analysis": f"review_analysis_{selected_id}_{question_no}",
            "pending_analysis": f"review_analysis_pending_{selected_id}_{question_no}",
        }

    def current_question_values(question: dict[str, Any]) -> dict[str, Any]:
        keys = question_widget_keys(question["question_no"])
        raw_full_score = st.session_state.get(keys["full_score"], question.get("full_score") or 0)
        try:
            full_score = float(raw_full_score) if float(raw_full_score) > 0 else None
        except (TypeError, ValueError):
            full_score = None
        options_text = str(st.session_state.get(keys["options"], "\n".join(question.get("options", []))) or "")
        return {
            "question_type": str(st.session_state.get(keys["question_type"], question.get("question_type") or "待确认")).strip() or "待确认",
            "question_text": str(st.session_state.get(keys["question_text"], question.get("question_text") or "")).strip(),
            "options": [line.strip() for line in options_text.splitlines() if line.strip()],
            "answer": str(st.session_state.get(keys["answer"], question.get("answer") or "")).strip() or None,
            "full_score": full_score,
            "topic": str(st.session_state.get(keys["topic"], question.get("topic") or "")).strip(),
            "analysis": str(st.session_state.get(keys["analysis"], question.get("analysis") or "")).strip(),
        }

    def render_question_card(question: dict[str, Any]) -> None:
        question_no = question["question_no"]
        keys = question_widget_keys(question_no)
        if keys["pending_analysis"] in st.session_state:
            st.session_state[keys["analysis"]] = st.session_state.pop(keys["pending_analysis"])
        else:
            st.session_state.setdefault(keys["analysis"], question["analysis"] or "")
        with st.container(border=True):
            st.markdown(f"### 第 {question_no} 题 · {html_escape(str(question['question_type']))}")
            st.caption("整卷已确认" if exam["verified"] else "待教师核对")
            st.text_input("题型", value=question["question_type"], key=keys["question_type"])
            st.text_area("题干", value=question["question_text"], height=120, key=keys["question_text"])
            st.text_area("选项（每行一个）", value="\n".join(question.get("options", [])), height=100, key=keys["options"])
            left, middle, right = st.columns(3)
            with left:
                st.number_input(
                    "满分（0 表示未确认）",
                    min_value=0.0,
                    value=float(question["full_score"] or 0),
                    step=0.5,
                    key=keys["full_score"],
                )
            with middle:
                st.text_input("标准答案（可留空）", value=question["answer"] or "", key=keys["answer"])
            with right:
                st.text_input("知识点（可留空）", value=question["topic"] or "", key=keys["topic"])
            st.text_area("解析（可留空，可编辑）", height=110, key=keys["analysis"])
            save_col, ai_col = st.columns([1, 1])
            with save_col:
                submitted = st.button("保存本题", type="primary", use_container_width=True, key=f"save_review_{selected_id}_{question_no}")
            with ai_col:
                generate_analysis = st.button("AI解析", use_container_width=True, key=f"ai_review_{selected_id}_{question_no}")
            if generate_analysis:
                if not get_ai_key():
                    st.error("请先在教师工作台配置 DeepSeek API Key。")
                else:
                    values = current_question_values(question)
                    try:
                        generated = deepseek_chat(
                            [
                                {
                                    "role": "system",
                                    "content": "你是严谨的学科教师。根据给定题干、选项和教师确认的答案，生成可供教师审核的题目解析。指出考查点、推理步骤和正确答案依据；不确定处明确说明，不要臆造。只输出解析正文。",
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        f"学科：{exam['subject']}\n题号：{question_no}\n题型：{values['question_type']}\n"
                                        f"题干：{values['question_text']}\n选项：{' | '.join(values['options']) or '无'}\n"
                                        f"教师确认答案：{values['answer'] or '未提供'}\n知识点：{values['topic'] or '未提供'}"
                                    ),
                                },
                            ],
                            max_tokens=1000,
                        )
                        st.session_state[keys["pending_analysis"]] = generated
                        st.rerun()
                    except (RuntimeError, ValueError) as exc:
                        st.error(str(exc))
            if submitted:
                values = current_question_values(question)
                if not values["question_text"]:
                    st.error("题干不能为空。")
                else:
                    update_question(selected_id, question_no, values)
                    st.success(f"第 {question_no} 题已保存。")

    for start in range(0, len(questions), 2):
        question_cols = st.columns(2, gap="large")
        for col, question in zip(question_cols, questions[start : start + 2]):
            with col:
                render_question_card(question)

    st.divider()
    st.caption("可以批量生成全卷解析；AI 生成内容仍可逐题编辑。点击一键保存后，页面上所有题干、答案、分值、知识点和解析才会写入试卷。")
    bulk_ai_col, bulk_save_col = st.columns(2, gap="medium")
    bulk_result_key = f"bulk_review_ai_result_{selected_id}"
    if st.session_state.get(bulk_result_key):
        st.success(st.session_state[bulk_result_key])
    with bulk_ai_col:
        if st.button("一键 AI 解析全部题目", type="primary", use_container_width=True, key=f"bulk_ai_review_{selected_id}"):
            if not get_ai_key():
                st.error("请先在教师工作台配置 DeepSeek API Key。")
            else:
                st.session_state.pop(bulk_result_key, None)
                batch_size = 6
                batches = [questions[i : i + batch_size] for i in range(0, len(questions), batch_size)]
                progress = st.progress(0.0)
                status = st.empty()
                generated_count = 0
                failed_items: list[str] = []
                for batch_index, batch in enumerate(batches, start=1):
                    prompt_items = []
                    expected_numbers = set()
                    for question in batch:
                        values = current_question_values(question)
                        if not values["question_text"]:
                            failed_items.append(f"第 {question['question_no']} 题题干为空")
                            continue
                        expected_numbers.add(question["question_no"])
                        prompt_items.append(
                            {
                                "question_number": question["question_no"],
                                "question_type": values["question_type"],
                                "question_text": values["question_text"],
                                "options": values["options"],
                                "teacher_confirmed_answer": values["answer"],
                                "knowledge_point": values["topic"],
                            }
                        )
                    if prompt_items:
                        status.info(f"AI 正在解析第 {batch_index}/{len(batches)} 批题目……")
                        try:
                            response = deepseek_chat(
                                [
                                    {
                                        "role": "system",
                                        "content": (
                                            "你是严谨的学科教师。请为输入的每一道题分别生成简洁、准确、适合学生复习的中文解析，"
                                            "说明考查点、关键推理步骤和标准答案依据。若教师未提供标准答案，不要猜答案，需明确写明标准答案待教师核对。"
                                            "不得改写或补造题干、选项、答案。必须覆盖所有输入题号，只返回 JSON 对象，格式为："
                                            "{\"analyses\":[{\"question_number\":1,\"analysis\":\"解析正文\"}]}。"
                                        ),
                                    },
                                    {"role": "user", "content": json.dumps({"学科": exam["subject"], "questions": prompt_items}, ensure_ascii=False)},
                                ],
                                max_tokens=min(5000, max(1200, 500 * len(prompt_items))),
                                json_mode=True,
                            )
                            payload = parse_json_response(response)
                            results = payload.get("analyses", []) if isinstance(payload, dict) else []
                            result_map = {}
                            for item in results if isinstance(results, list) else []:
                                if not isinstance(item, dict):
                                    continue
                                try:
                                    result_no = int(item.get("question_number"))
                                except (TypeError, ValueError):
                                    continue
                                analysis_text = str(item.get("analysis") or "").strip()
                                if result_no in expected_numbers and analysis_text:
                                    result_map[result_no] = analysis_text
                            for question_no in expected_numbers:
                                analysis_text = result_map.get(question_no)
                                if analysis_text:
                                    pending_key = question_widget_keys(question_no)["pending_analysis"]
                                    st.session_state[pending_key] = analysis_text
                                    generated_count += 1
                                else:
                                    failed_items.append(f"第 {question_no} 题：AI 未返回有效解析")
                        except (RuntimeError, ValueError) as exc:
                            failed_items.extend(f"第 {question['question_no']} 题：{exc}" for question in batch if question["question_no"] in expected_numbers)
                    progress.progress(batch_index / len(batches))
                summary = f"一键解析完成：成功生成 {generated_count}/{len(questions)} 道题的解析。"
                if failed_items:
                    summary += " 未完成：" + "；".join(failed_items[:8])
                    if len(failed_items) > 8:
                        summary += f"；另有 {len(failed_items) - 8} 项未列出"
                st.session_state[bulk_result_key] = summary
                st.rerun()
    with bulk_save_col:
        if st.button("一键保存全部题目", type="primary", use_container_width=True, key=f"bulk_save_review_{selected_id}"):
            save_items = []
            empty_questions = []
            for question in questions:
                values = current_question_values(question)
                if not values["question_text"]:
                    empty_questions.append(question["question_no"])
                save_items.append((question["question_no"], values))
            if empty_questions:
                st.error("以下题目题干不能为空，请补齐后再保存：" + "、".join(str(number) for number in empty_questions[:20]))
            else:
                update_questions_batch(selected_id, save_items)
                st.success(f"已一次性保存本卷全部 {len(save_items)} 道题目的当前内容。")

    if st.button("确认本卷题目，允许导入成绩", type="primary", key=f"verify_exam_{selected_id}"):
        mark_exam_verified(selected_id)
        st.success("本卷已确认。现在可以进入“导入成绩”上传成绩表。")


def render_import_scores() -> None:
    render_header("导入成绩", "教师上传阅卷系统导出的真实成绩表；程序不会生成学号、姓名或成绩。")
    teacher_account = st.session_state["account"]
    claim_legacy_exams(teacher_account)
    exams = get_exams(teacher_account)
    if not exams:
        st.info("请先在“上传试卷”页面创建考试。")
        return
    exam_ids = [exam["exam_id"] for exam in exams]
    score_counts = {exam_id: get_score_count(exam_id) for exam_id in exam_ids}
    exam_labels = {exam["exam_id"]: f"{exam['name']} · 当前 {score_counts[exam['exam_id']]} 条成绩" for exam in exams}
    selected_id = st.selectbox("选择对应考试", exam_ids, format_func=lambda exam_id: exam_labels[exam_id])
    selected_exam = next(exam for exam in exams if exam["exam_id"] == selected_id)
    if not selected_exam["verified"]:
        st.warning("该考试还没有完成题目核对。请先进入“题目核对”页面，确认整卷后再导入成绩。")
        return
    upload_col, preview_col = st.columns([.9, 1.4], gap="large")
    with upload_col:
        with st.container(border=True):
            st.markdown("### 上传成绩表")
            uploaded = st.file_uploader("上传 xlsx / xls 成绩表", type=["xlsx", "xls"], key="score_uploader")
            st.caption("程序按表格中的真实学号、姓名、题号和得分解析；不会补造缺失数据。")
            if st.button("解析并保存成绩", type="primary", use_container_width=True):
                if uploaded is None:
                    st.error("请先选择成绩表。")
                else:
                    try:
                        scores, warnings = parse_scorebook(uploaded)
                        save_scores(selected_id, scores)
                        st.session_state["last_score_exam_id"] = selected_id
                        for warning in warnings:
                            st.warning(warning)
                        st.success(f"已保存 {scores['学号'].nunique()} 名学生、{len(scores)} 条题目得分记录。")
                    except (ValueError, KeyError) as exc:
                        st.error(str(exc))

    with preview_col:
        st.markdown('<div class="section-title">成绩预览与发布</div>', unsafe_allow_html=True)
        result = get_scores(selected_id)
        if result.empty:
            st.info("该考试还没有成绩记录。")
        else:
            st.dataframe(result, use_container_width=True, hide_index=True)
            question_numbers = {question["question_no"] for question in get_questions(selected_id)}
            score_numbers = set(result["题号"].astype(int).tolist())
            unknown_numbers = sorted(score_numbers - question_numbers)
            if unknown_numbers:
                st.warning(f"成绩表中有题号未出现在 AI 识别结果中：{unknown_numbers}。请教师核对。")
            if st.button("确认发布给学生", type="primary", key=f"publish_{selected_id}"):
                publish_exam(selected_id)
                st.success("考试已发布。学生使用成绩表中的学号注册并登录后即可查看自己的数据和题目答疑。")


def render_upload_answer_sheets() -> None:
    render_header("上传学生答题卡", "教师选择考试并按学号上传答题卡；同一考试同一学号再次上传会覆盖旧文件。")
    teacher_account = st.session_state["account"]
    claim_legacy_exams(teacher_account)
    exams = get_exams(teacher_account)
    if not exams:
        st.info("请先在“上传试卷”页面创建考试。")
        return

    exam_ids = [exam["exam_id"] for exam in exams]
    exam_labels = {
        exam["exam_id"]: f"{exam['name']} · {'已发布' if exam['published'] else '未发布'} · {get_score_count(exam['exam_id'])} 条成绩"
        for exam in exams
    }
    selected_id = st.selectbox("选择对应考试", exam_ids, format_func=lambda exam_id: exam_labels[exam_id])
    upload_col, list_col = st.columns([.9, 1.4], gap="large")
    with upload_col:
        with st.container(border=True):
            st.markdown("### 上传学生答题卡")
            student_id = st.text_input("学生学号", placeholder="必须与成绩表中的学号完全一致")
            uploaded = st.file_uploader(
                "上传该学生的答题卡",
                type=["png", "jpg", "jpeg", "webp", "pdf"],
                key="answer_sheet_uploader",
                help="支持 PNG、JPG、WEBP 图片或 PDF。一个考试和一个学号只保留最新上传的文件。",
            )
            if st.button("保存答题卡", type="primary", use_container_width=True):
                clean_student_id = student_id.strip()
                if not clean_student_id:
                    st.error("请先填写学生学号。")
                elif uploaded is None:
                    st.error("请先选择答题卡文件。")
                else:
                    score_rows = get_scores(selected_id, clean_student_id)
                    all_scores = get_scores(selected_id)
                    if not all_scores.empty and score_rows.empty:
                        st.error("该学号不在本场考试的成绩表中，请检查学号是否输入正确。")
                    else:
                        try:
                            save_answer_sheet(selected_id, clean_student_id, uploaded, teacher_account)
                            st.success(f"已保存学号 {clean_student_id} 的答题卡。学生端发布后即可查看。")
                        except ValueError as exc:
                            st.error(str(exc))

    with list_col:
        st.markdown('<div class="section-title">已上传答题卡</div>', unsafe_allow_html=True)
        sheets = get_answer_sheets(selected_id)
        if sheets:
            table = pd.DataFrame(
                [
                    {
                        "学号": sheet["student_id"],
                        "文件名": sheet["filename"],
                        "上传时间": sheet["created_at"],
                    }
                    for sheet in sheets
                ]
            )
            st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.info("该考试还没有上传答题卡。")


def render_class_report() -> None:
    render_header("班级报告", "报告只基于教师上传的成绩和 AI 识别的题目字段，不预置任何统计结果。")
    teacher_account = st.session_state["account"]
    claim_legacy_exams(teacher_account)
    exams = get_exams(teacher_account)
    if not exams:
        st.info("目前没有考试数据。")
        return
    exam_ids = [exam["exam_id"] for exam in exams]
    score_counts = {exam_id: get_score_count(exam_id) for exam_id in exam_ids}
    exam_labels = {exam["exam_id"]: f"{exam['name']} · 当前 {score_counts[exam['exam_id']]} 条成绩" for exam in exams}
    default_index = next((index for index, exam_id in enumerate(exam_ids) if score_counts[exam_id] > 0), 0)
    selected_id = st.selectbox("选择考试", exam_ids, index=default_index, format_func=lambda exam_id: exam_labels[exam_id])
    scores = get_scores(selected_id)
    questions = get_questions(selected_id)
    if scores.empty:
        st.info("该考试还没有上传成绩。")
        return

    student_count = scores["学号"].nunique()
    total_scores = scores.groupby(["学号", "姓名"], as_index=False)["得分"].sum().sort_values("得分", ascending=False)
    metric_row(
        [
            ("学生人数", str(student_count), "来自成绩表"),
            ("成绩记录", str(len(scores)), "题目级记录"),
            ("班级平均总分", f"{total_scores['得分'].mean():.2f}", "按上传得分计算"),
            ("识别题目", str(len(questions)), "来自试卷 AI 识别"),
        ]
    )
    st.markdown('<div class="section-title">学生成绩汇总</div>', unsafe_allow_html=True)
    with st.expander("展开学生成绩汇总", expanded=False):
        st.dataframe(total_scores, use_container_width=True, hide_index=True)

    question_meta = {question["question_no"]: question for question in questions}
    question_rows: list[dict[str, Any]] = []
    for question_no, group in scores.groupby("题号"):
        question = question_meta.get(int(question_no), {})
        full_score = question.get("full_score")
        row: dict[str, Any] = {
            "题号": int(question_no),
            "题干": question.get("question_text", "成绩表中存在，试卷未识别"),
            "平均得分": round(float(group["得分"].mean()), 3),
            "知识点": question.get("topic") or "待教师核对",
        }
        if full_score:
            row["满分"] = full_score
            row["平均得分率"] = f"{group['得分'].mean() / full_score * 100:.1f}%"
        else:
            row["满分"] = "未识别"
            row["平均得分率"] = "未计算"
        question_rows.append(row)
    st.markdown('<div class="section-title">题目表现</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(question_rows), use_container_width=True, hide_index=True)

    topic_stats: dict[str, dict[str, float]] = {}
    for question_no, group in scores.groupby("题号"):
        question = question_meta.get(int(question_no), {})
        full_score = question.get("full_score")
        try:
            full_score_value = float(full_score) if full_score is not None else None
        except (TypeError, ValueError):
            full_score_value = None
        if not full_score_value or full_score_value <= 0:
            continue
        topic = question.get("topic") or "待教师核对"
        item = topic_stats.setdefault(topic, {"得分": 0.0, "满分": 0.0, "题目数": 0.0, "记录数": 0.0})
        item["得分"] += float(group["得分"].sum())
        item["满分"] += full_score_value * len(group)
        item["题目数"] += 1
        item["记录数"] += len(group)

    if topic_stats:
        topic_rows = []
        for topic, values in topic_stats.items():
            rate = values["得分"] / values["满分"] * 100 if values["满分"] else 0.0
            topic_rows.append(
                {
                    "知识点": topic,
                    "得分率": round(rate, 1),
                    "题目数": int(values["题目数"]),
                    "有效记录数": int(values["记录数"]),
                }
            )
        topic_df = pd.DataFrame(topic_rows).sort_values("得分率", ascending=True)
        st.markdown('<div class="section-title">知识点得分率</div>', unsafe_allow_html=True)
        if go is not None:
            figure = go.Figure(
                go.Bar(
                    x=topic_df["知识点"],
                    y=topic_df["得分率"],
                    text=[f"{value:.1f}%" for value in topic_df["得分率"]],
                    textposition="auto",
                    marker_color="#5747e8",
                )
            )
            figure.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=30, b=100),
                yaxis=dict(title="得分率（%）", range=[0, 100]),
                xaxis=dict(title="知识点", tickangle=-35),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#555963"),
                transition=dict(duration=350),
            )
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.bar_chart(topic_df.set_index("知识点")[["得分率"]])
        st.dataframe(
            topic_df.assign(得分率=topic_df["得分率"].map(lambda value: f"{value:.1f}%")),
            use_container_width=True,
            hide_index=True,
        )
        chart_col, distribution_col = st.columns([1, 1.2], gap="large")
        with chart_col:
            st.markdown('<div class="section-title">知识点能力分布</div>', unsafe_allow_html=True)
            if go is not None and len(topic_df) >= 3:
                radar_topics = topic_df["知识点"].astype(str).tolist()
                radar_values = topic_df["得分率"].astype(float).tolist()
                figure = go.Figure(
                    go.Scatterpolar(
                        r=radar_values + radar_values[:1],
                        theta=radar_topics + radar_topics[:1],
                        fill="toself",
                        fillcolor="rgba(87,71,232,.14)",
                        line=dict(color="#5747e8", width=2),
                        name="班级得分率",
                    )
                )
                figure.update_layout(
                    height=350,
                    margin=dict(l=35, r=35, t=30, b=30),
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#555963"),
                )
                st.plotly_chart(figure, use_container_width=True)
            else:
                st.info("至少有 3 个可计算得分率的知识点后，才显示能力分布图。")
        with distribution_col:
            st.markdown('<div class="section-title">学生总分分布</div>', unsafe_allow_html=True)
            if go is not None:
                figure = go.Figure(
                    go.Histogram(
                        x=total_scores["得分"],
                        nbinsx=min(12, max(4, int(len(total_scores) ** 0.5))),
                        marker_color="#7856dc",
                        opacity=.88,
                    )
                )
                figure.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=45),
                    xaxis_title="上传成绩表中的总得分",
                    yaxis_title="学生人数",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#555963"),
                    showlegend=False,
                )
                st.plotly_chart(figure, use_container_width=True)
            else:
                st.bar_chart(total_scores["得分"].value_counts().sort_index())
    else:
        st.info("当前考试的题目还没有可用于计算得分率的满分字段，请先在题目核对页面补充满分。")


def render_teacher_intervention() -> None:
    teacher_account = st.session_state["account"]
    render_header("教学干预与评估", "根据班级知识点得分率制定干预措施，并在后续复测后记录效果。")
    claim_legacy_exams(teacher_account)
    exams = [exam for exam in get_exams(teacher_account) if get_score_count(exam["exam_id"]) > 0]
    if not exams:
        st.info("目前没有可用于教学干预的考试成绩。请先核对试卷并导入成绩。")
        return

    exam_ids = [exam["exam_id"] for exam in exams]
    exam_id = st.selectbox(
        "选择基准考试",
        exam_ids,
        key="teacher_intervention_exam",
        format_func=lambda value: next(exam["name"] for exam in exams if exam["exam_id"] == value),
    )
    topic_df = get_topic_stats(exam_id)
    if topic_df.empty:
        st.warning("该考试暂时没有可计算得分率的知识点，请先在题目核对页面补充满分和知识点。")
        return

    class_scores = get_scores(exam_id)
    question_meta = {question["question_no"]: question for question in get_questions(exam_id)}
    topic_errors: dict[str, dict[str, int]] = {}
    question_errors: list[dict[str, Any]] = []
    for question_no, group in class_scores.groupby("题号"):
        question = question_meta.get(int(question_no), {})
        try:
            full_score = float(question.get("full_score"))
        except (TypeError, ValueError):
            continue
        if full_score <= 0:
            continue
        missed = int((group["得分"] < full_score).sum())
        records = len(group)
        topic = question.get("topic") or "待教师核对"
        counts = topic_errors.setdefault(topic, {"未满分": 0, "记录数": 0})
        counts["未满分"] += missed
        counts["记录数"] += records
        question_errors.append(
            {
                "题号": int(question_no),
                "知识点": topic,
                "未满分人数": missed,
                "有效人数": records,
                "错误率": missed / records * 100 if records else 0.0,
                "题干": question.get("question_text") or "",
            }
        )
    topic_error_rows = [
        {
            "知识点": topic,
            "未满分人次": values["未满分"],
            "作答人次": values["记录数"],
            "错误率": values["未满分"] / values["记录数"] * 100 if values["记录数"] else 0.0,
        }
        for topic, values in topic_errors.items()
    ]
    topic_error_df = pd.DataFrame(topic_error_rows)
    if not topic_error_df.empty:
        topic_error_df = topic_error_df.sort_values("错误率", ascending=False)
    question_error_df = pd.DataFrame(question_errors)
    if not question_error_df.empty:
        question_error_df = question_error_df.sort_values("错误率", ascending=False)

    st.markdown('<div class="section-title">本次试卷错误情况分析</div>', unsafe_allow_html=True)
    if not topic_error_df.empty:
        total_missed = int(topic_error_df["未满分人次"].sum())
        total_attempts = int(topic_error_df["作答人次"].sum())
        overall_error_rate = total_missed / total_attempts * 100 if total_attempts else 0.0
        highest_topic = topic_error_df.iloc[0]
        metric_row(
            [
                ("整体未满分率", f"{overall_error_rate:.1f}%", f"{total_missed} / {total_attempts} 条题目作答记录"),
                ("最高错误率知识点", str(highest_topic["知识点"]), f"{highest_topic['错误率']:.1f}% 未满分"),
                ("已分析知识点", str(len(topic_error_df)), "按教师核对的满分与知识点统计"),
            ]
        )
        with st.container(border=True):
            st.markdown(
                f"本次分析基于 **{total_attempts}** 条真实题目作答记录。未满分率最高的知识点是 **{highest_topic['知识点']}**，"
                f"错误率为 **{highest_topic['错误率']:.1f}%**（{int(highest_topic['未满分人次'])}/{int(highest_topic['作答人次'])} 条记录未达满分）。"
                "下方列出需要优先关注的知识点和题目；AI 建议会根据这些统计给出教学活动与复测办法。"
            )
        high_topics = topic_error_df[topic_error_df["错误率"] >= 50]
        if not high_topics.empty:
            st.warning(
                "以下知识点有至少一半作答记录未达到满分："
                + "；".join(f"{row['知识点']}（{row['错误率']:.1f}%）" for _, row in high_topics.iterrows())
            )
        else:
            top_topics = topic_error_df.head(3)
            st.info(
                "本次没有知识点错误率达到 50%；相对薄弱项为："
                + "；".join(f"{row['知识点']}（{row['错误率']:.1f}%）" for _, row in top_topics.iterrows())
            )
        st.dataframe(
            topic_error_df.assign(错误率=topic_error_df["错误率"].map(lambda value: f"{value:.1f}%")),
            use_container_width=True,
            hide_index=True,
        )
        if not question_error_df.empty:
            st.caption("未满分比例最高的题目")
            st.dataframe(
                question_error_df.head(5)[["题号", "知识点", "未满分人数", "有效人数", "错误率"]].assign(
                    错误率=lambda frame: frame["错误率"].map(lambda value: f"{value:.1f}%")
                ),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("本次考试没有足够的满分信息计算错误率。请先在题目核对页面补充满分和知识点。")

    st.markdown('<div class="section-title">AI 教学建议</div>', unsafe_allow_html=True)
    intervention_ai_key = f"teacher_intervention_ai_{teacher_account}_{exam_id}"
    if st.button("生成 / 更新 AI 教学建议", type="primary", key=f"generate_intervention_ai_{exam_id}"):
        if not get_ai_key():
            st.warning("请先在教师工作台配置 DeepSeek API Key。")
        elif topic_error_df.empty:
            st.warning("没有可分析的知识点错误数据。")
        else:
            prompt_data = {
                "考试": next(exam["name"] for exam in exams if exam["exam_id"] == exam_id),
                "知识点错误统计": topic_error_df.head(10).to_dict("records"),
                "高错误率题目": question_error_df.head(10)[["题号", "知识点", "错误率", "题干"]].to_dict("records") if not question_error_df.empty else [],
            }
            try:
                with st.spinner("AI 正在分析班级错漏并制定教学建议……"):
                    recommendation = deepseek_chat(
                        [
                            {
                                "role": "system",
                                "content": "你是严谨的教研教师。只依据提供的真实统计和题目内容，先总结错误率较高的知识点及证据，再给出可执行的课堂干预、练习类型、时间安排和复测评估方法。若数据不足要说明，不得编造学生特点或统计结果。用中文分点回答。",
                            },
                            {"role": "user", "content": json.dumps(prompt_data, ensure_ascii=False)},
                        ],
                        max_tokens=1800,
                    )
                st.session_state[intervention_ai_key] = recommendation
            except RuntimeError as exc:
                st.error(str(exc))
    if st.session_state.get(intervention_ai_key):
        render_ai_result(st.session_state[intervention_ai_key])

    topic_names = topic_df["知识点"].astype(str).tolist()
    selected_topic = st.selectbox("目标知识点", topic_names, key=f"intervention_topic_{exam_id}")
    baseline = float(topic_df.loc[topic_df["知识点"] == selected_topic, "得分率"].iloc[0])
    st.info(f"当前班级基准得分率：{baseline:.1f}%。干预后可在下方记录复测得分率，系统会自动计算变化。")

    with st.form("new_intervention_form"):
        target_group = st.text_input("干预对象", value="该知识点得分较低的学生")
        action = st.text_area("干预措施", placeholder="例如：安排语法专题讲解、错题订正和二次练习。", height=100)
        planned_date = st.date_input("计划日期", value=datetime.now().date())
        notes = st.text_area("补充说明（可选）", height=80)
        submitted = st.form_submit_button("保存教学干预", type="primary")
    if submitted:
        if not target_group.strip() or not action.strip():
            st.error("请填写干预对象和干预措施。")
        else:
            save_intervention(
                teacher_account,
                exam_id,
                selected_topic,
                target_group.strip(),
                action.strip(),
                planned_date.isoformat(),
                baseline,
                notes.strip(),
            )
            st.success("教学干预已保存。完成复测后，在下方补充干预后得分率。")

    records = get_interventions(teacher_account)
    st.markdown('<div class="section-title">干预记录与评估</div>', unsafe_allow_html=True)
    if not records:
        st.info("还没有教学干预记录。")
        return
    summary_rows = []
    improvements = []
    for record in records:
        before = record["before_rate"]
        after = record["after_rate"]
        change = after - before if before is not None and after is not None else None
        if change is not None:
            improvements.append(change)
        summary_rows.append(
            {
                "考试": record["exam_name"],
                "知识点": record["topic"],
                "干预对象": record["target_group"],
                "干预措施": record["action"],
                "干预前": f"{before:.1f}%" if before is not None else "未记录",
                "干预后": f"{after:.1f}%" if after is not None else "待复测",
                "变化": f"{change:+.1f}%" if change is not None else "待评估",
                "状态": record["status"],
            }
        )
    metric_row(
        [
            ("干预记录", str(len(records)), "当前教师创建"),
            ("已完成评估", str(sum(record["after_rate"] is not None for record in records)), "已填写干预后得分率"),
            ("平均变化", f"{sum(improvements) / len(improvements):+.1f}%" if improvements else "待评估", "干预后减干预前"),
        ]
    )
    st.table(pd.DataFrame(summary_rows))

    for record in records:
        with st.expander(f"{record['exam_name']} · {record['topic']} · {record['status']}"):
            st.write(f"干预对象：{record['target_group']}")
            st.write(f"干预措施：{record['action']}")
            st.caption(f"计划日期：{record['planned_date'] or '未填写'} · 创建时间：{record['created_at']}")
            with st.form(f"evaluate_intervention_{record['intervention_id']}"):
                status = st.selectbox(
                    "状态",
                    ["计划中", "进行中", "已完成", "已取消"],
                    index=["计划中", "进行中", "已完成", "已取消"].index(record["status"]) if record["status"] in {"计划中", "进行中", "已完成", "已取消"} else 0,
                    key=f"intervention_status_{record['intervention_id']}",
                )
                after_text = st.text_input(
                    "干预后得分率（%）",
                    value="" if record["after_rate"] is None else f"{record['after_rate']:.1f}",
                    key=f"intervention_after_{record['intervention_id']}",
                )
                evaluation_date = st.date_input(
                    "评估日期",
                    value=datetime.fromisoformat(record["evaluation_date"]).date() if record["evaluation_date"] else datetime.now().date(),
                    key=f"intervention_date_{record['intervention_id']}",
                )
                evaluation_notes = st.text_area(
                    "评估说明",
                    value=record["notes"] or "",
                    key=f"intervention_notes_{record['intervention_id']}",
                    height=80,
                )
                update_submitted = st.form_submit_button("保存评估")
            if update_submitted:
                try:
                    after_rate = float(after_text.strip()) if after_text.strip() else None
                except ValueError:
                    after_rate = -1
                if after_rate is not None and not 0 <= after_rate <= 100:
                    st.error("干预后得分率必须是 0 到 100 之间的数字。")
                else:
                    update_intervention(
                        record["intervention_id"],
                        teacher_account,
                        status,
                        after_rate,
                        evaluation_date.isoformat(),
                        evaluation_notes.strip(),
                    )
                    st.success("评估结果已保存。")


def render_student_dashboard() -> None:
    account = st.session_state["account"]
    render_header("学生端", f"当前账号：{account}。只有教师发布且成绩表中匹配到你的考试才会显示。")
    render_student_ai_config_card()
    exams = get_student_exams(account)
    if not exams:
        st.info("目前没有属于你的已发布考试。请确认教师已经上传成绩并发布，且你使用的是成绩表中的学号注册。")
        return

    rows = []
    total_records = 0
    subjects: set[str] = set()
    for exam in exams:
        scores = get_scores(exam["exam_id"], account)
        questions = get_questions(exam["exam_id"])
        total_records += len(scores)
        subjects.add(str(exam["subject"]))
        max_score = sum(float(question["full_score"]) for question in questions if question.get("full_score") is not None)
        earned = float(scores["得分"].sum()) if not scores.empty else None
        rows.append(
            {
                "考试": exam["name"],
                "学科": exam["subject"],
                "班级": exam["class_name"],
                "题目得分合计": round(earned, 3) if earned is not None else "暂无成绩",
                "满分": round(max_score, 3) if max_score else "未核对",
                "题目记录数": len(scores),
                "状态": "已发布",
            }
        )
    metric_row(
        [
            ("已发布考试", str(len(exams)), "教师发布后可见"),
            ("成绩记录", str(total_records), "来自真实成绩表"),
            ("涉及学科", str(len(subjects)), "来自考试信息"),
            ("AI 答疑", "已配置" if get_ai_key() else "待配置", "当前浏览器会话"),
        ]
    )
    st.markdown('<div class="section-title">我的考试</div>', unsafe_allow_html=True)
    exam_cols = st.columns(2, gap="large")
    for index, (exam, row) in enumerate(zip(exams, rows)):
        with exam_cols[index % len(exam_cols)]:
            with st.container(border=True):
                st.markdown(f"### {html_escape(str(exam['name']))}")
                st.caption(f"{html_escape(str(exam['subject']))} · {html_escape(str(exam['class_name']))} · 已发布")
                score_col, records_col = st.columns(2)
                with score_col:
                    st.metric("题目得分合计", f"{row['题目得分合计']:.1f}" if isinstance(row["题目得分合计"], (int, float)) else row["题目得分合计"])
                with records_col:
                    st.metric("成绩题目数", row["题目记录数"])
                if isinstance(row["满分"], (int, float)) and row["满分"] > 0:
                    st.progress(min(1.0, max(0.0, float(row["题目得分合计"]) / float(row["满分"]))), text=f"得分率 {float(row['题目得分合计']) / float(row['满分']) * 100:.1f}%")
                else:
                    st.caption("试卷满分待教师核对，暂不计算得分率。")
    with st.expander("查看成绩记录表", expanded=False):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info("进入“试卷分析”查看本次考试报告，进入“逐题答疑”查看题目解析、AI 答疑和错题本。")


def render_student_ai_config_card() -> None:
    """Show student AI setup and connection status on the student home page."""
    with st.container(border=True):
        st.markdown("### 学生 AI 答疑配置")
        st.caption("输入 DeepSeek API Key 后即可使用逐题答疑和 AI 学习分析；密钥仅保存在当前浏览器会话。")
        key_col, model_col, action_col = st.columns([2.2, 1.2, 1.1])
        with key_col:
            st.text_input(
                "DeepSeek API Key",
                type="password",
                key="deepseek_api_key_input",
                placeholder="sk-...",
                help="密钥不会写入项目代码或数据库。",
            )
        with model_col:
            st.text_input("模型名称", key="deepseek_model_input")
        with action_col:
            st.write("")
            if st.button("保存 AI 配置", type="primary", use_container_width=True, key="student_home_save_ai"):
                st.session_state["deepseek_api_key_value"] = str(st.session_state.get("deepseek_api_key_input", "")).strip()
                st.session_state["deepseek_model_value"] = str(st.session_state.get("deepseek_model_input", "")).strip() or "deepseek-flash"
                st.success("AI 配置已保存到本次会话。")
            if st.button("测试连接", use_container_width=True, key="student_home_test_ai"):
                if not get_ai_key():
                    st.warning("请先输入并保存 API Key。")
                else:
                    with st.spinner("正在测试 DeepSeek 连接……"):
                        try:
                            deepseek_chat(
                                [
                                    {"role": "system", "content": "你是连接测试助手。"},
                                    {"role": "user", "content": "只回复：连接成功"},
                                ],
                                max_tokens=16,
                            )
                            st.success(f"DeepSeek 连接成功（{get_ai_model()}）。")
                        except RuntimeError as exc:
                            st.error(str(exc))
        if get_ai_key():
            st.success(f"DeepSeek 已配置 · {get_ai_model()}")
        else:
            st.warning("尚未配置 DeepSeek API Key，配置后即可使用 AI 答疑。")


def render_student_report() -> None:
    account = st.session_state["account"]
    render_header("试卷分析报告", "按考试查看你的得分、班级位置、题目表现和知识点掌握情况。")
    exams = get_student_exams(account)
    scored_exams = [exam for exam in exams if not get_scores(exam["exam_id"], account).empty]
    if not scored_exams:
        st.info("目前没有可分析的成绩。请等待教师导入成绩并发布考试。")
        return
    exam_ids = [exam["exam_id"] for exam in scored_exams]
    exam_id = st.session_state.get("student_report_exam")
    if exam_id not in exam_ids:
        exam_id = exam_ids[0]
        st.session_state["student_report_exam"] = exam_id
    exam_id = st.selectbox(
        "选择考试",
        exam_ids,
        index=exam_ids.index(exam_id),
        key="student_report_exam",
        format_func=lambda value: next(exam["name"] for exam in scored_exams if exam["exam_id"] == value),
    )
    exam = next(exam for exam in scored_exams if exam["exam_id"] == exam_id)
    scores = get_scores(exam_id, account)
    all_scores = get_scores(exam_id)
    questions = get_questions(exam_id)
    question_meta = {question["question_no"]: question for question in questions}
    total_score = float(scores["得分"].sum())
    full_score = sum(float(question["full_score"]) for question in questions if question.get("full_score") is not None)
    rate = total_score / full_score * 100 if full_score else None
    topic_values: dict[str, dict[str, float]] = {}
    for _, score_row in scores.iterrows():
        question = question_meta.get(int(score_row["题号"]), {})
        try:
            item_full_score = float(question.get("full_score"))
        except (TypeError, ValueError):
            continue
        if item_full_score <= 0:
            continue
        topic = question.get("topic") or "待教师核对"
        item = topic_values.setdefault(topic, {"得分": 0.0, "满分": 0.0})
        item["得分"] += float(score_row["得分"])
        item["满分"] += item_full_score
    topic_df = pd.DataFrame(
        [
            {"知识点": topic, "得分率": round(values["得分"] / values["满分"] * 100, 1)}
            for topic, values in topic_values.items()
            if values["满分"] > 0
        ]
    )
    class_totals = all_scores.groupby("学号", as_index=False)["得分"].sum().sort_values("得分", ascending=False)
    rank = int((class_totals["得分"] > total_score).sum()) + 1 if not class_totals.empty else None
    class_average = float(class_totals["得分"].mean()) if not class_totals.empty else 0.0

    metric_row(
        [
            ("本次得分", f"{total_score:.1f}", f"满分 {full_score:.1f}" if full_score else "满分待核对"),
            ("得分率", f"{rate:.1f}%" if rate is not None else "未计算", "基于教师核对后的满分"),
            ("班级排名", f"第 {rank} 名" if rank else "未计算", f"共 {len(class_totals)} 人"),
            ("班级平均", f"{class_average:.1f}", "按学生总分计算"),
        ]
    )

    st.markdown('<div class="section-title">AI 试卷分析与复习建议</div>', unsafe_allow_html=True)
    report_key = f"student_exam_ai_report_{account}_{exam_id}"
    if st.button("生成 / 更新 AI 试卷分析报告", type="primary", key=f"generate_student_exam_report_{exam_id}"):
        if not get_ai_key():
            st.warning("当前会话尚未配置 DeepSeek API Key。")
        else:
            question_evidence = []
            for _, score_row in scores.iterrows():
                question = question_meta.get(int(score_row["题号"]), {})
                try:
                    q_full = float(question.get("full_score"))
                except (TypeError, ValueError):
                    q_full = None
                q_score = float(score_row["得分"])
                question_evidence.append(
                    {
                        "题号": int(score_row["题号"]),
                        "题型": question.get("question_type") or "待核对",
                        "题干": question.get("question_text") or "",
                        "知识点": question.get("topic") or "待教师核对",
                        "得分": q_score,
                        "满分": q_full,
                        "得分率": round(q_score / q_full * 100, 1) if q_full and q_full > 0 else None,
                        "教师核对解析": question.get("analysis") or "",
                    }
                )
            evidence = {
                "考试": exam["name"],
                "学科": exam["subject"],
                "个人总分": total_score,
                "试卷满分": full_score,
                "个人得分率": round(rate, 1) if rate is not None else None,
                "班级名次": rank,
                "知识点得分率": topic_df.to_dict("records"),
                "逐题证据": question_evidence,
            }
            try:
                with st.spinner("AI 正在归纳知识点错漏并制定复习计划……"):
                    report = deepseek_chat(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "你是耐心、严谨的学习诊断老师。先总结学生本卷中证据明确的知识点错漏和可能的解题思路问题，"
                                    "再给出逐项、具体、可执行的复习建议。每个薄弱点都要对应练习方式、频率或数量、检查方法；"
                                    "例如语法薄弱可安排同类题练习、整理易混规则或画时态关系图，但只有数据支持时才提出。"
                                    "根据题干和得分推断时要标为可能，不要把推测当事实，不要编造学生答案或错误原因。"
                                    "按“本卷诊断”“知识点错漏”“具体复习计划”“如何检查进步”分段，用中文回答。"
                                ),
                            },
                            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
                        ],
                        max_tokens=2200,
                    )
                st.session_state[report_key] = report
            except RuntimeError as exc:
                st.error(str(exc))
    if st.session_state.get(report_key):
        render_ai_result(st.session_state[report_key])

    if not topic_df.empty:
        topic_df = topic_df.sort_values("得分率", ascending=True)
        st.markdown('<div class="section-title">本次得分率</div>', unsafe_allow_html=True)
        if go is not None:
            figure = go.Figure(
                go.Bar(
                    x=topic_df["得分率"],
                    y=topic_df["知识点"],
                    orientation="h",
                    text=[f"{value:.1f}%" for value in topic_df["得分率"]],
                    textposition="auto",
                    marker_color="#5747e8",
                )
            )
            figure.update_layout(
                height=max(280, 48 * len(topic_df)),
                margin=dict(l=20, r=30, t=12, b=30),
                xaxis=dict(title="得分率（%）", range=[0, 100]),
                yaxis=dict(title="", automargin=True),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#555963"),
                transition=dict(duration=350),
            )
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("当前运行环境未安装图表组件，得分率明细见下表。")
        st.table(topic_df.assign(得分率=topic_df["得分率"].map(lambda value: f"{value:.1f}%")))
    else:
        st.info("当前试卷没有完整的知识点和满分信息，暂时无法绘制得分率图。")

    scored_below_full = 0
    comparable_questions = 0
    for _, score_row in scores.iterrows():
        question = question_meta.get(int(score_row["题号"]), {})
        try:
            item_full = float(question.get("full_score"))
        except (TypeError, ValueError):
            continue
        if item_full > 0:
            comparable_questions += 1
            scored_below_full += float(score_row["得分"]) < item_full
    st.markdown('<div class="section-title">本次试卷表现摘要</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            f"本次成绩表中有 **{len(scores)}** 条个人题目得分记录；其中 **{scored_below_full} / {comparable_questions}** 道已核对满分的题目得分低于满分。"
        )
        if not topic_df.empty:
            weakest = topic_df.iloc[0]
            strongest = topic_df.iloc[-1]
            st.write(
                f"知识点得分率最低的是 **{weakest['知识点']}（{weakest['得分率']:.1f}%）**；"
                f"最高的是 **{strongest['知识点']}（{strongest['得分率']:.1f}%）**。下方 AI 报告会据这些题目和知识点证据给出复习建议。"
            )
        else:
            st.caption("当前没有足够的教师核对满分字段进行知识点比较。")

    detail_rows = []
    for _, score_row in scores.iterrows():
        question = question_meta.get(int(score_row["题号"]), {})
        item_full_score = question.get("full_score")
        try:
            item_full_score = float(item_full_score) if item_full_score is not None else None
        except (TypeError, ValueError):
            item_full_score = None
        item_score = float(score_row["得分"])
        detail_rows.append(
            {
                "题号": int(score_row["题号"]),
                "题型": question.get("question_type") or "待核对",
                "得分": item_score,
                "满分": item_full_score if item_full_score is not None else "未识别",
                "得分率": f"{item_score / item_full_score * 100:.1f}%" if item_full_score else "未计算",
                "知识点": question.get("topic") or "待核对",
                "状态": "错题" if item_full_score and item_score < item_full_score else "已答对" if item_full_score else "待核对",
            }
        )
    st.markdown('<div class="section-title">逐题得分明细</div>', unsafe_allow_html=True)
    st.table(pd.DataFrame(detail_rows))

def render_student_profile() -> None:
    account = st.session_state["account"]
    render_header("学生档案", "档案只根据当前学号在教师发布数据中的真实记录生成，不预置姓名、成绩或评价。")
    exams = get_student_exams(account)
    with db_connect() as conn:
        name_row = conn.execute(
            "SELECT student_name FROM scores WHERE student_id = ? ORDER BY rowid LIMIT 1",
            (account,),
        ).fetchone()
    display_name = name_row["student_name"] if name_row else "待成绩表识别"
    all_score_frames = [get_scores(exam["exam_id"], account) for exam in exams]
    all_scores = pd.concat([frame for frame in all_score_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in all_score_frames) else pd.DataFrame()
    subject_count = len({exam["subject"] for exam in exams})
    exam_count = len(exams)
    average_score = float(all_scores["得分"].mean()) if not all_scores.empty else None
    metric_row(
        [
            ("学号", account, "当前登录账号"),
            ("姓名", str(display_name), "来自成绩表"),
            ("已发布考试", str(exam_count), "教师发布后可见"),
            ("涉及学科", str(subject_count), "来自考试信息"),
            ("题目平均得分", f"{average_score:.2f}" if average_score is not None else "暂无成绩", "按题目记录计算"),
        ]
    )
    if not exams:
        st.info("目前没有考试记录，学生档案会在教师发布成绩后自动形成。")
        return
    rows = []
    for exam in exams:
        exam_scores = get_scores(exam["exam_id"], account)
        if exam_scores.empty:
            continue
        rows.append(
            {
                "考试": exam["name"],
                "学科": exam["subject"],
                "班级": exam["class_name"],
                "总得分": round(float(exam_scores["得分"].sum()), 2),
                "题目记录": len(exam_scores),
                "状态": "已发布",
            }
        )

    profile_topic_totals: dict[str, dict[str, float]] = {}
    for exam in exams:
        exam_scores = get_scores(exam["exam_id"], account)
        question_meta = {question["question_no"]: question for question in get_questions(exam["exam_id"])}
        for _, score_row in exam_scores.iterrows():
            question = question_meta.get(int(score_row["题号"]), {})
            try:
                item_full_score = float(question.get("full_score"))
            except (TypeError, ValueError):
                continue
            if item_full_score <= 0:
                continue
            topic = f"{exam['subject']} · {question.get('topic') or '待教师核对'}"
            totals = profile_topic_totals.setdefault(topic, {"earned": 0.0, "possible": 0.0})
            totals["earned"] += float(score_row["得分"])
            totals["possible"] += item_full_score

    st.markdown('<div class="section-title">学习表现轨迹</div>', unsafe_allow_html=True)
    trajectory_col, ability_col = st.columns([1.1, 1], gap="large")
    with trajectory_col:
        with st.container(border=True):
            st.markdown("### 成绩轨迹")
            if rows:
                chronological_rows = rows[::-1]
                if go is not None:
                    figure = go.Figure(
                        go.Scatter(
                            x=[row["考试"] for row in chronological_rows],
                            y=[row["总得分"] for row in chronological_rows],
                            mode="lines+markers",
                            line=dict(color="#5747e8", width=2.5),
                            marker=dict(size=8, color="#5747e8"),
                        )
                    )
                    figure.update_layout(
                        height=320,
                        margin=dict(l=15, r=15, t=15, b=60),
                        xaxis_title="已发布考试（按创建时间）",
                        yaxis_title="总得分",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#555963"),
                        showlegend=False,
                    )
                    st.plotly_chart(figure, use_container_width=True)
                else:
                    st.bar_chart(pd.DataFrame(chronological_rows).set_index("考试")[["总得分"]])
            else:
                st.info("有成绩记录后会显示考试得分轨迹。")
    with ability_col:
        with st.container(border=True):
            st.markdown("### 知识点得分表现")
            profile_topic_df = pd.DataFrame(
                [
                    {"知识点": topic, "得分率": values["earned"] / values["possible"] * 100}
                    for topic, values in profile_topic_totals.items()
                    if values["possible"] > 0
                ]
            )
            if len(profile_topic_df) >= 3 and go is not None:
                profile_topic_df = profile_topic_df.sort_values("得分率").head(8)
                labels = profile_topic_df["知识点"].astype(str).tolist()
                values = profile_topic_df["得分率"].astype(float).tolist()
                figure = go.Figure(
                    go.Scatterpolar(
                        r=values + values[:1],
                        theta=labels + labels[:1],
                        fill="toself",
                        fillcolor="rgba(87,71,232,.14)",
                        line=dict(color="#5747e8", width=2),
                        name="个人得分率",
                    )
                )
                figure.update_layout(
                    height=320,
                    margin=dict(l=30, r=30, t=20, b=20),
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#555963"),
                )
                st.plotly_chart(figure, use_container_width=True)
            elif not profile_topic_df.empty:
                st.dataframe(
                    profile_topic_df.sort_values("得分率").assign(得分率=lambda frame: frame["得分率"].map(lambda value: f"{value:.1f}%")),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("题目满分和知识点经教师核对后，这里会显示个人知识点表现。")

    st.markdown('<div class="section-title">考试记录</div>', unsafe_allow_html=True)
    if rows:
        st.table(pd.DataFrame(rows))
    else:
        st.info("目前没有可展示的成绩记录。")

    st.markdown('<div class="section-title">AI 学习诊断摘要</div>', unsafe_allow_html=True)
    wrong_df = get_student_wrong_questions(account)
    topic_counts = (
        wrong_df.groupby(["学科", "知识点"], as_index=False).size().rename(columns={"size": "错题数"})
        if not wrong_df.empty
        else pd.DataFrame(columns=["学科", "知识点", "错题数"])
    )
    profile_key = f"student_profile_ai_diagnosis_{account}"
    if st.button("生成 / 更新 AI 诊断摘要", type="primary", key="generate_student_profile_diagnosis"):
        if not get_ai_key():
            st.warning("当前会话尚未配置 DeepSeek API Key。")
        elif all_scores.empty:
            st.info("目前没有成绩数据，暂时无法生成有依据的 AI 诊断。")
        else:
            evidence = {
                "考试记录": rows,
                "题目平均得分": round(average_score, 2) if average_score is not None else None,
                "各学科知识点错题数": topic_counts.to_dict("records"),
                "最近错题证据": wrong_df[["学科", "考试", "题号", "题型", "题干", "得分率", "知识点"]].head(30).to_dict("records") if not wrong_df.empty else [],
            }
            try:
                with st.spinner("AI 正在整理学习表现和诊断摘要……"):
                    diagnosis = deepseek_chat(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "你是严谨的学习档案分析老师。根据提供的成绩和错题证据，概括稳定优势、反复出现的知识漏洞，"
                                    "并在证据允许时分析可能的思维或解题习惯。必须区分事实与推测，不得把未提供的学生作答过程说成事实。"
                                    "给出3条以内具体可执行的后续学习行动和检查指标。使用中文，简洁但有依据。"
                                ),
                            },
                            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
                        ],
                        max_tokens=1600,
                    )
                st.session_state[profile_key] = diagnosis
            except RuntimeError as exc:
                st.error(str(exc))
    if st.session_state.get(profile_key):
        render_ai_result(st.session_state[profile_key])
    elif not exams:
        st.info("有已发布考试数据后，AI 才能根据真实成绩生成诊断摘要。")


def render_student_wrong_book_section(account: str) -> None:
    """Render the wrong-book tools inside the question-help page."""
    st.divider()
    st.markdown("## 错题本（已整合到逐题答疑）")
    st.caption("这里汇总所有已发布考试的错题；表格会完整展开，页面整体滚动，不使用表格内部滚动。")
    wrong_df = get_student_wrong_questions(account)
    if wrong_df.empty:
        st.info("目前没有错题记录。教师发布成绩后，得分低于满分的题目会自动进入这里。")
        return

    subject_summary = wrong_df.groupby("学科", as_index=False).size().rename(columns={"size": "错题数"})
    st.markdown('<div class="section-title">各科目错题汇总</div>', unsafe_allow_html=True)
    st.table(subject_summary)

    filter_col, search_col = st.columns([1, 2])
    subjects = ["全部科目"] + sorted(str(value) for value in wrong_df["学科"].dropna().unique())
    with filter_col:
        selected_subject = st.selectbox("筛选学科", subjects, key="integrated_wrong_book_subject")
    with search_col:
        keyword = st.text_input("搜索题干 / 知识点 / 考试名称", key="integrated_wrong_book_keyword")
    filtered = wrong_df if selected_subject == "全部科目" else wrong_df[wrong_df["学科"] == selected_subject]
    if keyword.strip():
        pattern = re.escape(keyword.strip())
        mask = (
            filtered["题干"].astype(str).str.contains(pattern, case=False, na=False, regex=True)
            | filtered["知识点"].astype(str).str.contains(pattern, case=False, na=False, regex=True)
            | filtered["考试"].astype(str).str.contains(pattern, case=False, na=False, regex=True)
        )
        filtered = filtered[mask]
    metric_row(
        [
            ("筛选后错题", str(len(filtered)), "按题目得分判断"),
            ("涉及考试", str(filtered["考试"].nunique()), "已发布考试"),
            ("涉及知识点", str(filtered["知识点"].nunique()), "来自教师核对内容"),
        ]
    )
    table_columns = ["学科", "考试", "题号", "题型", "题干", "我的得分", "满分", "得分率", "知识点"]
    st.table(filtered[table_columns])

    st.markdown('<div class="section-title">错题详情（默认展开）</div>', unsafe_allow_html=True)
    for row in filtered.to_dict("records"):
        with st.expander(f"{row['学科']} · {row['考试']} · 第 {row['题号']} 题 · {row['知识点']}", expanded=True):
            st.write(row["题干"])
            st.caption(f"我的得分：{row['我的得分']} / {row['满分']} · 得分率：{row['得分率']}")
            st.write(f"标准答案：{row['标准答案']}")
            st.write(f"解析：{row['解析']}")

    export_title = f"EduTrack 错题本 · {selected_subject}"
    safe_subject = re.sub(r'[\\/:*?"<>|]+', "_", selected_subject)
    export_version = st.radio("导出版本", ["带答案和解析", "不带答案"], horizontal=True, key="integrated_wrong_book_export")
    include_answers = export_version == "带答案和解析"
    version_name = "带答案" if include_answers else "不带答案"
    st.download_button(
        f"导出错题汇总文档（{version_name}）",
        data=build_wrong_book_docx(filtered.to_dict("records"), f"{export_title} · {version_name}", include_answers),
        file_name=f"EduTrack-错题本-{safe_subject}-{version_name}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary",
        key="integrated_wrong_book_download",
    )


def render_student_questions() -> None:
    account = st.session_state["account"]
    render_header("逐题答疑", "逐题查看得分、标准答案和解析，并在同一页面筛选、复习和导出错题。")
    exams = get_student_exams(account)
    if not exams:
        st.info("目前没有属于你的已发布考试。")
        return
    exam_ids = [exam["exam_id"] for exam in exams]
    exam_id = st.selectbox("选择考试", exam_ids, format_func=lambda value: next(exam["name"] for exam in exams if exam["exam_id"] == value))
    questions = get_questions(exam_id)
    if not questions:
        st.info("该考试没有识别到题目。")
        render_student_wrong_book_section(account)
        return
    scores = get_scores(exam_id, account)
    score_map = {int(row["题号"]): float(row["得分"]) for _, row in scores.iterrows()}
    question_map = {question["question_no"]: question for question in questions}
    selected_key = f"student_selected_question_{exam_id}"
    selected_no = st.session_state.get(selected_key, questions[0]["question_no"])
    if selected_no not in question_map:
        selected_no = questions[0]["question_no"]
        st.session_state[selected_key] = selected_no

    def is_wrong(question: dict[str, Any]) -> bool:
        score = score_map.get(question["question_no"])
        full_score = question.get("full_score")
        try:
            full_score_value = float(full_score) if full_score is not None else None
        except (TypeError, ValueError):
            full_score_value = None
        return score <= 0 if score is not None and not full_score_value else bool(
            score is not None and full_score_value and score < full_score_value
        )

    nav_col, content_col = st.columns([1.3, 3.7], gap="large")
    with nav_col:
        st.markdown('<div class="section-title">题目导航</div>', unsafe_allow_html=True)
        st.caption(f"共 {len(questions)} 题 · 已答 {len(score_map)} 题")
        filter_mode = st.radio("筛选题目", ["全部题目", "只看错题"], key=f"question_filter_{exam_id}")
        visible_questions = [question for question in questions if filter_mode == "全部题目" or is_wrong(question)]
        if not visible_questions:
            st.info("当前考试没有可显示的错题。")
        else:
            if selected_no not in {question["question_no"] for question in visible_questions}:
                selected_no = visible_questions[0]["question_no"]
                st.session_state[selected_key] = selected_no
            for start in range(0, len(visible_questions), 4):
                button_cols = st.columns(4)
                for button_col, question in zip(button_cols, visible_questions[start : start + 4]):
                    question_no = question["question_no"]
                    if question_no == selected_no:
                        selected_class = " selected"
                    else:
                        selected_class = " wrong" if is_wrong(question) else ""
                    button_col.markdown(
                        f'<div class="question-number-card{selected_class}">第 {question_no} 题</div>',
                        unsafe_allow_html=True,
                    )
                    if button_col.button(
                        "打开",
                        key=f"student_question_nav_{exam_id}_{question_no}",
                        type="primary" if question_no == selected_no else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state[selected_key] = question_no
                        st.rerun()
        st.caption("错：得分低于满分；对：满分；未答：没有成绩记录。")

    if not visible_questions:
        render_student_wrong_book_section(account)
        return

    # The selected question may be missing after a filter change or an older
    # session state restore. Recover to the first visible question safely.
    if st.session_state.get(selected_key) not in question_map:
        selected_no = visible_questions[0]["question_no"]
        st.session_state[selected_key] = selected_no

    question_no = st.session_state.get(selected_key, visible_questions[0]["question_no"])
    question = question_map[question_no]
    student_score = score_map.get(question_no)
    with content_col:
        st.markdown(f"## 第 {question_no} 题 · {question['question_type']}")
        detail_col, score_col = st.columns([3.4, 1.1], gap="large")
        with detail_col:
            st.markdown('<div class="section-title">题目原文</div>', unsafe_allow_html=True)
            st.write(question["question_text"])
            if question.get("options"):
                st.markdown("\n".join(f"- {option}" for option in question["options"]))
            if question.get("topic"):
                st.caption(f"知识点：{question['topic']}")
        with score_col:
            st.metric("我的得分", student_score if student_score is not None else "未上传")
            if question.get("full_score"):
                st.caption(f"满分：{question['full_score']}")
            st.caption(f"状态：{'错题' if is_wrong(question) else '已答对' if student_score is not None else '未答'}")
        st.divider()
        if question.get("answer"):
            st.info("标准答案来自教师核对后的试卷内容。")
            st.write(f"标准答案：{question['answer']}")
        if question.get("analysis"):
            st.markdown('<div class="section-title">题目解析</div>', unsafe_allow_html=True)
            st.write(question["analysis"])
        with st.container(border=True):
            render_ai_box(exam_id, question, student_score)

    render_student_wrong_book_section(account)


def render_student_wrong_book() -> None:
    """Compatibility wrapper for sessions that still hold the old page name."""
    render_header("逐题答疑", "错题本已经整合到逐题答疑页面。")
    render_student_wrong_book_section(st.session_state["account"])


def render_student_answer_sheets() -> None:
    account = st.session_state["account"]
    render_header("我的答题卡", "这里仅显示教师上传、且学号与当前账号完全匹配的答题卡。")
    exams = get_student_exams(account)
    if not exams:
        st.info("目前没有属于你的已发布考试或答题卡。")
        return

    exam_ids = [exam["exam_id"] for exam in exams]
    exam_id = st.selectbox(
        "选择考试",
        exam_ids,
        format_func=lambda value: next(exam["name"] for exam in exams if exam["exam_id"] == value),
    )
    sheets = get_answer_sheets(exam_id, account)
    if not sheets:
        st.info("该考试暂时没有上传属于你的答题卡。")
        return

    for sheet in sheets:
        st.markdown(f"### {sheet['filename']}")
        st.caption(f"学号：{sheet['student_id']} · 上传时间：{sheet['created_at']}")
        st.download_button(
            "下载答题卡",
            data=sheet["data"],
            file_name=sheet["filename"],
            mime=sheet["mime_type"],
            key=f"download_answer_sheet_{sheet['exam_id']}_{sheet['student_id']}",
        )
        if sheet["mime_type"].startswith("image/"):
            st.image(sheet["data"], use_container_width=True)
        elif sheet["mime_type"] == "application/pdf":
            if hasattr(st, "pdf"):
                st.pdf(sheet["data"])
            else:
                encoded = base64.b64encode(sheet["data"]).decode("ascii")
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="720" type="application/pdf"></iframe>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("该文件类型暂不支持网页内预览，请点击“下载答题卡”查看。")


def sidebar() -> tuple[str, str]:
    role = st.session_state["role"]
    st.sidebar.markdown(f'{BRAND_MARK}<div class="muted">真实上传 · AI 识别 · 学生答疑</div>', unsafe_allow_html=True)
    st.sidebar.divider()
    st.sidebar.caption(f"当前账号：{st.session_state['account']} · 角色：{role}")
    st.sidebar.caption("数据库没有预置考试、题目或成绩。")
    if role == "学生端":
        pages = ["我的考试", "我的答题卡", "逐题答疑", "试卷分析", "学生档案"]
    else:
        pages = ["教师工作台", "上传试卷", "题目核对", "导入成绩", "上传答题卡", "班级报告", "教学干预"]
    page = st.sidebar.radio("页面导航", pages)
    st.sidebar.divider()
    if get_ai_key():
        st.sidebar.success(f"DeepSeek 已配置 · {get_ai_model()}")
    else:
        st.sidebar.warning("DeepSeek 未配置")
    if st.sidebar.button("退出登录", use_container_width=True):
        for key in [
            "authenticated",
            "account",
            "role",
            "ai_messages",
            "deepseek_api_key_input",
            "deepseek_api_key_value",
            "deepseek_model_input",
            "deepseek_model_value",
        ]:
            st.session_state.pop(key, None)
        st.rerun()
    return role, page


def main() -> None:
    init_db()
    inject_css()
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("ai_messages", {})
    st.session_state.setdefault("deepseek_model_input", "deepseek-flash")
    st.session_state.setdefault("deepseek_model_value", "")
    if not st.session_state["authenticated"]:
        render_login_page()
        return

    role, page = sidebar()
    if role == "学生端":
        if page == "我的考试":
            render_student_dashboard()
        elif page == "我的答题卡":
            render_student_answer_sheets()
        elif page == "试卷分析":
            render_student_report()
        elif page == "学生档案":
            render_student_profile()
        elif page == "错题本":
            render_student_wrong_book()
        else:
            render_student_questions()
    else:
        if page == "教师工作台":
            render_teacher_dashboard()
        elif page == "上传试卷":
            render_create_exam()
        elif page == "题目核对":
            render_question_review()
        elif page == "导入成绩":
            render_import_scores()
        elif page == "上传答题卡":
            render_upload_answer_sheets()
        elif page == "教学干预":
            render_teacher_intervention()
        else:
            render_class_report()


if __name__ == "__main__":
    main()
