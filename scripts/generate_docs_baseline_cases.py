from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from agentic_rag.factory import create_collection_service


QUESTION_PREFIXES = [
    "{topic}有哪些要求？",
    "{topic}的办理流程是什么？",
    "{topic}需要注意什么？",
    "{topic}如何执行？",
    "{topic}的适用条件是什么？",
    "{topic}涉及哪些材料或留痕要求？",
    "{topic}在通知中是如何规定的？",
    "{topic}相关的风险控制要求是什么？",
]

STOPWORDS = {
    "公司",
    "客户",
    "业务",
    "进行",
    "相关",
    "融资融券",
    "信用",
    "证券",
    "通知",
    "要求",
    "系统",
    "账户",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate docs baseline eval cases from indexed chunks.")
    parser.add_argument("--collection-name", default="docs-baseline-20260620")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", default="storage/eval/docs_baseline_20260620_cases.jsonl")
    args = parser.parse_args()

    collection = _collection_by_name(args.collection_name)
    rows = _load_chunks(Path(collection.keyword_index_path))
    selected = _select_rows(rows, args.count)
    cases = [_case_from_row(row, collection.collection_id, index) for index, row in enumerate(selected, start=1)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"path": str(out_path), "case_count": len(cases)}, ensure_ascii=False, indent=2))


def _collection_by_name(name: str):
    matches = [
        collection
        for collection in create_collection_service().list_collections().collections
        if collection.name == name
    ]
    if not matches:
        raise SystemExit(f"Collection not found: {name}")
    return matches[0]


def _load_chunks(keyword_index_path: Path) -> list[dict]:
    with sqlite3.connect(str(keyword_index_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, source, file_id, content, metadata_json, chunk_index, page_start, page_end
            FROM chunks
            WHERE length(trim(content)) >= 80
            ORDER BY source, chunk_index
            """
        ).fetchall()
    return [
        {
            "id": str(row["id"]),
            "source": str(row["source"]),
            "file_id": str(row["file_id"]),
            "content": _clean_text(str(row["content"])),
            "metadata": json.loads(str(row["metadata_json"])),
            "chunk_index": row["chunk_index"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
        }
        for row in rows
    ]


def _select_rows(rows: list[dict], count: int) -> list[dict]:
    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_file[row["file_id"]].append(row)

    selected: list[dict] = []
    for file_rows in by_file.values():
        selected.extend(_spread(file_rows, min(6, len(file_rows))))

    remaining = [row for row in rows if row["id"] not in {item["id"] for item in selected}]
    selected.extend(_spread(remaining, max(0, count - len(selected))))
    return selected[:count]


def _spread(rows: list[dict], count: int) -> list[dict]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return rows
    indexes = sorted({round(index * (len(rows) - 1) / (count - 1)) for index in range(count)})
    return [rows[index] for index in indexes[:count]]


def _case_from_row(row: dict, collection_id: str, index: int) -> dict:
    metadata = row["metadata"]
    file_name = str(metadata.get("file_name") or Path(row["source"]).name)
    faq = _faq_pair(row["content"])
    if faq is not None:
        question, answer = faq
        topic = _topic(question, file_name)
    else:
        topic = _topic(row["content"], file_name)
        question = QUESTION_PREFIXES[(index - 1) % len(QUESTION_PREFIXES)].format(topic=topic)
        answer = _answer(row["content"])
    keywords = _keywords(question, topic, answer)

    return {
        "id": f"docs-baseline-20260620-{index:03d}",
        "query": question,
        "collection_id": collection_id,
        "expected_sources": [file_name],
        "expected_file_ids": [row["file_id"]],
        "expected_keywords": keywords,
        "top_k": 8,
        "reference_answer": answer,
        "source_file": file_name,
        "source": row["source"],
        "chunk_id": row["id"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
    }


def _topic(content: str, file_name: str) -> str:
    candidates = []
    candidates.extend(re.findall(r"《([^》]{4,40})》", content))
    candidates.extend(re.findall(r"([\u4e00-\u9fffA-Za-z0-9]{4,24}(?:流程|要求|标准|条件|规则|模板|通知|管理|监控|平仓|展期|授信|留痕|合同|担保物))", content))
    candidates.extend(re.findall(r"([\u4e00-\u9fffA-Za-z0-9]{4,24}(?:流程|要求|标准|条件|规则|模板|通知|管理|监控|平仓|展期|授信|留痕|合同|担保物))", file_name))
    for candidate in candidates:
        candidate = candidate.strip(" ，。、；;:：()（）")
        if 4 <= len(candidate) <= 28 and candidate not in STOPWORDS:
            return candidate
    return Path(file_name).stem[:24]


def _answer(content: str) -> str:
    sentences = re.split(r"(?<=[。；;])\s*", content)
    answer = "".join(sentence for sentence in sentences[:3]).strip()
    if len(answer) < 80:
        answer = content[:260].strip()
    return answer[:420]


def _faq_pair(content: str) -> tuple[str, str] | None:
    match = re.search(r"问题\s*\d+\s*[：:]\s*(.{6,80}?)[？?]\s*回答(?:\s*\d+)?[：:]\s*(.{20,500})", content)
    if not match:
        return None
    question = match.group(1).strip() + "？"
    answer = match.group(2).strip()
    answer = re.split(r"\s*问题\s*\d+\s*[：:]", answer)[0].strip()
    return question, answer[:420]


def _keywords(question: str, topic: str, answer: str) -> list[str]:
    values: list[str] = []
    for text in [question, topic, answer]:
        values.extend(re.findall(r"《[^》]{4,40}》", text))
        values.extend(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,12}", text))
    result = []
    for value in values:
        value = value.strip(" ，。、；;:：()（）")
        if value and value not in STOPWORDS and value not in result:
            result.append(value)
        if len(result) >= 1:
            break
    return result


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    main()
