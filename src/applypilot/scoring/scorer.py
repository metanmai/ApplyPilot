"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone

from applypilot.config import RESUME_PATH, load_profile
from applypilot.database import get_connection, get_jobs_by_stage
from applypilot.llm import get_client

log = logging.getLogger(__name__)


# ── Scoring Prompt ────────────────────────────────────────────────────────

SCORE_PROMPT = """You are a job fit evaluator. Given a candidate's resume and a job description, score how well the candidate fits the role.

SCORING CRITERIA:
- 9-10: Perfect match. Candidate has direct experience in nearly all required skills and qualifications.
- 7-8: Strong match. Candidate has most required skills, minor gaps easily bridged.
- 5-6: Moderate match. Candidate has relevant skills and can grow into the role.
- 3-4: Weak match. Some skills overlap, but significant gaps exist.
- 1-2: Poor match. Completely different field or fundamental misalignment.

IMPORTANT FACTORS - BE LENIENT:
- Weight TECHNICAL SKILLS and TECH STACK above all else (languages, frameworks, tools)
- Job requirements are often wish lists, not hard requirements — score based on core skills, not nice-to-haves
- 2+ years of professional experience is VIABLE for mid-level roles if skills match
- Value DEMONSTRATED IMPACT over years on the job (e.g., cost savings, performance improvements, ownership)
- Consider transferable experience (automation, scripting, API work, distributed systems)
- Factor in the candidate's project experience and internship work
- Be LENIENT about experience level gaps — focus on whether the candidate CAN do the job, not whether they check every box
- High-performers often grow faster than their years suggest — recognize potential

RESPOND IN EXACTLY THIS FORMAT (no other text):
SCORE: [1-10]
KEYWORDS: [comma-separated ATS keywords from the job description that match or could match the candidate]
REASONING: [2-3 sentences explaining the score]"""


def _parse_score_response(response: str) -> dict:
    """Parse the LLM's score response into structured data.

    Args:
        response: Raw LLM response text.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    score = 0
    keywords = ""
    reasoning = response

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = int(re.search(r"\d+", line).group())
                score = max(1, min(10, score))
            except (AttributeError, ValueError):
                score = 0
        elif line.startswith("KEYWORDS:"):
            keywords = line.replace("KEYWORDS:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return {"score": score, "keywords": keywords, "reasoning": reasoning}


def score_job(resume_text: str, job: dict) -> dict:
    """Score a single job against the resume.

    Args:
        resume_text: The candidate's full resume text.
        job: Job dict with keys: title, site, location, full_description.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    messages = [
        {"role": "system", "content": SCORE_PROMPT},
        {"role": "user", "content": f"RESUME:\n{resume_text}\n\n---\n\nJOB POSTING:\n{job_text}"},
    ]

    try:
        client = get_client()
        response = client.chat(messages, max_tokens=2048, temperature=0.2)
        return _parse_score_response(response)
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        return {"score": 0, "keywords": "", "reasoning": f"LLM error: {e}"}


def score_and_commit(job: dict) -> dict:
    """Score a single job and commit immediately.

    Args:
        job: Job dict with keys: title, site, location, full_description, url.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    # Get resume text
    resume_text = RESUME_PATH.read_text(encoding="utf-8")

    # Score using existing logic
    result = score_job(resume_text, job)

    # Commit immediately
    conn = get_connection()
    conn.execute(
        "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ? WHERE url = ?",
        (result['score'], f"{result['keywords']}\n{result['reasoning']}", datetime.now(timezone.utc).isoformat(), job['url'])
    )
    conn.commit()

    return result


def run_scoring(limit: int = 0, rescore: bool = False) -> dict:
    """Score unscored jobs that have full descriptions.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list}
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    if rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score", limit=limit)

    if not jobs:
        log.info("No unscored jobs with descriptions found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": []}

    # Convert sqlite3.Row to dicts if needed
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    log.info("Scoring %d jobs sequentially...", len(jobs))
    t0 = time.time()
    completed = 0
    errors = 0
    results: list[dict] = []

    for job in jobs:
        result = score_job(resume_text, job)
        result["url"] = job["url"]
        completed += 1

        if result["score"] == 0:
            errors += 1

        results.append(result)

        # Clean, readable log format with job details
        score_emoji = {9: "🟢", 8: "🟢", 7: "✅", 6: "📊", 5: "📊", 4: "🟡", 3: "🟡", 2: "🔴", 1: "🔴", 0: "⚫"}
        emoji = score_emoji.get(result["score"], "⚪")

        log.info("")
        log.info("=" * 70)
        log.info("│ [%d/%d] score=%d %s", completed, len(jobs), result["score"], emoji)
        log.info("│ Title:      %s", job.get("title", "?"))
        log.info("│ Company:    %s", job.get("site", "Unknown"))
        location = job.get("location", "N/A")
        if location and location != "N/A":
            log.info("│ Location:   %s", location[:50])
        if result["keywords"]:
            log.info("│ Keywords:   %s", result["keywords"])
        log.info("│ Reasoning:  %s", result["reasoning"][:150])
        if len(result["reasoning"]) > 150:
            log.info("│             %s", result["reasoning"][150:300])
        log.info("=" * 70)

    # Write scores to DB
    now = datetime.now(timezone.utc).isoformat()
    for r in results:
        conn.execute(
            "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ? WHERE url = ?",
            (r["score"], f"{r['keywords']}\n{r['reasoning']}", now, r["url"]),
        )
    conn.commit()

    elapsed = time.time() - t0
    log.info("Done: %d scored in %.1fs (%.1f jobs/sec)", len(results), elapsed, len(results) / elapsed if elapsed > 0 else 0)

    # Score distribution
    dist = conn.execute("""
        SELECT fit_score, COUNT(*) FROM jobs
        WHERE fit_score IS NOT NULL
        GROUP BY fit_score ORDER BY fit_score DESC
    """).fetchall()
    distribution = [(row[0], row[1]) for row in dist]

    return {
        "scored": len(results),
        "errors": errors,
        "elapsed": elapsed,
        "distribution": distribution,
    }
