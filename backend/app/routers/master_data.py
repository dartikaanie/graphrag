from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    AnswerDetail,
    AnswerListItem,
    AnswerListResponse,
    PageMeta,
    QuestionDetail,
    QuestionListItem,
    QuestionListResponse,
    StatsSummary,
    TagDetail,
    TagListItem,
    TagListResponse,
)
from app.services import master_data_service as svc

router = APIRouter(prefix="/api")


def _preview(text: str | None, length: int = 240) -> str:
    if not text:
        return ""
    return text if len(text) <= length else text[:length].rstrip() + "..."


@router.get("/questions", response_model=QuestionListResponse)
def get_questions(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    search: str | None = None,
    tag: str | None = None,
):
    rows, total = svc.list_questions(page, page_size, search, tag)
    items = [
        QuestionListItem(
            id=r["id"],
            title=r["title"] or "",
            domain_tag=r.get("domain_tag"),
            score=r.get("score"),
            view_count=r.get("view_count"),
            answer_count=r.get("answer_count"),
        )
        for r in rows
    ]
    meta = PageMeta(page=page, page_size=page_size, total=total, total_pages=svc.total_pages(total, page_size))
    return QuestionListResponse(items=items, meta=meta)


@router.get("/questions/{question_id}", response_model=QuestionDetail)
def get_question(question_id: int):
    detail = svc.get_question_detail(question_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Question not found")
    return detail


@router.get("/questions/{question_id}/answers")
def get_question_answers(question_id: int):
    return svc.get_question_answers(question_id)


@router.get("/questions/{question_id}/accepted-answer")
def get_question_accepted_answer(question_id: int):
    answer = svc.get_accepted_answer(question_id)
    if not answer:
        raise HTTPException(status_code=404, detail="No accepted answer for this question")
    return answer


@router.get("/answers", response_model=AnswerListResponse)
def get_answers(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    search: str | None = None,
):
    rows, total = svc.list_answers(page, page_size, search)
    items = [
        AnswerListItem(
            id=r["id"],
            body_preview=_preview(r.get("body")),
            score=r.get("score"),
            is_accepted=r.get("is_accepted"),
            question_id=r.get("question_id"),
        )
        for r in rows
    ]
    meta = PageMeta(page=page, page_size=page_size, total=total, total_pages=svc.total_pages(total, page_size))
    return AnswerListResponse(items=items, meta=meta)


@router.get("/answers/{answer_id}", response_model=AnswerDetail)
def get_answer(answer_id: int):
    detail = svc.get_answer_detail(answer_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Answer not found")
    return detail


@router.get("/tags", response_model=TagListResponse)
def get_tags(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    search: str | None = None,
):
    rows, total = svc.list_tags(page, page_size, search)
    items = [TagListItem(name=r["name"], question_count=r.get("question_count")) for r in rows]
    meta = PageMeta(page=page, page_size=page_size, total=total, total_pages=svc.total_pages(total, page_size))
    return TagListResponse(items=items, meta=meta)


@router.get("/tags/{tag_name}", response_model=TagDetail)
def get_tag(tag_name: str):
    detail = svc.get_tag_detail(tag_name)
    if not detail:
        raise HTTPException(status_code=404, detail="Tag not found")
    questions = [
        QuestionListItem(
            id=q["id"],
            title=q.get("title") or "",
            domain_tag=q.get("domainTag"),
            score=q.get("score"),
            view_count=q.get("viewCount"),
        )
        for q in detail["questions"]
    ]
    return TagDetail(name=detail["name"], question_count=detail["question_count"], questions=questions)


@router.get("/stats/summary", response_model=StatsSummary)
def get_stats_summary():
    return svc.get_stats_summary()
