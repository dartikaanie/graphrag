"""Pydantic response models for Phase 1 (master data + stats)."""

from typing import Any

from pydantic import BaseModel


class PageMeta(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int


class QuestionListItem(BaseModel):
    id: int
    title: str
    domain_tag: str | None = None
    score: int | None = None
    view_count: int | None = None
    tags: str | None = None
    answer_count: int | None = None


class QuestionListResponse(BaseModel):
    items: list[QuestionListItem]
    meta: PageMeta


class QuestionDetail(BaseModel):
    id: int
    attributes: dict[str, Any]
    graph_meta: dict[str, Any] | None = None


class AnswerListItem(BaseModel):
    id: int
    body_preview: str
    score: int | None = None
    is_accepted: bool | None = None
    question_id: int | None = None


class AnswerListResponse(BaseModel):
    items: list[AnswerListItem]
    meta: PageMeta


class AnswerDetail(BaseModel):
    id: int
    attributes: dict[str, Any]
    graph_meta: dict[str, Any] | None = None


class TagListItem(BaseModel):
    name: str
    question_count: int | None = None


class TagListResponse(BaseModel):
    items: list[TagListItem]
    meta: PageMeta


class TagDetail(BaseModel):
    name: str
    question_count: int | None = None
    questions: list[QuestionListItem]


class StatsSummary(BaseModel):
    total_questions: int
    total_answers: int
    total_tags: int
    total_edges: int
