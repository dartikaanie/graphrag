"""Amendment 16: the Question-level trustScore is hidden in the dashboard unless the display setting is on."""

from app.models.schemas import SettingsUpdate
from app.services import settings_service


def test_question_trust_display_is_off_by_default():
    assert settings_service.DEFAULTS["show_question_trust_score"] is False


def test_setting_is_accepted_and_only_set_when_given():
    assert SettingsUpdate(show_question_trust_score=True).model_dump(exclude_unset=True) == {
        "show_question_trust_score": True}
    assert "show_question_trust_score" not in SettingsUpdate(model="x").model_dump(exclude_unset=True)
