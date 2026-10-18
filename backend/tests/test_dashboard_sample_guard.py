"""Dashboard runs of A, B and D sample only the test split; a dev request is
refused instead of silently running on test positions 1..n."""

import pytest

from app.services import engine_service
from llm.sample_split import SampleMismatchError


@pytest.mark.parametrize("condition", ["A", "B", "D"])
def test_dev_split_refused_for_dashboard_a_b_d(condition):
    with pytest.raises(SampleMismatchError, match="support only sample_split=test"):
        engine_service._refuse_dashboard_dev_split({"condition": condition, "sample_split": "dev"})
    engine_service._refuse_dashboard_dev_split({"condition": condition, "sample_split": "test"})
    engine_service._refuse_dashboard_dev_split({"condition": condition})
