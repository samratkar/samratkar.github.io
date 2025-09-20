import pytest
from box_ai_agents_toolkit import authorize_app, get_oauth_client


@pytest.mark.skip
def test_box_authorize_app_tool():
    try:
        authorize_app()
    except Exception as e:
        assert str(e) == "Box application not authorized yet."


@pytest.mark.skip
def test_box_auth_client():
    try:
        get_oauth_client()
    except Exception as e:
        assert str(e) == "Box application not authorized yet."
