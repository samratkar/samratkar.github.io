import pytest
from box_ai_agents_toolkit import BoxClient, get_oauth_client

from server_context import BoxContext


class FakeRequestContext:
    """
    A fake request context class for testing purposes.
    This class simulates the request context that would be provided by the FastMCP server.
    """

    def __init__(self):
        self.lifespan_context = BoxContext(client=get_oauth_client())


class FakeContext:
    """
    A fake context class for testing purposes.
    This class simulates the context that would be provided by the FastMCP server.
    """

    def __init__(self):
        self.request_context = FakeRequestContext()


@pytest.fixture(scope="module")
def box_client() -> BoxClient:
    # return get_ccg_client()
    return get_oauth_client()


@pytest.fixture(scope="module")
def ctx():
    """
    Fixture to provide a Context object for testing.
    """
    ctx = FakeContext()
    return ctx
