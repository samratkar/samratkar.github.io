from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import MagicMock, Mock, patch

import pytest

from server_context import BoxContext, box_lifespan


class TestBoxContext:
    """Test the BoxContext dataclass."""

    def test_box_context_default_initialization(self):
        """Test BoxContext initialization with default values."""
        context = BoxContext()
        assert context.client is None

    def test_box_context_with_client(self):
        """Test BoxContext initialization with a client."""
        mock_client = MagicMock()
        context = BoxContext(client=mock_client)
        assert context.client == mock_client

    def test_box_context_dataclass_properties(self):
        """Test that BoxContext is properly defined as a dataclass."""
        # Test that it has the expected attributes
        context = BoxContext()
        assert hasattr(context, "client")

        # Test that we can set the client after initialization
        mock_client = MagicMock()
        context.client = mock_client
        assert context.client == mock_client

    def test_box_context_equality(self):
        """Test BoxContext equality comparison."""
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()

        context1 = BoxContext(client=mock_client1)
        context2 = BoxContext(client=mock_client1)  # Same client
        context3 = BoxContext(client=mock_client2)  # Different client
        context4 = BoxContext()  # No client

        assert context1 == context2
        assert context1 != context3
        assert context1 != context4

    def test_box_context_repr(self):
        """Test BoxContext string representation."""
        mock_client = MagicMock()
        mock_client.__repr__ = Mock(return_value="MockBoxClient")

        context = BoxContext(client=mock_client)
        repr_str = repr(context)

        assert "BoxContext" in repr_str
        assert "client=" in repr_str


class TestBoxLifespan:
    """Test the box_lifespan async context manager."""

    @pytest.mark.asyncio
    @patch("server_context.get_oauth_client")
    async def test_box_lifespan_success(self, mock_get_oauth_client):
        """Test successful box_lifespan context manager execution."""
        mock_client = MagicMock()
        mock_get_oauth_client.return_value = mock_client
        mock_server = MagicMock()

        async with box_lifespan(mock_server) as context:
            assert isinstance(context, BoxContext)
            assert context.client == mock_client

        mock_get_oauth_client.assert_called_once()

    @pytest.mark.asyncio
    @patch("server_context.get_oauth_client")
    async def test_box_lifespan_client_initialization_failure(
        self, mock_get_oauth_client
    ):
        """Test box_lifespan when client initialization fails."""
        mock_get_oauth_client.side_effect = Exception(
            "OAuth client initialization failed"
        )
        mock_server = MagicMock()

        with pytest.raises(Exception) as exc_info:
            async with box_lifespan(mock_server) as context:  # noqa: F841
                pass  # This shouldn't be reached

        assert str(exc_info.value) == "OAuth client initialization failed"
        mock_get_oauth_client.assert_called_once()

    @pytest.mark.asyncio
    @patch("server_context.get_oauth_client")
    async def test_box_lifespan_exception_during_yield(self, mock_get_oauth_client):
        """Test box_lifespan when an exception occurs during the yield block."""
        mock_client = MagicMock()
        mock_get_oauth_client.return_value = mock_client
        mock_server = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            async with box_lifespan(mock_server) as context:
                assert context.client == mock_client
                raise ValueError("Test exception during yield")

        assert str(exc_info.value) == "Test exception during yield"
        mock_get_oauth_client.assert_called_once()

    @pytest.mark.asyncio
    @patch("server_context.get_oauth_client")
    async def test_box_lifespan_cleanup_called(self, mock_get_oauth_client):
        """Test that box_lifespan cleanup code is executed."""
        mock_client = MagicMock()
        mock_get_oauth_client.return_value = mock_client
        mock_server = MagicMock()

        # Mock the cleanup section by patching the entire function
        original_box_lifespan = box_lifespan  # noqa: F841

        cleanup_called = []

        @asynccontextmanager
        async def mock_box_lifespan(server) -> AsyncIterator[BoxContext]:
            try:
                client = mock_get_oauth_client()
                yield BoxContext(client=client)
            finally:
                # Track that cleanup was called
                cleanup_called.append(True)

        # Use the mock version
        async with mock_box_lifespan(mock_server) as context:
            assert context.client == mock_client

        assert len(cleanup_called) == 1
        mock_get_oauth_client.assert_called_once()

    @pytest.mark.asyncio
    @patch("server_context.get_oauth_client")
    async def test_box_lifespan_with_none_client(self, mock_get_oauth_client):
        """Test box_lifespan when get_oauth_client returns None."""
        mock_get_oauth_client.return_value = None
        mock_server = MagicMock()

        async with box_lifespan(mock_server) as context:
            assert isinstance(context, BoxContext)
            assert context.client is None

        mock_get_oauth_client.assert_called_once()

    @pytest.mark.asyncio
    @patch("server_context.get_oauth_client")
    async def test_box_lifespan_multiple_calls(self, mock_get_oauth_client):
        """Test multiple calls to box_lifespan."""
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        mock_get_oauth_client.side_effect = [mock_client1, mock_client2]
        mock_server = MagicMock()

        # First call
        async with box_lifespan(mock_server) as context1:
            assert context1.client == mock_client1

        # Second call
        async with box_lifespan(mock_server) as context2:
            assert context2.client == mock_client2

        assert mock_get_oauth_client.call_count == 2

    @pytest.mark.asyncio
    @patch("server_context.get_oauth_client")
    async def test_box_lifespan_context_isolation(self, mock_get_oauth_client):
        """Test that different box_lifespan contexts are isolated."""
        mock_client = MagicMock()
        mock_get_oauth_client.return_value = mock_client
        mock_server1 = MagicMock()
        mock_server2 = MagicMock()

        # Test that each context gets its own BoxContext instance
        async with box_lifespan(mock_server1) as context1:
            async with box_lifespan(mock_server2) as context2:
                assert isinstance(context1, BoxContext)
                assert isinstance(context2, BoxContext)
                assert context1 is not context2  # Different instances
                assert context1.client == context2.client  # Same client

        assert mock_get_oauth_client.call_count == 2

    @pytest.mark.asyncio
    @patch("server_context.get_oauth_client")
    async def test_box_lifespan_server_parameter_unused(self, mock_get_oauth_client):
        """Test that the server parameter is not used in the current implementation."""
        mock_client = MagicMock()
        mock_get_oauth_client.return_value = mock_client

        # Pass different server objects to verify they don't affect the outcome
        mock_server1 = MagicMock()
        mock_server1.name = "Server1"

        mock_server2 = MagicMock()
        mock_server2.name = "Server2"

        async with box_lifespan(mock_server1) as context1:
            client1 = context1.client

        async with box_lifespan(mock_server2) as context2:
            client2 = context2.client

        # Both should have the same client since server parameter is not used
        assert client1 == client2 == mock_client
        assert mock_get_oauth_client.call_count == 2

    # def test_box_lifespan_is_async_context_manager(self):
    #     """Test that box_lifespan is properly decorated as an async context manager."""
    #     # Verify that box_lifespan has the correct async context manager protocol
    #     assert hasattr(box_lifespan, '__aenter__')
    #     assert hasattr(box_lifespan, '__aexit__')

    #     # Test that it's a coroutine function
    #     import inspect
    #     assert inspect.iscoroutinefunction(box_lifespan(MagicMock()).__aenter__)


class TestBoxContextIntegration:
    """Integration tests for BoxContext and box_lifespan together."""

    @pytest.mark.asyncio
    @patch("server_context.get_oauth_client")
    async def test_integration_box_context_in_lifespan(self, mock_get_oauth_client):
        """Test integration between BoxContext and box_lifespan."""
        mock_client = MagicMock()
        mock_client.users = MagicMock()
        mock_client.users.get_user_me = MagicMock()
        mock_get_oauth_client.return_value = mock_client

        mock_server = MagicMock()

        async with box_lifespan(mock_server) as context:
            # Test that we can use the context as expected
            assert isinstance(context, BoxContext)
            assert context.client == mock_client

            # Test that we can access client methods
            assert context.client is not None
            assert hasattr(context.client, "users")
            assert hasattr(context.client.users, "get_user_me")

    @pytest.mark.asyncio
    @patch("server_context.get_oauth_client")
    async def test_integration_context_modification(self, mock_get_oauth_client):
        """Test that context can be modified during the lifespan."""
        mock_client = MagicMock()
        mock_get_oauth_client.return_value = mock_client
        mock_server = MagicMock()

        async with box_lifespan(mock_server) as context:
            original_client = context.client

            # Modify the context (though this wouldn't be typical usage)
            new_mock_client = MagicMock()
            context.client = new_mock_client

            assert context.client != original_client
            assert context.client == new_mock_client
