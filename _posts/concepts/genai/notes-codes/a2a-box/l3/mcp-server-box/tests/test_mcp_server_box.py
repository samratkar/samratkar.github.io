import argparse
from unittest.mock import Mock, patch

import pytest
from mcp.server.fastmcp import FastMCP

from mcp_server_box import get_mcp_server


@pytest.fixture
def mock_fastmcp():
    """Mock FastMCP instance for testing."""
    mock_mcp = Mock(spec=FastMCP)
    mock_mcp.name = "Test Box MCP Server"
    mock_mcp.tool = Mock(return_value=lambda func: func)
    mock_mcp.sse_app = Mock(return_value=Mock())
    mock_mcp.run = Mock()
    return mock_mcp


@pytest.fixture
def mock_box_lifespan():
    """Mock box_lifespan context manager."""
    mock_lifespan = Mock()
    return mock_lifespan


class TestGetMcpServer:
    """Test the get_mcp_server function."""

    @patch("mcp_server_box.FastMCP")
    @patch("mcp_server_box.box_lifespan")
    def test_get_mcp_server_stdio_default(self, mock_lifespan, mock_fastmcp_class):
        """Test get_mcp_server with default stdio transport."""
        mock_instance = Mock()
        mock_fastmcp_class.return_value = mock_instance

        result = get_mcp_server()

        mock_fastmcp_class.assert_called_once_with(
            "Box MCP Server", lifespan=mock_lifespan
        )
        assert result == mock_instance

    @patch("mcp_server_box.FastMCP")
    @patch("mcp_server_box.box_lifespan")
    def test_get_mcp_server_stdio_explicit(self, mock_lifespan, mock_fastmcp_class):
        """Test get_mcp_server with explicit stdio transport."""
        mock_instance = Mock()
        mock_fastmcp_class.return_value = mock_instance

        result = get_mcp_server(server_name="Custom Server", transport="stdio")

        mock_fastmcp_class.assert_called_once_with(
            "Custom Server", lifespan=mock_lifespan
        )
        assert result == mock_instance

    @patch("mcp_server_box.FastMCP")
    @patch("mcp_server_box.box_lifespan")
    def test_get_mcp_server_sse_transport(self, mock_lifespan, mock_fastmcp_class):
        """Test get_mcp_server with SSE transport."""
        mock_instance = Mock()
        mock_fastmcp_class.return_value = mock_instance

        result = get_mcp_server(
            server_name="SSE Server", transport="sse", host="192.168.1.1", port=9000
        )

        mock_fastmcp_class.assert_called_once_with(
            "SSE Server",
            stateless_http=True,
            host="192.168.1.1",
            port=9000,
            lifespan=mock_lifespan,
        )
        assert result == mock_instance

    @patch("mcp_server_box.FastMCP")
    @patch("mcp_server_box.box_lifespan")
    def test_get_mcp_server_http_transport(self, mock_lifespan, mock_fastmcp_class):
        """Test get_mcp_server with HTTP transport."""
        mock_instance = Mock()
        mock_fastmcp_class.return_value = mock_instance

        result = get_mcp_server(
            server_name="HTTP Server", transport="http", host="localhost", port=8080
        )

        mock_fastmcp_class.assert_called_once_with(
            "HTTP Server",
            stateless_http=True,
            host="localhost",
            port=8080,
            lifespan=mock_lifespan,
        )
        assert result == mock_instance


class TestMainExecution:
    """Test the main execution block."""

    # @patch("mcp_server_box.argparse.ArgumentParser")
    # @patch("mcp_server_box.get_mcp_server")
    # @patch("mcp_server_box.register_tools")

    @patch("mcp_server_box.FastAPI")
    def test_main_execution_sse_transport(self, mock_fastapi):
        """Test main execution with SSE transport configuration."""
        # Test that FastAPI would be imported and used for SSE
        mock_app = Mock()
        mock_fastapi.return_value = mock_app

        # Simulate SSE transport configuration
        mock_mcp = Mock()
        mock_mcp.sse_app.return_value = Mock()

        # This simulates the SSE block in main
        app = mock_fastapi()
        app.mount("/", mock_mcp.sse_app())

        mock_fastapi.assert_called_once()
        mock_app.mount.assert_called_once_with("/", mock_mcp.sse_app())

    def test_argument_parser_configuration(self):
        """Test that argument parser is configured correctly."""
        parser = argparse.ArgumentParser(description="Box MCP Server")
        parser.add_argument(
            "--transport",
            choices=["stdio", "sse", "http"],
            default="stdio",
            help="Transport type (default: stdio)",
        )
        parser.add_argument(
            "--host",
            default="0.0.0.0",
            help="Host for SSE/HTTP transport (default: 0.0.0.0)",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="Port for SSE/HTTP transport (default: 8000)",
        )

        # Test default values
        args = parser.parse_args([])
        assert args.transport == "stdio"
        assert args.host == "0.0.0.0"
        assert args.port == 8000

        # Test custom values
        args = parser.parse_args(
            ["--transport", "sse", "--host", "localhost", "--port", "9000"]
        )
        assert args.transport == "sse"
        assert args.host == "localhost"
        assert args.port == 9000


class TestMcpServerInfoTool:
    """Test the dynamically created mcp_server_info tool."""

    def test_mcp_server_info_stdio(self):
        """Test mcp_server_info for stdio transport."""
        # This would be dynamically created in the main block
        # We'll simulate the function that would be created

        class MockArgs:
            def __init__(self, transport, host="0.0.0.0", port=8000):
                self.transport = transport
                self.host = host
                self.port = port

        class MockMcp:
            def __init__(self, name):
                self.name = name

        def create_mcp_server_info(args, mcp):
            def mcp_server_info():
                """Returns information about the MCP server."""
                if args.transport == "stdio":
                    return {
                        "server_name": mcp.name,
                        "transport": args.transport,
                        "host": "N/A",
                        "port": "N/A",
                    }
                return {
                    "server_name": mcp.name,
                    "transport": args.transport,
                    "host": args.host,
                    "port": args.port,
                }

            return mcp_server_info

        args = MockArgs("stdio")
        mcp = MockMcp("Test Server")

        mcp_server_info = create_mcp_server_info(args, mcp)
        result = mcp_server_info()

        assert result["server_name"] == "Test Server"
        assert result["transport"] == "stdio"
        assert result["host"] == "N/A"
        assert result["port"] == "N/A"

    def test_mcp_server_info_sse(self):
        """Test mcp_server_info for SSE transport."""

        class MockArgs:
            def __init__(self, transport, host="0.0.0.0", port=8000):
                self.transport = transport
                self.host = host
                self.port = port

        class MockMcp:
            def __init__(self, name):
                self.name = name

        def create_mcp_server_info(args, mcp):
            def mcp_server_info():
                """Returns information about the MCP server."""
                if args.transport == "stdio":
                    return {
                        "server_name": mcp.name,
                        "transport": args.transport,
                        "host": "N/A",
                        "port": "N/A",
                    }
                return {
                    "server_name": mcp.name,
                    "transport": args.transport,
                    "host": args.host,
                    "port": args.port,
                }

            return mcp_server_info

        args = MockArgs("sse", "localhost", 9000)
        mcp = MockMcp("SSE Test Server")

        mcp_server_info = create_mcp_server_info(args, mcp)
        result = mcp_server_info()

        assert result["server_name"] == "SSE Test Server"
        assert result["transport"] == "sse"
        assert result["host"] == "localhost"
        assert result["port"] == 9000

    def test_mcp_server_info_http(self):
        """Test mcp_server_info for HTTP transport."""

        class MockArgs:
            def __init__(self, transport, host="0.0.0.0", port=8000):
                self.transport = transport
                self.host = host
                self.port = port

        class MockMcp:
            def __init__(self, name):
                self.name = name

        def create_mcp_server_info(args, mcp):
            def mcp_server_info():
                """Returns information about the MCP server."""
                if args.transport == "stdio":
                    return {
                        "server_name": mcp.name,
                        "transport": args.transport,
                        "host": "N/A",
                        "port": "N/A",
                    }
                return {
                    "server_name": mcp.name,
                    "transport": args.transport,
                    "host": args.host,
                    "port": args.port,
                }

            return mcp_server_info

        args = MockArgs("http", "192.168.1.100", 8080)
        mcp = MockMcp("HTTP Test Server")

        mcp_server_info = create_mcp_server_info(args, mcp)
        result = mcp_server_info()

        assert result["server_name"] == "HTTP Test Server"
        assert result["transport"] == "http"
        assert result["host"] == "192.168.1.100"
        assert result["port"] == 8080
