"""
MCP Service - Service Layer for MCP Operations
Business logic for interacting with MCP servers.
"""
from typing import Any, Optional
from core.mcp_providers import MCPProvider


class MCPService:
    """
    Service for managing MCP operations (Service Layer Pattern).
    Provides high-level interface for MCP tool interactions.
    """
    
    def __init__(self, provider: MCPProvider):
        """
        Initialize MCP service with a provider.
        
        Args:
            provider: MCP provider instance (Dependency Injection)
        """
        self.provider = provider
    
    async def list_tools(self) -> list[dict[str, Any]]:
        """
        Get list of available tools from MCP server.
        
        Returns:
            List of tool definitions with name, description, and input schema
            
        Example:
            >>> service = MCPService(filesystem_provider)
            >>> async with service.provider.connect() as session:
            >>>     tools = await service.list_tools()
            >>>     print([tool['name'] for tool in tools])
        """
        if not self.provider.session:
            raise RuntimeError(
                "MCP session not initialized. "
                "Use 'async with provider.connect()' first."
            )
        
        response = await self.provider.session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in response.tools
        ]
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> Any:
        """
        Execute a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments as dictionary
            
        Returns:
            Tool execution result
            
        Raises:
            RuntimeError: If session is not initialized
            
        Example:
            >>> async with provider.connect():
            >>>     result = await service.call_tool("read_file", {"path": "test.py"})
        """
        if not self.provider.session:
            raise RuntimeError(
                "MCP session not initialized. "
                "Use 'async with provider.connect()' first."
            )
        
        result = await self.provider.session.call_tool(tool_name, arguments)
        return result
    
    # Filesystem-specific convenience methods
    async def read_file(self, file_path: str) -> str:
        """
        Read a file using MCP filesystem tools.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File contents as string
            
        Example:
            >>> content = await service.read_file("data/input.txt")
        """
        result = await self.call_tool("read_file", {"path": file_path})
        return result.content[0].text if result.content else ""
    
    async def write_file(self, file_path: str, content: str) -> None:
        """
        Write content to a file using MCP filesystem tools.
        
        Args:
            file_path: Path to the file
            content: Content to write
            
        Example:
            >>> await service.write_file("output/result.txt", "Hello, world!")
        """
        await self.call_tool("write_file", {"path": file_path, "content": content})
    
    async def list_directory(self, directory_path: str) -> list[str]:
        """
        List contents of a directory using MCP filesystem tools.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of file/directory names
            
        Example:
            >>> files = await service.list_directory("./data")
        """
        result = await self.call_tool("list_directory", {"path": directory_path})
        return result.content if result.content else []
    
    # GitHub-specific convenience methods
    async def search_repositories(
        self,
        query: str,
        max_results: int = 5
    ) -> list[dict]:
        """
        Search GitHub repositories using MCP.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of repository information
            
        Example:
            >>> repos = await service.search_repositories("langchain python", max_results=10)
        """
        result = await self.call_tool(
            "search_repositories",
            {"query": query, "page": 1, "perPage": max_results}
        )
        return result.content if result.content else []
    
    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: Optional[str] = None
    ) -> dict:
        """
        Create a GitHub issue using MCP.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: Issue title
            body: Issue body/description
            
        Returns:
            Created issue information
            
        Example:
            >>> issue = await service.create_issue(
            >>>     "owner", "repo", "Bug: Something broke", "Description here"
            >>> )
        """
        result = await self.call_tool(
            "create_issue",
            {"owner": owner, "repo": repo, "title": title, "body": body or ""}
        )
        return result.content if result.content else {}
    
    # Memory-specific convenience methods
    async def create_entities(
        self,
        entities: list[dict[str, Any]]
    ) -> None:
        """
        Store entities in MCP memory/knowledge graph.
        
        Args:
            entities: List of entity dictionaries with name, entityType, observations
            
        Example:
            >>> await service.create_entities([
            >>>     {
            >>>         "name": "Python",
            >>>         "entityType": "programming_language",
            >>>         "observations": ["Used for AI/ML", "Popular in 2024"]
            >>>     }
            >>> ])
        """
        await self.call_tool("create_entities", {"entities": entities})
    
    async def search_entities(self, query: str) -> list[dict]:
        """
        Search entities in MCP memory.
        
        Args:
            query: Search query
            
        Returns:
            List of matching entities
            
        Example:
            >>> results = await service.search_entities("Python")
        """
        result = await self.call_tool("search_entities", {"query": query})
        return result.content if result.content else []
    
    # PostgreSQL-specific convenience methods
    async def query_database(self, sql: str) -> list[dict]:
        """
        Execute SQL query using MCP PostgreSQL provider.
        
        Args:
            sql: SQL query string
            
        Returns:
            Query results as list of dictionaries
            
        Example:
            >>> results = await service.query_database("SELECT * FROM users LIMIT 10")
        """
        result = await self.call_tool("query", {"sql": sql})
        return result.content if result.content else []
    
    # Puppeteer-specific convenience methods
    async def navigate_to_url(self, url: str) -> str:
        """
        Navigate to a URL and get page content using MCP Puppeteer.
        
        Args:
            url: URL to navigate to
            
        Returns:
            Page HTML content
            
        Example:
            >>> html = await service.navigate_to_url("https://example.com")
        """
        result = await self.call_tool("navigate", {"url": url})
        return result.content[0].text if result.content else ""
    
    async def screenshot(self, url: str, output_path: str) -> None:
        """
        Take a screenshot of a webpage using MCP Puppeteer.
        
        Args:
            url: URL to screenshot
            output_path: Path to save screenshot
            
        Example:
            >>> await service.screenshot("https://example.com", "screenshot.png")
        """
        await self.call_tool(
            "screenshot",
            {"url": url, "path": output_path}
        )
