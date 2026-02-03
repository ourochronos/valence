"""Unified MCP server combining substrate and VKB tools.

Provides a single MCP server with all Valence tools, behavioral prompts,
and resources for HTTP transport.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    Prompt,
    PromptMessage,
    PromptArgument,
    GetPromptResult,
    Resource,
)

from ..core.exceptions import DatabaseException, ValidationException
from ..substrate.tools import SUBSTRATE_TOOLS, handle_substrate_tool
from ..vkb.tools import VKB_TOOLS, handle_vkb_tool

logger = logging.getLogger(__name__)

# Server name for MCP
SERVER_NAME = "valence"
SERVER_VERSION = "1.0.0"


def create_server() -> Server:
    """Create and configure the unified MCP server.

    Returns:
        Configured MCP Server instance
    """
    server = Server(SERVER_NAME)

    # ========================================================================
    # Tool Registration
    # ========================================================================

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return all available tools from both substrate and VKB."""
        return SUBSTRATE_TOOLS + VKB_TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool calls, routing to appropriate handler."""
        try:
            result: dict[str, Any]

            # Check if it's a substrate tool
            if name in [t.name for t in SUBSTRATE_TOOLS]:
                result = handle_substrate_tool(name, arguments)
            # Check if it's a VKB tool
            elif name in [t.name for t in VKB_TOOLS]:
                result = handle_vkb_tool(name, arguments)
            else:
                result = {"success": False, "error": f"Unknown tool: {name}"}

            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        except ValidationException as e:
            logger.warning(f"Validation error in tool {name}: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": False,
                            "error": f"Validation error: {e.message}",
                            "details": e.details,
                        }
                    ),
                )
            ]
        except DatabaseException as e:
            logger.error(f"Database error in tool {name}: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": False,
                            "error": f"Database error: {e.message}",
                        }
                    ),
                )
            ]
        except Exception as e:
            logger.exception(f"Unexpected error in tool {name}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": False,
                            "error": f"Internal error: {str(e)}",
                        }
                    ),
                )
            ]

    # ========================================================================
    # Resource Registration (Usage Instructions)
    # ========================================================================

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """List available resources."""
        return [
            Resource(
                uri="valence://instructions",
                name="Valence Usage Instructions",
                description="Guidelines for using Valence knowledge tools effectively",
                mimeType="text/markdown",
            ),
            Resource(
                uri="valence://tools",
                name="Valence Tool Reference",
                description="Quick reference for all available tools",
                mimeType="text/markdown",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a resource by URI."""
        if uri == "valence://instructions":
            return _get_usage_instructions()
        elif uri == "valence://tools":
            return _get_tool_reference()
        else:
            raise ValueError(f"Unknown resource: {uri}")

    # ========================================================================
    # Prompt Registration (Behavioral Conditioning)
    # ========================================================================

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        """List available prompts."""
        return [
            Prompt(
                name="valence-context",
                description="Inject Valence context and behavioral guidelines",
                arguments=[],
            ),
            Prompt(
                name="recall-context",
                description="Query knowledge base for relevant context on a topic",
                arguments=[
                    PromptArgument(
                        name="topic",
                        description="Topic to recall context about",
                        required=True,
                    )
                ],
            ),
        ]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None = None) -> GetPromptResult:
        """Get a prompt by name."""
        if name == "valence-context":
            return GetPromptResult(
                description="Valence context and behavioral guidelines",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=_get_context_prompt(),
                        ),
                    )
                ],
            )
        elif name == "recall-context":
            topic = (arguments or {}).get("topic", "")
            return GetPromptResult(
                description=f"Recall context about: {topic}",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"Before responding, use belief_query to search for relevant knowledge about: {topic}",
                        ),
                    )
                ],
            )
        else:
            raise ValueError(f"Unknown prompt: {name}")

    return server


def _get_usage_instructions() -> str:
    """Get the usage instructions resource content."""
    return """# Valence Knowledge Substrate

## Core Principle
Query first, respond with grounded information, capture insights for the future.

## Required Behaviors

### 1. ALWAYS Query Before Answering
When the user asks about:
- Past decisions or discussions
- User preferences or values
- Technical approaches previously explored
- Any topic that may have been discussed before

**FIRST** use `belief_query` to search for relevant knowledge, **THEN** respond with grounded information.

### 2. PROACTIVELY Capture Knowledge
Use `belief_create` or `insight_extract` when:
- A decision is made with clear rationale
- User expresses a preference or value
- A problem is solved with a novel approach
- Important factual information is shared
- Architectural or design choices are finalized

### 3. Track Sessions
- Call `session_start` at the beginning of conversations
- Call `session_end` with summary and themes when concluding
- This enables session-based context for future conversations

### 4. Record Patterns
When you notice recurring themes, preferences, or behaviors across multiple conversations,
use `pattern_record` to capture them.

## Tool Categories

### Knowledge Substrate (Beliefs)
- `belief_query` - Search for existing knowledge (USE FIRST)
- `belief_create` - Store new knowledge
- `belief_supersede` - Update existing knowledge with history
- `belief_get` - Get detailed belief information
- `entity_get`, `entity_search` - Work with entities (people, tools, concepts)
- `tension_list`, `tension_resolve` - Handle contradictions

### Conversation Tracking (VKB)
- `session_start`, `session_end`, `session_get`, `session_list` - Session management
- `exchange_add`, `exchange_list` - Conversation turn tracking
- `pattern_record`, `pattern_reinforce`, `pattern_list` - Behavioral patterns
- `insight_extract`, `insight_list` - Extract knowledge from conversations

## Best Practices

1. **Be proactive** - Don't wait to be asked to query or capture knowledge
2. **Be specific** - Use detailed queries and create focused beliefs
3. **Link entities** - Connect beliefs to relevant people, tools, and concepts
4. **Maintain context** - Use sessions to group related conversations
5. **Resolve tensions** - When you find conflicting information, work to resolve it
"""


def _get_tool_reference() -> str:
    """Get the tool reference resource content."""
    lines = ["# Valence Tool Reference\n"]

    lines.append("## Knowledge Substrate Tools\n")
    for tool in SUBSTRATE_TOOLS:
        # Just first line of description
        desc = tool.description.split("\n")[0]
        lines.append(f"- **{tool.name}**: {desc}")

    lines.append("\n## Conversation Tracking Tools\n")
    for tool in VKB_TOOLS:
        desc = tool.description.split("\n")[0]
        lines.append(f"- **{tool.name}**: {desc}")

    return "\n".join(lines)


def _get_context_prompt() -> str:
    """Get the context injection prompt."""
    return """You have access to Valence, a personal knowledge substrate.

**Core Behaviors:**
1. **Query First**: Before answering questions about past context, decisions, or preferences,
   use `belief_query` to search the knowledge base.
2. **Capture Proactively**: When valuable information emerges (decisions, preferences, insights),
   use `belief_create` or `insight_extract` to store it for future reference.
3. **Track Sessions**: Use `session_start` and `session_end` to maintain conversation context.

This ensures your responses are grounded in accumulated knowledge and that new insights
are preserved for future conversations.
"""
