"""
Plugins package - Extensibility framework.
"""

from .registry import (
    PluginType,
    PluginInfo,
    ConditionPlugin,
    DataSourcePlugin,
    AgentPlugin,
    PluginRegistry,
    plugin_registry,
    register_condition,
    register_data_source,
    register_agent,
)

__all__ = [
    "PluginType",
    "PluginInfo",
    "ConditionPlugin",
    "DataSourcePlugin",
    "AgentPlugin",
    "PluginRegistry",
    "plugin_registry",
    "register_condition",
    "register_data_source",
    "register_agent",
]
