"""
Extensibility Framework

Plugin architecture for:
- New health conditions
- New data sources
- New AI agents

Designed for easy extension without modifying core code.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Type, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class PluginType(Enum):
    """Types of plugins supported."""
    CONDITION = "condition"
    DATA_SOURCE = "data_source"
    AGENT = "agent"


@dataclass
class PluginInfo:
    """Metadata about a plugin."""
    name: str
    plugin_type: PluginType
    version: str
    description: str
    author: str = "Unknown"


class ConditionPlugin(ABC):
    """
    Base class for health condition plugins.
    
    Implement this to add support for new conditions (e.g., GERD, IBS).
    """
    
    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Plugin metadata."""
        pass
    
    @property
    @abstractmethod
    def condition_name(self) -> str:
        """Name of the health condition."""
        pass
    
    @abstractmethod
    def get_rules(self) -> List[Dict[str, Any]]:
        """
        Return rules for this condition.
        
        Each rule should have:
        - rule_id: str
        - severity: str (block, alert, warn, info)
        - check: Callable[[Food, UserProfile], bool]
        - message: str
        - suggestion: str
        """
        pass
    
    @abstractmethod
    def get_daily_targets(self) -> Dict[str, float]:
        """Return recommended daily targets for this condition."""
        pass


class DataSourcePlugin(ABC):
    """
    Base class for nutrition data source plugins.
    
    Implement this to add new nutrition databases.
    """
    
    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Plugin metadata."""
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the data source."""
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for foods by name."""
        pass
    
    @abstractmethod
    def get_nutrition(self, food_id: str) -> Optional[Dict[str, Any]]:
        """Get nutrition info for a specific food."""
        pass


class AgentPlugin(ABC):
    """
    Base class for AI agent plugins.
    
    Implement this to add new AI capabilities.
    """
    
    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Plugin metadata."""
        pass
    
    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Name of the agent."""
        pass
    
    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request and return response."""
        pass
    
    @abstractmethod
    def validate_safety(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate response against safety rules.
        
        REQUIRED: All agents MUST validate against medical rules.
        """
        pass


class PluginRegistry:
    """
    Central registry for all plugins.
    
    Plugins self-register on import.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._conditions: Dict[str, ConditionPlugin] = {}
            cls._instance._data_sources: Dict[str, DataSourcePlugin] = {}
            cls._instance._agents: Dict[str, AgentPlugin] = {}
        return cls._instance
    
    def register_condition(self, plugin: ConditionPlugin):
        """Register a condition plugin."""
        self._conditions[plugin.condition_name] = plugin
    
    def register_data_source(self, plugin: DataSourcePlugin):
        """Register a data source plugin."""
        self._data_sources[plugin.source_name] = plugin
    
    def register_agent(self, plugin: AgentPlugin):
        """Register an agent plugin."""
        self._agents[plugin.agent_name] = plugin
    
    def get_condition(self, name: str) -> Optional[ConditionPlugin]:
        """Get a condition plugin by name."""
        return self._conditions.get(name)
    
    def get_data_source(self, name: str) -> Optional[DataSourcePlugin]:
        """Get a data source plugin by name."""
        return self._data_sources.get(name)
    
    def get_agent(self, name: str) -> Optional[AgentPlugin]:
        """Get an agent plugin by name."""
        return self._agents.get(name)
    
    def list_plugins(self) -> Dict[str, List[PluginInfo]]:
        """List all registered plugins."""
        return {
            "conditions": [p.info for p in self._conditions.values()],
            "data_sources": [p.info for p in self._data_sources.values()],
            "agents": [p.info for p in self._agents.values()],
        }


# Global registry instance
plugin_registry = PluginRegistry()


# Decorator for easy plugin registration
def register_condition(cls: Type[ConditionPlugin]) -> Type[ConditionPlugin]:
    """Decorator to auto-register condition plugins."""
    instance = cls()
    plugin_registry.register_condition(instance)
    return cls


def register_data_source(cls: Type[DataSourcePlugin]) -> Type[DataSourcePlugin]:
    """Decorator to auto-register data source plugins."""
    instance = cls()
    plugin_registry.register_data_source(instance)
    return cls


def register_agent(cls: Type[AgentPlugin]) -> Type[AgentPlugin]:
    """Decorator to auto-register agent plugins."""
    instance = cls()
    plugin_registry.register_agent(instance)
    return cls
