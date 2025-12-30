"""
Performance Optimization Utilities

Caching, request timing, and optimization helpers.
Designed for free-tier resource constraints.
"""

import hashlib
import time
import functools
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Dict, TypeVar, Generic
from threading import Lock


T = TypeVar('T')


class LRUCache(Generic[T]):
    """
    Thread-safe Least Recently Used cache.
    
    Simple implementation for single-server deployment.
    """
    
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[T]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def set(self, key: str, value: T):
        """Set item in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            
            # Evict oldest if over capacity
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
    
    def __contains__(self, key: str) -> bool:
        return key in self._cache
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return (self._hits / total) * 100
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.1f}%",
        }
    
    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()


class TTLCache(Generic[T]):
    """
    Time-To-Live cache.
    
    Items expire after a specified duration.
    """
    
    @dataclass
    class CacheEntry:
        value: Any
        expires_at: datetime
    
    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 3600):
        self.maxsize = maxsize
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: Dict[str, TTLCache.CacheEntry] = {}
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[T]:
        """Get item if not expired."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if datetime.now() > entry.expires_at:
                del self._cache[key]
                return None
            
            return entry.value
    
    def set(self, key: str, value: T):
        """Set item with TTL."""
        with self._lock:
            self._cache[key] = self.CacheEntry(
                value=value,
                expires_at=datetime.now() + self.ttl,
            )
            
            # Cleanup expired entries if over capacity
            if len(self._cache) > self.maxsize:
                self._cleanup_expired()
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        now = datetime.now()
        expired = [k for k, v in self._cache.items() if now > v.expires_at]
        for key in expired:
            del self._cache[key]


def cached(cache: LRUCache, key_func: Optional[Callable] = None):
    """
    Decorator for caching function results.
    
    Usage:
        @cached(my_cache)
        def expensive_function(arg):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{args}:{kwargs}"
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        return wrapper
    return decorator


class RequestTimer:
    """
    Context manager for timing requests.
    
    Usage:
        with RequestTimer("api_call") as timer:
            result = do_something()
        print(f"Took {timer.duration_ms}ms")
    """
    
    def __init__(self, name: str = "request"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> "RequestTimer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000
    
    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.duration_ms / 1000


@dataclass
class PerformanceMetrics:
    """Aggregate performance metrics."""
    endpoint: str
    total_requests: int = 0
    total_time_ms: float = 0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0
    
    def record(self, duration_ms: float):
        """Record a request duration."""
        self.total_requests += 1
        self.total_time_ms += duration_ms
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
    
    @property
    def avg_time_ms(self) -> float:
        """Average request time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_time_ms / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "endpoint": self.endpoint,
            "total_requests": self.total_requests,
            "avg_time_ms": round(self.avg_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2) if self.min_time_ms != float('inf') else 0,
            "max_time_ms": round(self.max_time_ms, 2),
        }


class PerformanceMonitor:
    """
    Monitor and report performance metrics.
    """
    
    def __init__(self):
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = Lock()
    
    def record(self, endpoint: str, duration_ms: float):
        """Record request performance."""
        with self._lock:
            if endpoint not in self._metrics:
                self._metrics[endpoint] = PerformanceMetrics(endpoint=endpoint)
            self._metrics[endpoint].record(duration_ms)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            name: m.to_dict()
            for name, m in self._metrics.items()
        }
    
    def get_slow_endpoints(self, threshold_ms: float = 500) -> list:
        """Get endpoints exceeding threshold."""
        return [
            m.to_dict()
            for m in self._metrics.values()
            if m.avg_time_ms > threshold_ms
        ]


def hash_for_cache(data: bytes) -> str:
    """Generate cache key from binary data (e.g., images)."""
    return hashlib.sha256(data).hexdigest()


# Global instances
nutrition_cache = TTLCache[Dict](maxsize=10000, ttl_seconds=86400)  # 24h
ocr_cache = LRUCache[str](maxsize=1000)
performance_monitor = PerformanceMonitor()
