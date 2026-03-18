"""
Memory Management Pattern Implementation.

Provides dual-layer memory system with short-term memory for conversation
context and long-term memory for persistent knowledge retrieval. Supports
vector-based retrieval, conversation buffering, and semantic search.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MemoryEntry:
    """Represents a single memory entry."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingEntry:
    """Represents an entry in long-term memory with embedding."""
    content: str
    embedding: Optional[List[float]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryStore(ABC):
    """Abstract base class for memory stores."""

    @abstractmethod
    def add(self, entry: MemoryEntry) -> None:
        """Add an entry to memory."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Retrieve relevant entries from memory."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from memory."""
        pass


class ShortTermMemory(MemoryStore):
    """
    Short-term conversation memory buffer.

    Maintains recent conversation context with a configurable maximum
    number of turns. Implements FIFO eviction when capacity is exceeded.
    """

    def __init__(self, max_turns: int = 10):
        """
        Initialize short-term memory.

        Args:
            max_turns: Maximum number of conversation turns to retain
        """
        self.max_turns = max_turns
        self.entries: List[MemoryEntry] = []
        logger.info("short_term_memory_initialized", max_turns=max_turns)

    def add(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an entry to short-term memory.

        Args:
            role: Role of the speaker ("user", "assistant", "system")
            content: Content of the message
            metadata: Optional metadata for the entry
        """
        entry = MemoryEntry(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.entries.append(entry)

        # Enforce max_turns limit
        if len(self.entries) > self.max_turns * 2:  # 2 for alternating roles
            self.entries.pop(0)

        logger.info(
            "memory_entry_added",
            role=role,
            buffer_size=len(self.entries),
        )

    def retrieve(self, query: Optional[str] = None, top_k: int = 5) -> List[MemoryEntry]:
        """
        Retrieve recent entries from short-term memory.

        Args:
            query: Optional query (ignored for short-term memory)
            top_k: Number of recent entries to retrieve

        Returns:
            List of MemoryEntry objects
        """
        recent = self.entries[-top_k:] if len(self.entries) > top_k else self.entries
        logger.info("memory_retrieved", count=len(recent))
        return recent

    def get_context(self, num_turns: int = 5) -> List[Dict[str, str]]:
        """
        Get context as a list of role-content pairs.

        Args:
            num_turns: Number of turns to include

        Returns:
            List of dictionaries with 'role' and 'content' keys
        """
        recent_entries = self.retrieve(top_k=num_turns * 2)
        return [
            {"role": entry.role, "content": entry.content}
            for entry in recent_entries
        ]

    def clear(self) -> None:
        """Clear all entries from short-term memory."""
        self.entries.clear()
        logger.info("short_term_memory_cleared")

    def size(self) -> int:
        """Get current size of memory buffer."""
        return len(self.entries)


class LongTermMemory(MemoryStore):
    """
    Long-term semantic memory storage.

    Stores persistent knowledge with vector embeddings for semantic
    similarity retrieval. Supports arbitrary metadata and timestamps.
    """

    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize long-term memory.

        Args:
            embedding_model: Name of embedding model to use
        """
        self.embedding_model = embedding_model
        self.entries: List[EmbeddingEntry] = []
        logger.info(
            "long_term_memory_initialized",
            embedding_model=embedding_model,
        )

    def add(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an entry to long-term memory.

        Args:
            content: The text content to store
            embedding: Optional pre-computed embedding vector
            metadata: Optional metadata for the entry
        """
        entry = EmbeddingEntry(
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )
        self.entries.append(entry)

        logger.info(
            "long_term_memory_entry_added",
            content_length=len(content),
            has_embedding=embedding is not None,
        )

    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Store content and return its embedding.

        Args:
            content: The text content to store
            metadata: Optional metadata

        Returns:
            The embedding vector (placeholder implementation)
        """
        # In production, this would call the embedding API
        embedding = self._generate_embedding(content)
        self.add(content, embedding, metadata)
        return embedding

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve semantically similar entries.

        Args:
            query: Query text
            top_k: Number of top results to return

        Returns:
            List of similar content strings
        """
        if not self.entries:
            logger.info("long_term_memory_empty")
            return []

        # Get query embedding
        query_embedding = self._generate_embedding(query)

        # Calculate similarities (placeholder)
        similarities = [
            (i, self._cosine_similarity(query_embedding, entry.embedding))
            for i, entry in enumerate(self.entries)
            if entry.embedding is not None
        ]

        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = [
            self.entries[i].content
            for i, _ in similarities[:top_k]
        ]

        logger.info("long_term_memory_retrieved", count=len(results))
        return results

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search with similarity scores.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (content, similarity_score) tuples
        """
        if not self.entries:
            return []

        query_embedding = self._generate_embedding(query)
        similarities = [
            (entry.content, self._cosine_similarity(query_embedding, entry.embedding))
            for entry in self.entries
            if entry.embedding is not None
        ]

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def clear(self) -> None:
        """Clear all entries from long-term memory."""
        self.entries.clear()
        logger.info("long_term_memory_cleared")

    def size(self) -> int:
        """Get number of entries in long-term memory."""
        return len(self.entries)

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text (placeholder implementation).

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Placeholder: return a simple hash-based vector
        # In production, this would call the embedding API
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        vector = [(hash_int >> i) & 1 for i in range(384)]
        return [float(x) for x in vector]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: Optional[List[float]]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        if vec2 is None or len(vec1) == 0 or len(vec2) == 0:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(x ** 2 for x in vec1) ** 0.5
        norm2 = sum(x ** 2 for x in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class HybridMemory:
    """
    Hybrid memory system combining short-term and long-term memory.

    Provides unified interface for managing both conversation context
    and persistent knowledge.
    """

    def __init__(self, short_term_max_turns: int = 10):
        """
        Initialize hybrid memory system.

        Args:
            short_term_max_turns: Maximum turns for short-term memory
        """
        self.short_term = ShortTermMemory(max_turns=short_term_max_turns)
        self.long_term = LongTermMemory()

    def add_interaction(
        self,
        role: str,
        content: str,
        persist: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an interaction to memory.

        Args:
            role: Role of the speaker
            content: Content of the message
            persist: Whether to also store in long-term memory
            metadata: Optional metadata
        """
        self.short_term.add(role, content, metadata)

        if persist:
            self.long_term.add(content, metadata=metadata)

    def get_context(self, num_turns: int = 5) -> List[Dict[str, str]]:
        """Get conversation context from short-term memory."""
        return self.short_term.get_context(num_turns)

    def search_knowledge(self, query: str, top_k: int = 5) -> List[str]:
        """Search long-term knowledge base."""
        return self.long_term.retrieve(query, top_k)
