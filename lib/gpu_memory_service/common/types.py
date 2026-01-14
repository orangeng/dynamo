"""Shared types for GPU Memory Service."""

from dataclasses import dataclass
from enum import Enum, auto

from gpu_memory_service.common.protocol.messages import *  # noqa: F401,F403


class ServerState(str, Enum):
    """Server state - derived from actual connections."""
    EMPTY = "EMPTY"
    RW = "RW"
    COMMITTED = "COMMITTED"
    RO = "RO"


class StateEvent(Enum):
    """Events that trigger state transitions."""
    RW_CONNECT = auto()
    RW_COMMIT = auto()
    RW_ABORT = auto()
    RO_CONNECT = auto()
    RO_DISCONNECT = auto()


@dataclass
class StateSnapshot:
    """Current server state snapshot."""

    state: ServerState
    has_rw: bool
    ro_count: int
    waiting_writers: int
    committed: bool

    @property
    def is_ready(self) -> bool:
        """Ready = committed and no RW connection."""
        return self.committed and not self.has_rw


def derive_state(has_rw: bool, ro_count: int, committed: bool) -> ServerState:
    """Derive server state from connection info."""
    if has_rw:
        return ServerState.RW
    if ro_count > 0:
        return ServerState.RO
    if committed:
        return ServerState.COMMITTED
    return ServerState.EMPTY


# Permission sets: which message types require which connection mode
RW_REQUIRED: frozenset[type] = frozenset({
    AllocateRequest,
    FreeRequest,
    ClearAllRequest,
    MetadataPutRequest,
    MetadataDeleteRequest,
    CommitRequest,
})

RO_ALLOWED: frozenset[type] = frozenset({
    ExportRequest,
    GetAllocationRequest,
    ListAllocationsRequest,
    MetadataGetRequest,
    MetadataListRequest,
    GetLockStateRequest,
    GetAllocationStateRequest,
    GetStateHashRequest,
})

RW_ALLOWED: frozenset[type] = RW_REQUIRED | RO_ALLOWED
