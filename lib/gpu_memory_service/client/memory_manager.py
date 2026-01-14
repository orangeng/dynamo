# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service client-side memory manager.

This is the unified memory manager for the GPU Memory Service architecture.

Key properties:
- Uses GMSRPCClient over a Unix-domain socket.
- The socket connection itself is the RW/RO lock.
- In write mode, the manager can allocate + map RW and then publish via commit().
- In read mode, the manager can import + map RO and hold the RO lock during inference.
- sleep()/wake() releases and reacquires the RO lock (and remaps allocations).

This module uses cuda-python bindings for CUDA driver API calls:
- import FDs (cuMemImportFromShareableHandle)
- reserve VA (cuMemAddressReserve)
- map/unmap (cuMemMap/cuMemUnmap)
- enforce access (cuMemSetAccess)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import torch
from cuda.bindings import driver as cuda
from gpu_memory_service.client.rpc import GMSRPCClient
from gpu_memory_service.common.cuda_vmm_utils import (
    check_cuda_result,
    get_allocation_granularity,
)

logger = logging.getLogger(__name__)


class StaleWeightsError(Exception):
    """Raised when weight structure was changed while sleeping.

    This error indicates that a writer acquired the RW lock and modified the
    allocations (different sizes, different tensor layouts) while this reader
    was sleeping. The caller should re-import the model from scratch.

    Note: This does NOT detect content-only changes. A writer can update tensor
    values (e.g., for post-training/fine-tuning) without triggering this error.
    The validation only detects structural changes that would invalidate pointers.
    """

    pass


@dataclass(frozen=True)
class LocalMapping:
    """Immutable record of a local VA mapping."""

    allocation_id: str
    va: int
    size: int
    aligned_size: int
    handle: int  # 0 if unmapped but VA reserved
    tag: str
    access: Literal["ro", "rw"]

    def with_handle(self, handle: int) -> "LocalMapping":
        return LocalMapping(self.allocation_id, self.va, self.size, self.aligned_size, handle, self.tag, self.access)

    def with_access(self, access: Literal["ro", "rw"]) -> "LocalMapping":
        return LocalMapping(self.allocation_id, self.va, self.size, self.aligned_size, self.handle, self.tag, access)


class GMSClientMemoryManager:
    """Unified memory manager that can act as writer or reader.

    Modes:
    - mode="write": acquire RW lock, allocate/map RW, mutate metadata, commit/publish.
    - mode="read": acquire RO lock (READY only), import/map RO, sleep/wake.
    - mode="auto": try RW if available, else wait for RO (for multiprocess architectures).
    """

    def __init__(
        self,
        socket_path: str,
        *,
        mode: Literal["write", "read", "auto"],
        device: int = 0,
        timeout_ms: Optional[int] = None,
    ) -> None:
        self.socket_path = socket_path
        self.device = device
        self._timeout_ms = timeout_ms

        self._client: Optional[GMSRPCClient] = None
        self._mappings: Dict[int, LocalMapping] = {}  # va -> mapping
        self._allocation_id_to_va: Dict[str, int] = {}

        self._sleeping = False
        self._closed = False
        self._preserved_allocation_ids: List[str] = []
        self._published = False
        self._mode: Literal["write", "read"] = "read"  # Updated by _connect

        # VA-stable sleep/wake state
        self._va_preserved = False
        self._last_state_hash: str = ""  # Hash from server, saved on connect/commit

        # Ensure torch is on the right device for subsequent CUDA operations.
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        # Cache granularity for VA alignment
        self.granularity = get_allocation_granularity(device)

        if mode == "write":
            self._connect(lock_type="rw", timeout_ms=timeout_ms)
        elif mode == "read":
            self._connect(lock_type="ro", timeout_ms=timeout_ms)
        elif mode == "auto":
            self._connect(lock_type="rw_or_ro", timeout_ms=timeout_ms)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Must be 'write', 'read', or 'auto'."
            )

    def _connect(
        self,
        *,
        lock_type: Literal["rw", "ro", "rw_or_ro"],
        timeout_ms: Optional[int],
        update_state_hash: bool = True,
    ) -> None:
        self._client = GMSRPCClient(
            self.socket_path, lock_type=lock_type, timeout_ms=timeout_ms
        )
        self._sleeping = False
        # Update mode based on granted lock type (may differ from requested for rw_or_ro)
        granted = self._client.lock_type
        self._mode = "write" if granted == "rw" else "read"
        # Save state hash for stale detection on wake (skip during wake itself)
        if update_state_hash and self._client.committed:
            self._last_state_hash = self._client.get_state_hash()

    @property
    def mode(self) -> Literal["write", "read"]:
        """Current mode of the memory manager."""
        return self._mode

    @property
    def lock_type(self) -> Optional[Literal["rw", "ro"]]:
        if self._client is None:
            return None
        # Use granted lock type (may differ from requested for rw_or_ro mode)
        granted = self._client.lock_type
        return "rw" if granted == "rw" else "ro"

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    @property
    def is_sleeping(self) -> bool:
        return self._sleeping

    # ==================== Metadata convenience ====================

    def metadata_put(
        self, key: str, allocation_id: str, offset_bytes: int, value: bytes
    ) -> bool:
        return self._client_rpc.metadata_put(key, allocation_id, offset_bytes, value)

    def metadata_get(self, key: str) -> Optional[tuple[str, int, bytes]]:
        return self._client_rpc.metadata_get(key)

    def metadata_list(self, prefix: str = "") -> List[str]:
        return self._client_rpc.metadata_list(prefix)

    def metadata_delete(self, key: str) -> bool:
        return self._client_rpc.metadata_delete(key)

    def metadata_delete_prefix(self, prefix: str) -> int:
        """Delete all metadata keys with prefix (RW only)."""
        client = self._client_rpc
        return sum(1 for key in client.metadata_list(prefix) if client.metadata_delete(key))

    # ==================== Allocation operations ====================

    def list_allocations(self, tag: Optional[str] = None) -> List[Dict]:
        """List all allocations on the server."""
        return self._client_rpc.list_allocations(tag)

    def allocate_to_va(
        self,
        size: int,
        va: int,
        aligned_size: int,
        device: int,
        tag: str = "default",
    ) -> str:
        """Allocate on server and map to a pre-reserved VA.

        This method is used for integration with PyTorch's CUDAPluggableAllocator
        where the C++ extension reserves VA before calling this method.

        The C++ extension's my_malloc() reserves VA first, then calls the Python
        callback which should use this method to:
        1. Allocate physical memory on the server
        2. Export the FD
        3. Import and map to the pre-reserved VA via cumem.import_and_map()
        4. Track the mapping in Python for sleep/wake

        Args:
            size: Original requested size
            va: Pre-reserved virtual address from C++ extension
            aligned_size: Aligned size matching the VA reservation
            device: CUDA device index
            tag: Allocation tag

        Returns:
            allocation_id from the server
        """
        self._require_rw()
        client = self._client_rpc

        # Import the C++ extension for import_and_map
        try:
            from gpu_memory_service.client.torch.extensions import (
                _allocator_ext as cumem,
            )
        except ImportError as e:
            raise RuntimeError(
                "Missing CUDA VMM pluggable allocator extension. "
                "Build gpu_memory_service with extensions first."
            ) from e

        # Allocate on server
        allocation_id, server_aligned = client.allocate(aligned_size, tag)
        if int(server_aligned) != int(aligned_size):
            raise RuntimeError(
                f"Alignment mismatch: client={aligned_size} server={server_aligned}"
            )

        # Export FD and map to the pre-reserved VA
        fd = client.export(allocation_id)
        try:
            # Map RW for writer
            cumem.import_and_map(
                int(va), int(fd), int(aligned_size), int(device), False
            )
        finally:
            os.close(fd)

        # Get the handle from the C++ extension for tracking
        # cumem stores handle internally, but we need it for LocalMapping
        # Query the allocation info from cumem
        alloc_infos = cumem.get_all_allocations()
        handle = 0
        for info in alloc_infos:
            if int(info[0]) == int(va):
                handle = int(info[3])
                break

        # Track in Python memory manager for sleep/wake
        self._track_mapping(
            LocalMapping(
                allocation_id=allocation_id,
                va=va,
                size=size,
                aligned_size=aligned_size,
                handle=handle,
                tag=tag,
                access="rw",
            )
        )

        return allocation_id

    def import_allocation(self, allocation_id: str) -> int:
        """Import an existing allocation and map locally.

        In RO mode, maps read-only. In RW mode, maps read-write.
        """
        if allocation_id in self._allocation_id_to_va:
            return self._allocation_id_to_va[allocation_id]

        client = self._client_rpc
        alloc_info = client.get_allocation(allocation_id)
        aligned_size = int(alloc_info.aligned_size)
        size = int(alloc_info.size)
        tag = str(getattr(alloc_info, "tag", "default"))

        fd = client.export(allocation_id)
        try:
            # Import the handle from the FD
            # C signature: cuMemImportFromShareableHandle(handle*, osHandle, handleType)
            # cuda-python: (osHandle, handleType) -> (result, handle)
            result, handle = cuda.cuMemImportFromShareableHandle(
                fd,
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            )
            check_cuda_result(result, "cuMemImportFromShareableHandle")
        finally:
            os.close(fd)

        # Reserve VA
        result, va = cuda.cuMemAddressReserve(aligned_size, self.granularity, 0, 0)
        check_cuda_result(result, "cuMemAddressReserve")

        # Map the handle to the VA
        (result,) = cuda.cuMemMap(va, aligned_size, 0, handle, 0)
        check_cuda_result(result, "cuMemMap")

        access: Literal["ro", "rw"] = "rw" if self.lock_type == "rw" else "ro"
        self._set_access(int(va), aligned_size, access=access)

        self._track_mapping(
            LocalMapping(
                allocation_id=allocation_id,
                va=int(va),
                size=size,
                aligned_size=aligned_size,
                handle=int(handle),
                tag=tag,
                access=access,
            )
        )

        return int(va)

    def clear_all(self) -> int:
        """Clear all allocations on the server (RW only). Local mappings are unmapped first."""
        self._require_rw()
        self._unmap_all()
        return self._client_rpc.clear_all()

    # ==================== Publish / mode switching ====================

    def commit(self) -> bool:
        """Publish weights (RW only).

        Client responsibilities:
        - cudaDeviceSynchronize() before publishing
        - flip local mappings to RO before publishing

        Server responsibilities:
        - transition to COMMITTED
        - close the RW socket (publish + release)
        """
        self._require_rw()

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        # After publishing, prevent further writes locally.
        for va, m in list(self._mappings.items()):
            if m.access != "ro":
                self._set_access(m.va, m.aligned_size, access="ro")
                self._mappings[va] = m.with_access("ro")

        ok = self._client_rpc.commit()
        self._published = bool(ok)
        # _client.commit() closes the socket on success; reflect that here.
        if ok:
            self._client = None
        return bool(ok)

    def switch_to_read(self, timeout_ms: Optional[int] = None) -> None:
        """Acquire an RO lock after publishing.

        This is intended for the common flow where a writer loads weights and then
        becomes a reader for inference.
        """
        if self._closed:
            raise RuntimeError("Memory manager is closed")
        if self._sleeping:
            raise RuntimeError(
                "Cannot switch_to_read() while sleeping; call wake() first"
            )
        if self._client is not None:
            if self.lock_type == "ro":
                return
            raise RuntimeError(
                "switch_to_read() requires the RW connection to be released (call commit() first)"
            )

        eff_timeout = timeout_ms if timeout_ms is not None else self._timeout_ms
        self._connect(lock_type="ro", timeout_ms=eff_timeout)

    # ==================== Sleep / wake (read mode) ====================

    def sleep(self) -> None:
        """Release RO lock and unmap local allocations (VA-stable).

        VAs are preserved during sleep so tensor pointers remain stable.
        On wake, allocations are remapped to the same VAs.
        """
        if self._closed:
            raise RuntimeError("Memory manager is closed")
        if self._sleeping:
            return
        if self.lock_type != "ro":
            raise RuntimeError("sleep() requires RO mode")

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        # Preserve allocation IDs for remapping on wake
        self._preserved_allocation_ids = list(self._allocation_id_to_va.keys())

        # Unmap physical memory but keep VA reservations
        self._unmap_preserving_va()
        self._va_preserved = True

        self._client_rpc.close()
        self._client = None
        self._sleeping = True

    def wake(self, timeout_ms: Optional[int] = None) -> bool:
        """Reacquire RO lock and remap preserved allocations (VA-stable).

        Allocations are remapped to the same VAs they had before sleep,
        ensuring tensor pointers remain valid.

        Args:
            timeout_ms: Timeout for RO lock acquisition.

        Returns:
            True on success.

        Raises:
            TimeoutError: If timeout_ms expires waiting for RO lock.
            StaleWeightsError: If weights were structurally changed while sleeping.
        """
        if self._closed:
            raise RuntimeError("Memory manager is closed")
        if not self._sleeping:
            return True

        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        eff_timeout = timeout_ms if timeout_ms is not None else self._timeout_ms
        self._connect(lock_type="ro", timeout_ms=eff_timeout, update_state_hash=False)

        # Check if state changed while sleeping
        current_hash = self._client_rpc.get_state_hash()
        if self._last_state_hash and current_hash != self._last_state_hash:
            raise StaleWeightsError(
                f"State changed while sleeping: hash {self._last_state_hash[:16]}... -> {current_hash[:16]}..."
            )

        # Remap to preserved VAs
        remapped_count = 0
        failed_count = 0
        total_bytes = 0
        for alloc_id in self._preserved_allocation_ids:
            try:
                va = self._remap_preserved_va(alloc_id)
                mapping = self._mappings.get(va)
                if mapping:
                    total_bytes += mapping.aligned_size
                remapped_count += 1
            except StaleWeightsError:
                raise  # Let StaleWeightsError propagate
            except Exception as e:
                logger.warning(f"Failed to remap {alloc_id}: {e}")
                failed_count += 1

        logger.info(
            f"[GPU Memory Service] Wake complete on device {self.device}: "
            f"remapped {remapped_count} allocations ({total_bytes / (1 << 30):.2f} GiB), "
            f"{failed_count} failed"
        )

        self._sleeping = False
        self._va_preserved = False
        return True

    # ==================== Cleanup ====================

    def close(self) -> None:
        if self._closed:
            return

        # Ensure kernels are done before tearing down mappings.
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        # Release all mappings including preserved VA reservations
        self._unmap_all()

        if self._client is not None:
            self._client.close()
            self._client = None
        self._closed = True
        self._sleeping = False
        self._va_preserved = False
        self._preserved_allocation_ids.clear()

    def __enter__(self) -> "GMSClientMemoryManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        return False

    # ==================== Internals ====================

    @property
    def _client_rpc(self) -> GMSRPCClient:
        """Get connected client or raise. Use instead of _require_connected() + assert."""
        if self._client is None:
            if self._sleeping:
                raise RuntimeError("Memory manager is sleeping")
            raise RuntimeError("Memory manager is not connected")
        return self._client

    def _require_rw(self) -> None:
        """Raise if not in RW mode."""
        if self.lock_type != "rw":
            raise RuntimeError("Operation requires RW mode")

    def _track_mapping(self, m: LocalMapping) -> None:
        self._mappings[m.va] = m
        self._allocation_id_to_va[m.allocation_id] = m.va

    def _set_access(self, va: int, size: int, *, access: Literal["ro", "rw"]) -> None:
        acc = cuda.CUmemAccessDesc()
        acc.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        acc.location.id = self.device
        acc.flags = (
            cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
            if access == "ro"
            else cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        )
        # cuda-python expects a list of access descriptors, not a single object
        (result,) = cuda.cuMemSetAccess(va, size, [acc], 1)
        check_cuda_result(result, "cuMemSetAccess")

    def _unmap_preserving_va(self) -> None:
        """Unmap physical memory but PRESERVE VA reservations for sleep/wake.

        This keeps the VA reservation intact so tensors maintain stable pointers.
        On wake, we can remap to the same VAs.
        """
        unmapped_count = 0
        total_bytes = 0
        for va, mapping in list(self._mappings.items()):
            if mapping.handle == 0:
                continue  # Already unmapped
            try:
                (result,) = cuda.cuMemUnmap(va, mapping.aligned_size)
                if result != cuda.CUresult.CUDA_SUCCESS:
                    logger.warning(f"cuMemUnmap failed for VA 0x{va:x}: error {result}")
                (result,) = cuda.cuMemRelease(mapping.handle)
                if result != cuda.CUresult.CUDA_SUCCESS:
                    logger.warning(f"cuMemRelease failed for handle {mapping.handle}: error {result}")
                self._mappings[va] = mapping.with_handle(0)  # Mark unmapped, VA reserved
                unmapped_count += 1
                total_bytes += mapping.aligned_size
            except Exception as e:
                logger.warning(f"Error unmapping VA 0x{va:x} (preserving reservation): {e}")
        logger.info(
            f"[GPU Memory Service] Unmapped {unmapped_count} allocations ({total_bytes / (1 << 30):.2f} GiB), "
            f"preserving {len(self._mappings)} VA reservations"
        )

    def _remap_preserved_va(self, allocation_id: str) -> int:
        """Remap an allocation to its preserved VA.

        Requires the VA to already be reserved (from before sleep).
        Validates allocation still exists and size matches.

        Returns the VA.
        Raises StaleWeightsError if allocation is missing or size changed.
        """
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        va = self._allocation_id_to_va.get(allocation_id)
        if va is None:
            raise RuntimeError(f"No preserved VA for allocation {allocation_id}")

        mapping = self._mappings.get(va)
        if mapping is None:
            raise RuntimeError(f"No mapping info for VA 0x{va:x}")

        if mapping.handle != 0:
            return va  # Already mapped

        client = self._client_rpc

        # Validate allocation still exists and size matches
        try:
            alloc_info = client.get_allocation(allocation_id)
        except Exception as e:
            raise StaleWeightsError(f"Allocation {allocation_id} no longer exists on server: {e}") from e

        server_aligned_size = int(alloc_info.aligned_size)
        if server_aligned_size != mapping.aligned_size:
            raise StaleWeightsError(
                f"Allocation {allocation_id} size changed: expected {mapping.aligned_size}, got {server_aligned_size}"
            )

        # Re-import the handle
        fd = client.export(allocation_id)
        try:
            # C signature: cuMemImportFromShareableHandle(handle*, osHandle, handleType)
            # cuda-python: (osHandle, handleType) -> (result, handle)
            result, handle = cuda.cuMemImportFromShareableHandle(
                fd,
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            )
            check_cuda_result(result, "cuMemImportFromShareableHandle")
        finally:
            os.close(fd)

        # Map to the SAME VA (which is still reserved)
        (result,) = cuda.cuMemMap(va, mapping.aligned_size, 0, handle, 0)
        check_cuda_result(result, "cuMemMap")

        # Set access permissions based on current lock type
        access: Literal["ro", "rw"] = "rw" if self.lock_type == "rw" else "ro"
        self._set_access(va, mapping.aligned_size, access=access)

        # Synchronize to ensure mapping is complete before any access
        cuda.cuCtxSynchronize()

        # Validate the pointer is accessible (this is what Triton checks)
        result, dev_ptr = cuda.cuPointerGetAttribute(
            cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_POINTER, va
        )
        if result != cuda.CUresult.CUDA_SUCCESS:
            err_result, err_str = cuda.cuGetErrorString(result)
            err_msg = ""
            if err_result == cuda.CUresult.CUDA_SUCCESS and err_str:
                err_msg = (
                    err_str.decode() if isinstance(err_str, bytes) else str(err_str)
                )
            logger.warning(
                f"[GPU Memory Service] cuPointerGetAttribute failed for VA 0x{va:x} after remap: "
                f"error {result} ({err_msg})"
            )
        else:
            logger.debug(
                f"[GPU Memory Service] Remapped VA 0x{va:x} validated OK (device={self.device})"
            )

        # Update mapping with new handle and access
        updated = mapping.with_handle(int(handle))
        self._mappings[va] = updated.with_access(access)

        return va

    def _unmap_all(self) -> None:
        """Unmap and release all local mappings including VA reservations."""
        for va, mapping in list(self._mappings.items()):
            try:
                if mapping.handle != 0:
                    cuda.cuMemUnmap(va, mapping.aligned_size)
                    cuda.cuMemRelease(mapping.handle)
                cuda.cuMemAddressFree(va, mapping.aligned_size)
            except Exception as e:
                logger.warning(f"Error unmapping VA 0x{va:x}: {e}")
        self._mappings.clear()
        self._allocation_id_to_va.clear()
