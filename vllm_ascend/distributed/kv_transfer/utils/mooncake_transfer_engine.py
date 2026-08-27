import threading

import torch
from vllm.logger import logger

# Default protocol used when `protocol` are not set in `kv_connector_extra_config`.
DEFAULT_MOONCAKE_PROTOCOL = "ascend"

class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def get_transfer_engine(
        self,
        hostname: str,
        device_name: str | None = None,
        protocol: str = DEFAULT_MOONCAKE_PROTOCOL,
    ):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                            "to run vLLM with MooncakeConnector."
                        ) from e
                    self.transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    logger.info(
                        "Mooncake TransferEngine initialize: hostname=%s, protocol=%s, device_name=%s",
                        hostname,
                        protocol,
                        device_name,
                    )
                    ret_value = self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", protocol, device_name)
                    if ret_value != 0:
                        raise RuntimeError(f"TransferEngine initialization failed with ret_value: {ret_value}")
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            # Location ("npu:{id}") tells Mooncake where the buffers live so the
            # engine can pick topology-aware transfer paths.
            location = f"npu:{torch.npu.current_device()}"
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size, location)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self.is_register_buffer = True


global_te = GlobalTE()
