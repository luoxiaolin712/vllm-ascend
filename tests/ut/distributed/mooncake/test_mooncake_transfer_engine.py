# SPDX-License-Identifier: Apache-2.0
"""UTs for Mooncake TransferEngine protocol/device_name configuration.

Covers the pass-through of the 3rd/4th arguments of
``TransferEngine.initialize(hostname, "P2PHANDSHAKE", protocol, device_name)``,
which are sourced from ``kv_connector_extra_config`` with defaults
``protocol="ascend"`` and ``device_name=""``.

Also covers ``register_memory`` receiving the buffer location
``"npu:{npu_id}"`` required by Mooncake
"""

import sys
import unittest
from unittest.mock import patch

from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE

_MODULE_PATH = "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine"


class _FakeTransferEngine:
    instances: list["_FakeTransferEngine"] = []

    def __init__(self):
        self.initialize_calls: list[tuple] = []
        self.register_memory_calls: list[tuple] = []
        _FakeTransferEngine.instances.append(self)

    def initialize(self, *args):
        self.initialize_calls.append(args)
        return 0

    def register_memory(self, *args):
        self.register_memory_calls.append(args)
        return 0


class _GlobalTEBase(unittest.TestCase):
    def setUp(self):
        _FakeTransferEngine.instances.clear()
        # tests/ut/conftest.py guarantees a "mooncake.engine" entry in
        # sys.modules (mocked on CPU runners, real when mooncake is installed).
        self._patcher = patch.object(sys.modules["mooncake.engine"], "TransferEngine", _FakeTransferEngine)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()


class TestGlobalTEProtocolDeviceName(_GlobalTEBase):
    def test_default_protocol_and_device_name(self):
        engine = GlobalTE().get_transfer_engine("10.0.0.1")
        self.assertEqual(
            _FakeTransferEngine.instances[0].initialize_calls,
            [("10.0.0.1", "P2PHANDSHAKE", "ascend", "")],
        )
        self.assertIs(engine, _FakeTransferEngine.instances[0])

    def test_device_name_none_normalizes_to_empty_string(self):
        GlobalTE().get_transfer_engine("10.0.0.1", device_name=None)
        self.assertEqual(
            _FakeTransferEngine.instances[0].initialize_calls,
            [("10.0.0.1", "P2PHANDSHAKE", "ascend", "")],
        )

    def test_custom_protocol_and_device_name(self):
        device_name = "mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4"
        GlobalTE().get_transfer_engine("10.0.0.1", device_name=device_name, protocol="rdma")
        self.assertEqual(
            _FakeTransferEngine.instances[0].initialize_calls,
            [("10.0.0.1", "P2PHANDSHAKE", "rdma", device_name)],
        )

    def test_engine_is_created_only_once(self):
        global_te = GlobalTE()
        first = global_te.get_transfer_engine("10.0.0.1", device_name="mlx5_bond_1", protocol="rdma")
        second = global_te.get_transfer_engine("10.0.0.1")
        self.assertIs(first, second)
        self.assertEqual(len(_FakeTransferEngine.instances), 1)
        # The second call must not re-initialize the engine.
        self.assertEqual(len(_FakeTransferEngine.instances[0].initialize_calls), 1)


class TestGlobalTERegisterBufferLocation(_GlobalTEBase):
    def _make_global_te(self) -> GlobalTE:
        global_te = GlobalTE()
        global_te.transfer_engine = _FakeTransferEngine()
        return global_te

    def test_register_memory_receives_npu_location(self):
        global_te = self._make_global_te()
        with patch(f"{_MODULE_PATH}.torch.npu.current_device", return_value=3):
            global_te.register_buffer([100, 200], [10, 20])
        self.assertEqual(
            _FakeTransferEngine.instances[-1].register_memory_calls,
            [(100, 10, "npu:3"), (200, 20, "npu:3")],
        )

    def test_register_buffer_is_idempotent(self):
        global_te = self._make_global_te()
        with patch(f"{_MODULE_PATH}.torch.npu.current_device", return_value=0):
            global_te.register_buffer([100], [10])
            global_te.register_buffer([300], [30])
        self.assertEqual(_FakeTransferEngine.instances[-1].register_memory_calls, [(100, 10, "npu:0")])


if __name__ == "__main__":
    unittest.main()
