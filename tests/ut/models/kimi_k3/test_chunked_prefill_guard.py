#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""S-2: the chunked-prefill guard for Kimi-K3's cross-layer residual chain.

Upstream defaults ``enable_chunked_prefill`` to True, and chunked prefill
restarts the per-request ``block_residual`` accumulator on every chunk -- so the
default launch silently mixes later chunks against an incomplete chain. The
guard turns that into a startup error.

Scope matters as much here as the check itself: a guard keyed too broadly would
reject configurations that are perfectly fine. These tests therefore pin both
directions -- fires for the offending combination, silent for every other one.
"""

import unittest
from unittest.mock import MagicMock, patch

from vllm_ascend.models.kimi_k3.model import (
    _CHUNKED_PREFILL_OVERRIDE_ENV,
    _check_chunked_prefill_compat,
)


def _cfg(attn_res_block_size=12):
    cfg = MagicMock()
    cfg.attn_res_block_size = attn_res_block_size
    return cfg


def _vllm_cfg(chunked_prefill):
    vc = MagicMock()
    vc.scheduler_config.enable_chunked_prefill = chunked_prefill
    return vc


class TestChunkedPrefillGuard(unittest.TestCase):
    def test_fires_when_residual_chain_and_chunked_prefill_both_present(self):
        """The offending pair must raise, and the message must say how to fix it."""
        with self.assertRaises(ValueError) as ctx:
            _check_chunked_prefill_compat(_vllm_cfg(True), _cfg(attn_res_block_size=12))
        msg = str(ctx.exception)
        assert "--no-enable-chunked-prefill" in msg, (
            "the error must name the flag that fixes it; got: " + msg
        )

    def test_silent_when_chunked_prefill_disabled(self):
        """The Stage 1 configuration: no exception, no warning."""
        with patch("vllm_ascend.models.kimi_k3.model.logger") as mock_logger:
            _check_chunked_prefill_compat(_vllm_cfg(False), _cfg())
        mock_logger.warning.assert_not_called()

    def test_silent_for_a_config_without_the_residual_chain(self):
        """Scope: a checkpoint with no ``attn_res_block_size`` is untouched.

        This is the "zero impact on existing models" evidence -- the guard keys
        on the marker that creates the chain, so a config lacking it is not
        rejected even with chunked prefill on.
        """
        with patch("vllm_ascend.models.kimi_k3.model.logger") as mock_logger:
            _check_chunked_prefill_compat(_vllm_cfg(True), _cfg(attn_res_block_size=None))
        mock_logger.warning.assert_not_called()

    def test_silent_when_scheduler_config_is_absent(self):
        """A config without a scheduler_config must not blow up here."""
        vc = MagicMock(spec=[])
        del vc.scheduler_config
        _check_chunked_prefill_compat(vc, _cfg())

    def test_override_env_downgrades_to_a_warning(self):
        """The documented escape hatch lets the run proceed, loudly."""
        with (
            patch.dict("os.environ", {_CHUNKED_PREFILL_OVERRIDE_ENV: "1"}),
            patch("vllm_ascend.models.kimi_k3.model.logger") as mock_logger,
        ):
            _check_chunked_prefill_compat(_vllm_cfg(True), _cfg())
        mock_logger.warning.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
