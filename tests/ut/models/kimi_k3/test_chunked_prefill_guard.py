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

Two things this file is careful about, both learned from defects in this run:

1. **Real config fields, not stand-ins.** An earlier version drove every case
   through ``MagicMock``, which cannot fail when a field is *renamed* upstream --
   ``mock.any_field_name`` always exists. The guard could have silently become
   dead code while the tests stayed green. ``TestAgainstRealVllmConfig`` below
   resolves a real ``VllmConfig`` through ``EngineArgs`` so the field names are
   the ones the engine actually uses.
2. **The override is an allow-list, not a truthiness test.** ``os.environ.get``
   returns a string, so ``if os.environ.get(NAME)`` treats ``"0"`` as enabled.
   A user writing ``=0`` to *switch the guard off* would switch it *on*.
   ``TestOverrideEnvValues`` pins every spelling.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.models.kimi_k3.model import (
    _CHUNKED_PREFILL_OVERRIDE_ENV,
    _check_chunked_prefill_compat,
    _chunked_prefill_override_enabled,
)

CONFIG_PATH = (
    "/home/c00618179/vllm-ascend/.day0/KimiK3ForConditionalGeneration_20260928_1"
    "/impl/dummy_config"
)


def _cfg(attn_res_block_size=12):
    """Deliberately still a stand-in for the *HF* config.

    The point of the guard is that it reads ``attn_res_block_size`` off the HF
    text config; ``TestAgainstRealVllmConfig`` exercises the real object for
    that field, so this helper only serves the negative cases where a mock is
    the clearest way to say "this field is absent".
    """
    cfg = MagicMock()
    cfg.attn_res_block_size = attn_res_block_size
    return cfg


def _vllm_cfg(chunked_prefill):
    vc = MagicMock()
    vc.scheduler_config.enable_chunked_prefill = chunked_prefill
    return vc


class TestOverrideEnvValues(unittest.TestCase):
    """B-2: only explicit "on" spellings may enable the escape hatch."""

    def _enabled(self, value):
        env = {} if value is _UNSET else {_CHUNKED_PREFILL_OVERRIDE_ENV: value}
        with patch.dict(os.environ, env, clear=False):
            if value is _UNSET:
                os.environ.pop(_CHUNKED_PREFILL_OVERRIDE_ENV, None)
            return _chunked_prefill_override_enabled()

    def test_only_explicit_on_values_enable(self):
        for v in ("1", "true", "TRUE", "yes", " 1 "):
            assert self._enabled(v) is True, f"{v!r} should enable the override"

    def test_off_values_do_not_enable(self):
        r"""The defect this pins: `=0` must not turn the guard *on*.

        With the original truthiness test every one of these was truthy
        (they are non-empty strings), so a user setting `=0` to proceed past
        the error got the override engaged -- the opposite of the value's meaning
        and of what the error message promises.
        """
        for v in ("0", "false", "FALSE", "no", "", "2", "off", "disabled"):
            assert self._enabled(v) is False, f"{v!r} must NOT enable the override"

    def test_unset_does_not_enable(self):
        assert self._enabled(_UNSET) is False


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

    def test_override_set_to_zero_still_raises(self):
        """B-2 control: `=0` must leave the guard *active*, not open it.

        Paired with the case above so the suite distinguishes the two spellings
        rather than merely asserting "some override value is handled".
        """
        with patch.dict("os.environ", {_CHUNKED_PREFILL_OVERRIDE_ENV: "0"}):
            with self.assertRaises(ValueError):
                _check_chunked_prefill_compat(_vllm_cfg(True), _cfg())


class TestAgainstRealVllmConfig(unittest.TestCase):
    """B-3: drive the guard with a real ``VllmConfig``, not a mock.

    Field-name drift is the failure mode mocks cannot catch: if upstream renamed
    ``scheduler_config.enable_chunked_prefill`` or the HF config dropped
    ``attn_res_block_size``, the ``MagicMock`` cases above would still pass while
    the guard quietly stopped protecting anything. Resolving through
    ``EngineArgs.create_engine_config()`` uses the same parse path the engine
    does, so a rename shows up as a failure here.
    """

    def _real_config(self, chunked_prefill):
        if not os.path.isdir(CONFIG_PATH):
            pytest.skip("run config not present (run-scoped artifact)")

        from vllm.engine.arg_utils import EngineArgs

        from vllm_ascend import register_model

        register_model()
        return EngineArgs(
            model=CONFIG_PATH,
            load_format="dummy",
            tensor_parallel_size=8,
            max_model_len=4096,
            enforce_eager=True,
            trust_remote_code=True,
            dtype="bfloat16",
            tokenizer_mode="kimi_k3",
            enable_chunked_prefill=chunked_prefill,
        ).create_engine_config()

    def test_real_config_default_reaches_the_guard_and_raises(self):
        """The default launch (chunked prefill on) must be refused."""
        vc = self._real_config(True)
        hf = vc.model_config.hf_text_config

        # Assert the field names the guard depends on, so a rename fails loudly
        # here rather than turning the guard into dead code.
        assert getattr(hf, "attn_res_block_size", None) == 12, (
            "the HF config no longer exposes attn_res_block_size=12; the guard's "
            "key has moved and it would silently stop protecting anything"
        )
        assert vc.scheduler_config.enable_chunked_prefill is True

        with self.assertRaises(ValueError):
            _check_chunked_prefill_compat(vc, hf)

    def test_real_config_with_flag_off_is_silent(self):
        """The Stage 1 command (--no-enable-chunked-prefill) must pass."""
        vc = self._real_config(False)
        hf = vc.model_config.hf_text_config

        assert vc.scheduler_config.enable_chunked_prefill is False
        # No exception is the assertion.
        _check_chunked_prefill_compat(vc, hf)


class _Unset:
    def __repr__(self):
        return "<unset>"


_UNSET = _Unset()


if __name__ == "__main__":
    unittest.main(verbosity=2)
