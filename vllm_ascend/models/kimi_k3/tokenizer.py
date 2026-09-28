# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is a part of the vllm-ascend project.
#
"""Self-hosted `kimi_k3` tokenizer mode.

Why this exists
---------------
The Kimi-K3 checkpoint ships **no Jinja chat template**: there is no `.jinja`
file and `tokenizer_config.json` has no `chat_template` key. Instead the
checkpoint's tokenizer (`tokenization_kimi.TikTokenTokenizer`, loaded through
`trust_remote_code`) implements `apply_chat_template` in Python and drives the
XTML segment encoder in `encoding_k3.py`.

vLLM's chat entrypoint only calls that method when the tokenizer exposes a
truthy `chat_template`. With `chat_template` absent, transformers refuses the
default ("As of transformers v4.44, default chat template is no longer
allowed"), so chat requests fail with HTTP 400 even though a complete encoder is
present.

This wrapper supplies the missing marker and keeps the real work where it
belongs -- in the checkpoint's own encoder:

* `chat_template` is set to a **sentinel string, never to Jinja**. The template
  is never rendered; its only job is to satisfy the presence check. (Writing
  real Jinja here would mean re-implementing XTML in a template language, and a
  template that tried to delegate would recurse, since delegation re-enters
  `apply_chat_template`.)
* `apply_chat_template` is delegated to the checkpoint tokenizer, preserving
  every argument the caller passes (`tools`, `thinking_effort`, `thinking`,
  `response_format`, ...). The XTML semantics stay in one place.

This is deliberately a *thin* adapter -- the design calls for the service trio
to be self-hosted rather than relying on upstream registrations, and this is the
tokenizer leg of it.
"""

from typing import Any

#: Marker that satisfies vLLM's "tokenizer must define a chat template" check.
#: Never compiled or rendered -- see module docstring.
_NON_JINJA_SENTINEL = "kimi-k3-native-chat-encoder (see encoding_k3.py; not Jinja)"


class KimiK3Tokenizer:
    """Adapter exposing the checkpoint's programmatic XTML chat encoder."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "KimiK3Tokenizer":
        """Load the checkpoint tokenizer and wrap it.

        `trust_remote_code` must stay on: the checkpoint's
        `tokenization_kimi.TikTokenTokenizer` overrides `encode` with a different
        signature than the base class, so the generic tokenizer cannot stand in
        for it.
        """
        from transformers import AutoTokenizer

        inner = AutoTokenizer.from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            cache_dir=download_dir,
            **kwargs,
        )
        return cls(inner)

    # -- chat ------------------------------------------------------------
    @property
    def chat_template(self) -> str:
        return _NON_JINJA_SENTINEL

    def get_chat_template(self, chat_template=None, tools=None) -> str | None:
        """Return the template marker for the renderer's resolution step.

        vLLM's `hf` renderer calls this first and, if it returns None, raises
        "default chat template is no longer allowed". Returning the sentinel
        satisfies that gate; `apply_chat_template` below then bypasses Jinja
        entirely and delegates to the checkpoint encoder.
        """
        return _NON_JINJA_SENTINEL

    def apply_chat_template(self, *args, **kwargs):
        """Delegate to the checkpoint's own `apply_chat_template`.

        All arguments are forwarded verbatim so the encoder sees the same
        `tools` / `thinking_effort` / `response_format` it expects. The
        sentinel `chat_template` is dropped: it exists only to pass the
        renderer's presence check, and forwarding it would make the checkpoint
        encoder treat it as a real template.

        `chat_template` may arrive positionally (vLLM passes
        `apply_chat_template(conversation, chat_template=..., **kwargs)`), so
        both spellings are stripped.
        """
        kwargs.pop("chat_template", None)
        if len(args) > 1:
            # Drop a positional chat_template (arg 1) without disturbing the
            # remaining positional args.
            args = (args[0],) + tuple(a for a in args[2:])
        return self._inner.apply_chat_template(*args, **kwargs)

    # -- transparent delegation -----------------------------------------
    def __getattr__(self, name: str):
        """Forward everything else to the wrapped tokenizer.

        `__getattr__` only fires for attributes this class does not define, so
        `chat_template` and `apply_chat_template` above win and the rest
        (encode/decode/vocab/...) pass straight through.

        The `_inner` guard matters: unpickling calls `__getattr__` (e.g. for
        `__setstate__`) *before* `__init__` has run, so without this check the
        lookup would re-enter this method forever and raise RecursionError.
        """
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def max_chars_per_token(self) -> int:
        """Longest single token string, part of the `TokenizerLike` protocol.

        vLLM uses it to size the prompt-length to max-token budget split. The
        checkpoint tokenizer does not implement it, so it is derived from the
        vocabulary here (same definition the generic HF adapter uses).
        """
        cached = self.__dict__.get("_max_chars_per_token")
        if cached is not None:
            return cached
        vocab = self._inner.get_vocab()
        value = max((len(tok) for tok in vocab), default=1)
        self.__dict__["_max_chars_per_token"] = value
        return value

    @property
    def max_token_id(self) -> int:
        """Largest token id, part of the `TokenizerLike` protocol.

        Derived from the vocabulary, taking `vocab_size` into account because
        some tokenizers include added special tokens in `vocab_size` but not in
        `get_vocab()` -- the same reconciliation the generic HF adapter does.
        """
        cached = self.__dict__.get("_max_token_id")
        if cached is not None:
            return cached
        vocab = self._inner.get_vocab()
        value = max(vocab.values()) if vocab else 0
        inner_vocab_size = getattr(self._inner, "vocab_size", None)
        if isinstance(inner_vocab_size, int):
            value = max(value, inner_vocab_size)
        self.__dict__["_max_token_id"] = value
        return value

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)
