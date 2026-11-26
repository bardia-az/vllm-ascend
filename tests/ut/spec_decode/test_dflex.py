# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.worker.v2.spec_decode import init_speculator


@pytest.mark.parametrize("layout", [{"sample_from_anchor": True}, {"sample_from_anchor": False}, {}])
def test_dflex_v2_dispatch(layout):
    draft = SimpleNamespace(architectures=["DFlashDraftModel"], hf_config=SimpleNamespace(dflash_config=layout))
    spec = SimpleNamespace(
        method="dflex",
        enforce_eager=False,
        draft_model_config=draft,
        use_dspark=lambda: False,
        use_dflash=lambda: False,
        use_eagle=lambda: True,
    )
    cfg = SimpleNamespace(speculative_config=spec)
    device = torch.device("cpu")
    with (
        patch("vllm_ascend.worker.v2.spec_decode.dflash.speculator.AscendDFlashSpeculator") as dflash,
        patch("vllm_ascend.worker.v2.spec_decode.eagle.speculator.AscendEagleSpeculator"),
    ):
        if layout.get("sample_from_anchor"):
            assert init_speculator(cfg, device) is dflash.return_value
            dflash.assert_called_once_with(cfg, device)
        else:
            with pytest.raises(ValueError, match="sample_from_anchor=True"):
                init_speculator(cfg, device)


def test_dflex_v1_requires_v2():
    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=1"):
        get_spec_decode_method("dflex", None, torch.device("cpu"), None)
