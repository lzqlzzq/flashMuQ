import gc
import statistics
import time
import unittest

import torch
from easydict import EasyDict
from torch.nn.attention.flex_attention import BlockMask
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.models.wav2vec2_conformer.configuration_wav2vec2_conformer import (
    Wav2Vec2ConformerConfig,
)
from muq.muq.modules.flash_conformer import (
    Wav2Vec2ConformerEncoder,
    Wav2Vec2ConformerSelfAttention,
    _selected_sdpa_backend,
    _prepare_bidirectional_attention_mask,
    backend_available,
)


def attention_config(implementation="sdpa", dropout=0.0):
    return EasyDict(
        hidden_size=8,
        num_attention_heads=2,
        position_embeddings_type="rotary",
        attention_dropout=dropout,
        is_causal=False,
        _attn_implementation=implementation,
    )


def identity_rotary(sequence_length, head_size):
    return torch.stack(
        (
            torch.ones(sequence_length, 1, 1, head_size),
            torch.zeros(sequence_length, 1, 1, head_size),
        )
    )


def muq_large_encoder_config(implementation="sdpa"):
    config = Wav2Vec2ConformerConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=12,
        num_attention_heads=16,
        conv_depthwise_kernel_size=31,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        position_embeddings_type="rotary",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        conformer_conv_dropout=0.1,
        layerdrop=0.0,
        max_source_positions=5000,
        rotary_embedding_base=10000,
    )
    config._attn_implementation = implementation
    return config


def available_accelerator():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_accelerator_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def current_allocated_memory(device):
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    return torch.mps.current_allocated_memory()


def driver_allocated_memory(device):
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()
    return None


def benchmark_full_encoder(implementation, device):
    torch.manual_seed(29)
    encoder = Wav2Vec2ConformerEncoder(
        muq_large_encoder_config(implementation)
    ).to(device=device, dtype=torch.float16).eval()
    hidden_states = torch.randn(
        1,
        750,
        1024,
        device=device,
        dtype=torch.float16,
    )
    if implementation == "sdpa":
        query = hidden_states.view(1, 750, 16, 64).transpose(1, 2)
        actual_backend = _selected_sdpa_backend(query)
        if not backend_available(actual_backend, query):
            raise RuntimeError(
                f"Reported SDPA backend {actual_backend!r} is not available"
            )
        del query
    else:
        actual_backend = implementation

    with torch.inference_mode():
        for _ in range(2):
            output = encoder(hidden_states).last_hidden_state
        synchronize(device)
        del output
        empty_accelerator_cache(device)

        allocated_before = current_allocated_memory(device)
        driver_before = driver_allocated_memory(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        elapsed_ms = []
        output = None
        for _ in range(5):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = encoder(hidden_states).last_hidden_state
                end.record()
                synchronize(device)
                elapsed_ms.append(start.elapsed_time(end))
            else:
                synchronize(device)
                start_time = time.perf_counter()
                output = encoder(hidden_states).last_hidden_state
                synchronize(device)
                elapsed_ms.append((time.perf_counter() - start_time) * 1000.0)

        allocated_after = current_allocated_memory(device)
        peak_delta = None
        if device.type == "cuda":
            peak_delta = max(
                0,
                torch.cuda.max_memory_allocated(device) - allocated_before,
            )
        metrics = {
            "implementation": implementation,
            "actual_backend": actual_backend,
            "allocated_before": allocated_before,
            "allocated_after": allocated_after,
            "allocated_delta": allocated_after - allocated_before,
            "peak_delta": peak_delta,
            "driver_before": driver_before,
            "driver_after": driver_allocated_memory(device),
            "median_forward_ms": statistics.median(elapsed_ms),
        }
        output_for_comparison = output.float().cpu()

    del output
    del hidden_states
    del encoder
    gc.collect()
    empty_accelerator_cache(device)
    return metrics, output_for_comparison


class FlashConformerAttentionTest(unittest.TestCase):
    def test_backend_available_reports_concrete_sdpa_dispatch(self):
        query = torch.randn(1, 2, 5, 4)
        selected = _selected_sdpa_backend(query)

        self.assertTrue(backend_available("sdpa", query))
        self.assertTrue(backend_available(selected, query))
        with self.assertRaisesRegex(ValueError, "Unknown PyTorch SDPA backend"):
            backend_available("not_a_backend", query)

    def test_sdpa_preserves_batch_time_head_dimensions_and_gradients(self):
        torch.manual_seed(3)
        module = Wav2Vec2ConformerSelfAttention(attention_config()).eval()
        hidden_states = torch.randn(2, 5, 8, requires_grad=True)
        rotary = identity_rotary(sequence_length=5, head_size=4)

        output, probabilities = module(
            hidden_states,
            relative_position_embeddings=rotary,
        )

        self.assertEqual(output.shape, (2, 5, 8))
        self.assertIsNone(probabilities)
        output.square().mean().backward()
        self.assertEqual(hidden_states.grad.shape, hidden_states.shape)
        self.assertTrue(torch.isfinite(hidden_states.grad).all())

    def test_eval_disables_attention_dropout(self):
        torch.manual_seed(5)
        module = Wav2Vec2ConformerSelfAttention(
            attention_config(dropout=0.75)
        ).eval()
        hidden_states = torch.randn(2, 5, 8)
        rotary = identity_rotary(sequence_length=5, head_size=4)

        first, _ = module(hidden_states, relative_position_embeddings=rotary)
        second, _ = module(hidden_states, relative_position_embeddings=rotary)

        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_training_uses_configured_attention_dropout(self):
        torch.manual_seed(7)
        module = Wav2Vec2ConformerSelfAttention(
            attention_config(implementation="eager", dropout=0.5)
        ).train()
        hidden_states = torch.randn(2, 5, 8)
        rotary = identity_rotary(sequence_length=5, head_size=4)

        torch.manual_seed(11)
        first, first_probabilities = module(
            hidden_states,
            relative_position_embeddings=rotary,
            output_attentions=True,
        )
        torch.manual_seed(13)
        second, second_probabilities = module(
            hidden_states,
            relative_position_embeddings=rotary,
            output_attentions=True,
        )

        self.assertEqual(first_probabilities.shape, (2, 2, 5, 5))
        self.assertFalse(torch.equal(first, second))
        self.assertFalse(torch.equal(first_probabilities, second_probabilities))

    def test_mask_formatter_uses_backend_specific_dimensions(self):
        hidden_states = torch.randn(2, 5, 8)
        padding_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]]
        )

        sdpa_mask = _prepare_bidirectional_attention_mask(
            attention_config("sdpa"), padding_mask, hidden_states
        )
        eager_mask = _prepare_bidirectional_attention_mask(
            attention_config("eager"), padding_mask, hidden_states
        )
        flash_mask = _prepare_bidirectional_attention_mask(
            attention_config("flash_attention_2"), padding_mask, hidden_states
        )
        self.assertEqual(sdpa_mask.shape, (2, 1, 5, 5))
        self.assertEqual(sdpa_mask.dtype, torch.bool)
        self.assertEqual(eager_mask.shape, (2, 1, 5, 5))
        self.assertTrue(eager_mask.is_floating_point())
        self.assertEqual(flash_mask.shape, (2, 5))
        self.assertEqual(flash_mask.dtype, torch.bool)

    @unittest.skipUnless(torch.cuda.is_available(), "FlexAttention backend test requires CUDA")
    def test_flex_formatter_returns_block_mask_when_backend_is_available(self):
        hidden_states = torch.randn(2, 5, 8)
        padding_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]]
        )
        flex_mask = _prepare_bidirectional_attention_mask(
            attention_config("flex_attention"),
            padding_mask.cuda(),
            hidden_states.cuda(),
        )
        self.assertIsInstance(flex_mask, BlockMask)

    def test_padding_keys_receive_zero_attention_probability(self):
        torch.manual_seed(19)
        config = attention_config("eager")
        module = Wav2Vec2ConformerSelfAttention(config).eval()
        hidden_states = torch.randn(1, 5, 8)
        padding_mask = torch.tensor([[True, True, True, False, False]])
        formatted_mask = _prepare_bidirectional_attention_mask(
            config,
            padding_mask,
            hidden_states,
        )

        _, probabilities = module(
            hidden_states,
            attention_mask=formatted_mask,
            relative_position_embeddings=identity_rotary(5, 4),
            output_attentions=True,
        )

        torch.testing.assert_close(
            probabilities[..., 3:],
            torch.zeros_like(probabilities[..., 3:]),
            rtol=0.0,
            atol=0.0,
        )

    def test_none_and_all_valid_masks_may_skip_formatting(self):
        hidden_states = torch.randn(2, 5, 8)
        config = attention_config("sdpa")

        self.assertIsNone(
            _prepare_bidirectional_attention_mask(config, None, hidden_states)
        )
        self.assertIsNone(
            _prepare_bidirectional_attention_mask(
                config,
                torch.ones(2, 5, dtype=torch.bool),
                hidden_states,
            )
        )

    def test_padding_mask_cannot_be_silently_dropped(self):
        implementation = "test_dropped_padding_mask"
        original = ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.get(implementation)
        ALL_MASK_ATTENTION_FUNCTIONS.register(implementation, lambda **kwargs: None)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register(
            implementation,
            lambda module, query, key, value, attention_mask, **kwargs: (
                query.transpose(1, 2),
                None,
            ),
        )
        self.addCleanup(
            lambda: ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(
                implementation, None
            )
            if original is None
            else ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.__setitem__(
                implementation, original
            )
        )
        self.addCleanup(
            lambda: ALL_ATTENTION_FUNCTIONS._global_mapping.pop(implementation, None)
        )

        with self.assertRaisesRegex(RuntimeError, "dropped a padding mask"):
            _prepare_bidirectional_attention_mask(
                attention_config(implementation),
                torch.tensor([[True, True, False]]),
                torch.randn(1, 3, 8),
            )

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown attention implementation"):
            Wav2Vec2ConformerSelfAttention(attention_config("not-a-backend"))

    def test_full_muq_large_sdpa_encoder_matches_eager_with_padding(self):
        input_generator = torch.Generator().manual_seed(17)
        hidden_states = torch.randn(2, 25, 1024, generator=input_generator)
        attention_mask = torch.tensor(
            [
                [True] * 20 + [False] * 5,
                [True] * 23 + [False] * 2,
            ]
        )

        torch.manual_seed(23)
        reference = Wav2Vec2ConformerEncoder(
            muq_large_encoder_config("eager")
        ).to(dtype=torch.float16).eval()
        reference_keys = tuple(reference.state_dict())
        with torch.inference_mode():
            reference_output = reference(
                hidden_states.to(dtype=torch.float16),
                attention_mask=attention_mask,
            ).last_hidden_state.float()
        del reference
        gc.collect()

        torch.manual_seed(23)
        candidate = Wav2Vec2ConformerEncoder(
            muq_large_encoder_config("sdpa")
        ).to(dtype=torch.float16).eval()
        self.assertEqual(reference_keys, tuple(candidate.state_dict()))
        with torch.inference_mode():
            candidate_output = candidate(
                hidden_states.to(dtype=torch.float16),
                attention_mask=attention_mask,
            ).last_hidden_state.float()

        self.assertEqual(candidate_output.shape, (2, 25, 1024))
        torch.testing.assert_close(
            candidate_output,
            reference_output,
            rtol=5e-2,
            atol=5e-2,
        )

    @unittest.skipUnless(
        available_accelerator() is not None,
        "MuQ-large attention benchmark requires CUDA or MPS",
    )
    def test_full_muq_large_eager_vs_sdpa_memory_and_forward_time(self):
        device = available_accelerator()
        eager_metrics, eager_output = benchmark_full_encoder("eager", device)
        sdpa_metrics, sdpa_output = benchmark_full_encoder("sdpa", device)

        torch.testing.assert_close(
            sdpa_output,
            eager_output,
            rtol=5e-2,
            atol=5e-2,
        )
        memory_ratio = (
            sdpa_metrics["allocated_delta"] / eager_metrics["allocated_delta"]
            if eager_metrics["allocated_delta"] != 0
            else float("nan")
        )
        time_ratio = (
            sdpa_metrics["median_forward_ms"]
            / eager_metrics["median_forward_ms"]
        )
        print(
            "muq_large_attention_benchmark "
            f"device={device.type} batch=1 sequence_length=750 dtype=float16"
        )
        for metrics in (eager_metrics, sdpa_metrics):
            print(
                "muq_large_attention_backend "
                f"configured_backend={metrics['implementation']} "
                f"actual_backend={metrics['actual_backend']} "
                f"allocated_before={metrics['allocated_before']} "
                f"allocated_after={metrics['allocated_after']} "
                f"allocated_delta={metrics['allocated_delta']} "
                f"peak_delta={metrics['peak_delta']} "
                f"driver_before={metrics['driver_before']} "
                f"driver_after={metrics['driver_after']} "
                f"median_forward_ms={metrics['median_forward_ms']:.3f}"
            )
        print(
            "muq_large_attention_comparison "
            f"sdpa_over_eager_allocated_delta={memory_ratio:.6f} "
            f"sdpa_over_eager_forward_time={time_ratio:.6f}"
        )


if __name__ == "__main__":
    unittest.main()
