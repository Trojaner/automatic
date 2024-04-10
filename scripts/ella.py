import gradio as gr
import huggingface_hub as hf
import safetensors
import torch
from collections import OrderedDict
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from modules import shared, scripts, processing, devices
from transformers import T5EncoderModel, T5Tokenizer
from typing import Any, Optional, Union

REPO = "QQGYLab/ELLA"
MODELS = {
    "None": "None",
    "sd1.5-tsc-t5xl": "ella-sd1.5-tsc-t5xl.safetensors",
}


class AdaLayerNorm(torch.nn.Module):
    def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None):
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

        self.norm = torch.nn.LayerNorm(
            embedding_dim, elementwise_affine=False, eps=1e-6
        )

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class SquaredReLU(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))


class PerceiverAttentionBlock(torch.nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", torch.nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        normed_latents = self.ln_1(latents, timestep_embedding)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
        )
        latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        return latents


class PerceiverResampler(torch.nn.Module):
    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        output_dim=None,
        input_dim=None,
        time_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latents = torch.nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
        self.time_aware_linear = torch.nn.Linear(
            time_embedding_dim or width, width, bias=True
        )

        if self.input_dim is not None:
            self.proj_in = torch.nn.Linear(input_dim, width)

        self.perceiver_blocks = torch.nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    width, heads, time_embedding_dim=time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = torch.nn.Sequential(
                torch.nn.Linear(width, output_dim), torch.nn.LayerNorm(output_dim)
            )

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor = None):
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
        latents = learnable_latents + self.time_aware_linear(
            torch.torch.nn.functional.silu(timestep_embedding)
        )
        if self.input_dim is not None:
            x = self.proj_in(x)
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        return latents


class T5TextEmbedder(torch.nn.Module):
    def __init__(self, pretrained_path="google/flan-t5-xl", max_length=None):
        super().__init__()

        device_map = {"": shared.device}
        dtype = torch.bfloat16 if devices.dtype == torch.bfloat16 else torch.float16

        self.model = T5EncoderModel.from_pretrained(
            pretrained_path,
            torch_dtype=dtype,
            device_map=device_map,
            cache_dir=shared.opts.diffusers_dir,
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_path, cache_dir=shared.opts.diffusers_dir
        )
        self.max_length = max_length

    def forward(
        self, caption, text_input_ids=None, attention_mask=None, max_length=None
    ):
        if max_length is None:
            max_length = self.max_length

        if text_input_ids is None or attention_mask is None:
            if max_length is not None:
                text_inputs = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
            else:
                text_inputs = self.tokenizer(
                    caption, return_tensors="pt", add_special_tokens=True
                )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
        text_input_ids = text_input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        outputs = self.model(text_input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state
        return embeddings


class ELLA(torch.nn.Module):
    def __init__(
        self,
        time_channel=320,
        time_embed_dim=768,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        width=768,
        layers=6,
        heads=8,
        num_latents=64,
        input_dim=2048,
    ):
        super().__init__()

        self.position = Timesteps(
            time_channel, flip_sin_to_cos=True, downscale_freq_shift=0
        )

        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,
        )

        self.connector = PerceiverResampler(
            width=width,
            layers=layers,
            heads=heads,
            num_latents=num_latents,
            input_dim=input_dim,
            time_embedding_dim=time_embed_dim,
        )

    def forward(self, text_encode_features, timesteps):
        device = text_encode_features.device
        dtype = text_encode_features.dtype

        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        ori_time_feature = (
            ori_time_feature.unsqueeze(dim=1)
            if ori_time_feature.ndim == 2
            else ori_time_feature
        )

        ori_time_feature = ori_time_feature.expand(len(text_encode_features), -1, -1)
        time_embedding = self.time_embedding(ori_time_feature)

        encoder_hidden_states = self.connector(
            text_encode_features, timestep_embedding=time_embedding
        )

        return encoder_hidden_states


class ELLAProxyUNet(torch.torch.nn.Module):
    def __init__(self, ella, unet):
        super().__init__()
        # In order to still use the diffusers pipeline, including various workaround

        self.ella = ella
        self.unet = unet
        self.config = unet.config
        self.dtype = unet.dtype
        self.device = unet.device

        self.flexible_max_length_workaround = None

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        added_cond_kwargs: Optional[dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if self.flexible_max_length_workaround is not None:
            time_aware_encoder_hidden_state_list = []
            for i, max_length in enumerate(self.flexible_max_length_workaround):
                time_aware_encoder_hidden_state_list.append(
                    self.ella(encoder_hidden_states[i : i + 1, :max_length], timestep)
                )
            # No matter how many tokens are text features, the ella output must be 64 tokens.
            time_aware_encoder_hidden_states = torch.cat(
                time_aware_encoder_hidden_state_list, dim=0
            )
        else:
            time_aware_encoder_hidden_states = self.ella(
                encoder_hidden_states, timestep
            )

        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=time_aware_encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )


def load_ella(filename, device, dtype):
    ella = ELLA()
    safetensors.torch.load_model(ella, filename, strict=True)
    ella.to(device, dtype=dtype)
    return ella


def load_ella_for_pipe(pipe, ella):
    pipe.unet = ELLAProxyUNet(ella, pipe.unet)


def offload_ella_for_pipe(pipe):
    if hasattr(pipe.unet, "unet"):
        pipe.unet = pipe.unet.unet


class Script(scripts.Script):
    t5_encoder = None
    ella = None

    def title(self):
        return "ELLA"

    def show(self, is_img2img):
        return shared.backend == shared.Backend.DIFFUSERS

    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML(
                '<a href="https://github.com/TencentQQGYLab/ELLA/">&nbsp ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment</a><br><span>Select a model for ELLA</span><br>'
            )
        with gr.Row():
            enabled = gr.Checkbox(label="Enabled", value=True)
            model = gr.Dropdown(label="Model", choices=MODELS.keys(), value="None")
        return enabled, model

    def run(self, p: processing.StableDiffusionProcessing, enabled, model):

        if not enabled or model == "None":
            return

        if shared.sd_model_type != "sd":
            shared.log.error(
                f"ELLA: incorrect base model: {shared.sd_model.__class__.__name__}"
            )
            return

        if self.ella is None:
            model_ckpt_filename = MODELS[model]
            ella_path = hf.hf_hub_download(
                repo_id=REPO,
                filename=model_ckpt_filename,
                cache_dir=shared.opts.diffusers_dir,
            )

            if ella_path is None:
                shared.log.error(
                    f"ELLA model download failed: model={model} file={ella_path}"
                )
                return

            self.ella = load_ella(ella_path, shared.device, devices.dtype)

        if self.t5_encoder is None:
            self.t5_encoder = T5TextEmbedder()

        # hijack unet

        try:
            load_ella_for_pipe(shared.sd_model, self.ella)

            prompt_embeds = self.t5_encoder([p.prompt], max_length=None).to(
                shared.device, devices.dtype
            )

            negative_prompt_embeds = self.t5_encoder(
                [p.negative_prompt], max_length=None
            ).to(shared.device, devices.dtype)

            # diffusers pipeline concatenate `prompt_embeds` too early...
            # https://github.com/huggingface/diffusers/blob/b6d7e31d10df675d86c6fe7838044712c6dca4e9/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L913

            shared.sd_model.unet.flexible_max_length_workaround = [
                negative_prompt_embeds.size(1)
            ] * p.batch_size + [prompt_embeds.size(1)] * p.batch_size

            # pad embeds
            max_length = max([prompt_embeds.size(1), negative_prompt_embeds.size(1)])
            b, _, d = prompt_embeds.shape

            prompt_embeds = torch.cat(
                [
                    prompt_embeds,
                    torch.zeros(
                        (b, max_length - prompt_embeds.size(1), d),
                        device=shared.device,
                        dtype=devices.dtype,
                    ),
                ],
                dim=1,
            )

            negative_prompt_embeds = torch.cat(
                [
                    negative_prompt_embeds,
                    torch.zeros(
                        (b, max_length - negative_prompt_embeds.size(1), d),
                        device=shared.device,
                        dtype=devices.dtype,
                    ),
                ],
                dim=1,
            )

            p.prompt_embeds = [prompt_embeds]
            p.negative_embeds = [negative_prompt_embeds]

            processed: processing.Processed = processing.process_images(
                p
            )  # runs processing using main loop
        finally:
            offload_ella_for_pipe(shared.sd_model)

        devices.torch_gc()
        return processed
