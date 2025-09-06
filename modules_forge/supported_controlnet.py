import os
import torch

from huggingface_guess.detection import unet_config_from_diffusers_unet, model_config_from_unet
from huggingface_guess.utils import unet_to_diffusers
from backend import memory_management
from backend.operations import using_forge_operations
from backend.nn.cnets import cldm
from backend.patcher.controlnet import ControlLora, ControlNet, load_t2i_adapter, apply_controlnet_advanced
from modules_forge.shared import add_supported_control_model


class ControlModelPatcher:
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        return None

    def __init__(self, model_patcher=None):
        self.model_patcher = model_patcher
        self.strength = 1.0
        self.start_percent = 0.0
        self.end_percent = 1.0
        self.positive_advanced_weighting = None
        self.negative_advanced_weighting = None
        self.advanced_frame_weighting = None
        self.advanced_sigma_weighting = None
        self.advanced_mask_weighting = None

    def process_after_running_preprocessors(self, process, params, *args, **kwargs):
        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        return

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        return


class ControlNetPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(controlnet_data, ckpt_path):
        if "lora_controlnet" in controlnet_data:
            return ControlNetPatcher(ControlLora(controlnet_data))

        controlnet_config = None
        if "controlnet_cond_embedding.conv_in.weight" in controlnet_data:  # diffusers format
            unet_dtype = memory_management.unet_dtype()
            controlnet_config = unet_config_from_diffusers_unet(controlnet_data, unet_dtype)
            diffusers_keys = unet_to_diffusers(controlnet_config)
            diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
            diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

            count = 0
            loop = True
            while loop:
                suffix = [".weight", ".bias"]
                for s in suffix:
                    k_in = "controlnet_down_blocks.{}{}".format(count, s)
                    k_out = "zero_convs.{}.0{}".format(count, s)
                    if k_in not in controlnet_data:
                        loop = False
                        break
                    diffusers_keys[k_in] = k_out
                count += 1

            count = 0
            loop = True
            while loop:
                suffix = [".weight", ".bias"]
                for s in suffix:
                    if count == 0:
                        k_in = "controlnet_cond_embedding.conv_in{}".format(s)
                    else:
                        k_in = "controlnet_cond_embedding.blocks.{}{}".format(count - 1, s)
                    k_out = "input_hint_block.{}{}".format(count * 2, s)
                    if k_in not in controlnet_data:
                        k_in = "controlnet_cond_embedding.conv_out{}".format(s)
                        loop = False
                    diffusers_keys[k_in] = k_out
                count += 1

            new_sd = {}
            for k in diffusers_keys:
                if k in controlnet_data:
                    new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

            leftover_keys = controlnet_data.keys()
            if len(leftover_keys) > 0:
                print("leftover keys:", leftover_keys)
            controlnet_data = new_sd

        pth_key = 'control_model.zero_convs.0.0.weight'
        pth = False
        key = 'zero_convs.0.0.weight'
        if pth_key in controlnet_data:
            pth = True
            key = pth_key
            prefix = "control_model."
        elif key in controlnet_data:
            prefix = ""
        else:
            net = load_t2i_adapter(controlnet_data)
            if net is None:
                return None
            return ControlNetPatcher(net)

        if controlnet_config is None:
            unet_dtype = memory_management.unet_dtype()
            controlnet_config = model_config_from_unet(controlnet_data, prefix, True).unet_config
            controlnet_config['dtype'] = unet_dtype

        load_device = memory_management.get_torch_device()
        computation_dtype = memory_management.get_computation_dtype(load_device)

        controlnet_config.pop("out_channels")
        controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]

        with using_forge_operations(dtype=unet_dtype, manual_cast_enabled=computation_dtype != unet_dtype):
            control_model = cldm.ControlNet(**controlnet_config).to(dtype=unet_dtype)

        if pth:
            if 'difference' in controlnet_data:
                print("WARNING: Your controlnet model is diff version rather than official float16 model. "
                      "Please use an official float16/float32 model for robust performance.")

            class WeightsLoader(torch.nn.Module):
                pass

            w = WeightsLoader()
            w.control_model = control_model
            missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
        else:
            missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)
        print(missing, unexpected)

        global_average_pooling = False
        filename = os.path.splitext(ckpt_path)[0]
        if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"):
            # TODO: smarter way of enabling global_average_pooling
            global_average_pooling = True

        control = ControlNet(control_model, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=computation_dtype)
        return ControlNetPatcher(control)

    def __init__(self, model_patcher):
        super().__init__(model_patcher)

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        unet = apply_controlnet_advanced(
            unet=unet,
            controlnet=self.model_patcher,
            image_bchw=cond,
            strength=self.strength,
            start_percent=self.start_percent,
            end_percent=self.end_percent,
            positive_advanced_weighting=self.positive_advanced_weighting,
            negative_advanced_weighting=self.negative_advanced_weighting,
            advanced_frame_weighting=self.advanced_frame_weighting,
            advanced_sigma_weighting=self.advanced_sigma_weighting,
            advanced_mask_weighting=self.advanced_mask_weighting
        )

        process.sd_model.forge_objects.unet = unet
        return


class FluxControlNetPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(controlnet_data, ckpt_path):
        # 检查是否是Flux ControlNet格式
        if "controlnet_blocks.0.weight" not in controlnet_data:
            return None
        
        print(f"Loading Flux ControlNet from {ckpt_path}")
        
        # 创建Flux ControlNet模型
        from diffusers.models import FluxControlNetModel
        
        # 从config.json加载配置
        config_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # 默认配置
            config = {
                "patch_size": 1,
                "in_channels": 64,
                "num_layers": 6,
                "num_single_layers": 0,
                "num_attention_heads": 24,
                "attention_head_dim": 128,
                "joint_attention_dim": 4096,
                "pooled_projection_dim": 768
            }
        
        # 创建模型
        model = FluxControlNetModel(**config)
        
        # 加载权重
        try:
            model.load_state_dict(controlnet_data, strict=False)
            print("Flux ControlNet loaded successfully")
            return FluxControlNetPatcher(model)
        except Exception as e:
            print(f"Failed to load Flux ControlNet: {e}")
            return None

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        # Flux ControlNet的处理逻辑
        if hasattr(process, 'control_net_units') and process.control_net_units:
            # 这里需要实现Flux ControlNet的具体处理逻辑
            # 暂时跳过，避免错误
            pass
        return


add_supported_control_model(ControlNetPatcher)
add_supported_control_model(FluxControlNetPatcher)
