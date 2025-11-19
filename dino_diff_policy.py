import torch
from einops import rearrange
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

print()
dummy_tensor = torch.randn(1, 3, 224, 224)
image_h, image_w = dummy_tensor.shape[2:]
kernel_h, kernel_w = backbone.patch_embed.proj.kernel_size
patch_tokens = backbone.forward_features(dummy_tensor)["x_norm_patchtokens"] #.reshape(1, 224//14, 224//14, -1)

reshaped_pt = rearrange(patch_tokens, "b (h w) c -> b c h w", h=image_h//kernel_h, w=image_w//kernel_w)
print(reshaped_pt.shape)