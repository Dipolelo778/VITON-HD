# composite.py

import torch

def composite(warped_cloth, person_image, segmentation_mask, inpainted=None):
    """
    Args:
        warped_cloth: [B, 3, H, W] output from GMM
        person_image: [B, 3, H, W] original person photo
        segmentation_mask: [B, 1, H, W] binary mask for cloth region
        inpainted: [B, 3, H, W] optional, from inpainting network

    Returns:
        final_image: [B, 3, H, W]
    """

    # Expand mask to 3 channels
    mask = segmentation_mask.repeat(1, 3, 1, 1)

    # Compose
    clothed_region = warped_cloth * mask
    body_region = person_image * (1 - mask)

    if inpainted is not None:
        # Blend inpainted area under the cloth
        final_image = clothed_region + inpainted * (1 - mask)
    else:
        final_image = clothed_region + body_region

    return final_image


if __name__ == "__main__":
    warped = torch.randn(1, 3, 256, 192)
    person = torch.randn(1, 3, 256, 192)
    mask = torch.randint(0, 2, (1, 1, 256, 192)).float()
    inpainted = torch.randn(1, 3, 256, 192)

    result = composite(warped, person, mask, inpainted)
    print(result.shape)
