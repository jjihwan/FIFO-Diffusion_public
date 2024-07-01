import numpy as np
import cv2
from PIL import Image

def load_realesrgan_model(model_path):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=False
    )
    return upsampler

def upscale_image_with_realesrgan(image, upsampler):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 타일 크기 설정 (더 작은 크기로 설정하여 메모리 사용량 줄임)
    tile_size = 256
    
    height, width = img.shape[:2]
    sr_image = np.zeros((height * 4, width * 4, 3), dtype=np.uint8)  # 4배 업스케일링
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = img[y:y+tile_size, x:x+tile_size]
            sr_tile, _ = upsampler.enhance(tile, outscale=4)
            sr_image[y*4:y*4+sr_tile.shape[0], x*4:x*4+sr_tile.shape[1]] = sr_tile
    
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
    sr_image = Image.fromarray(sr_image)
    return sr_image
