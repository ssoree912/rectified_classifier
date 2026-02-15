import torch
import torch.nn.functional as F
import numpy as np
import cv2
import sys
import os

# Real-ESRGAN import
sys.path.append('/Users/hwangsolhee/Desktop/mlpr/D3/Real-ESRGAN')
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


class BasicSRProcessor:
    """다운샘플링 + Real-ESRGAN SR 기본 처리기"""
    
    def __init__(self, scale=4, model_name='RealESRGAN_x4plus', device='cuda', tile=512):
        self.scale = scale
        self.device = device
        
        # Real-ESRGAN 모델 경로
        model_path = f'/Users/hwangsolhee/Desktop/mlpr/D3/Real-ESRGAN/weights/{model_name}.pth'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # RRDBNet 아키텍처 설정
        model = RRDBNet(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64, 
            num_block=23, 
            num_grow_ch=32, 
            scale=scale
        )
        
        # RealESRGANer 초기화
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=True,  # FP16 사용
            device=device
        )
        
        print(f"Real-ESRGAN initialized with {model_name}, scale={scale}")
    
    def downsample(self, img_tensor):
        """
        이미지 다운샘플링
        Args:
            img_tensor: torch.Tensor (B, C, H, W), 범위 [0, 1]
        Returns:
            downsampled: torch.Tensor (B, C, H/scale, W/scale)
        """
        return F.interpolate(
            img_tensor, 
            scale_factor=1/self.scale, 
            mode='bicubic', 
            align_corners=False
        )
    
    def tensor_to_numpy(self, tensor):
        """Tensor를 numpy로 변환 (Real-ESRGAN 입력용)"""
        # (B, C, H, W) -> (B, H, W, C), [0,1] -> [0,255]
        numpy_imgs = []
        for i in range(tensor.size(0)):
            img = tensor[i].permute(1, 2, 0).cpu().numpy()
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
            numpy_imgs.append(img)
        return numpy_imgs
    
    def numpy_to_tensor(self, numpy_imgs):
        """numpy를 Tensor로 변환"""
        # (H, W, C) -> (C, H, W), [0,255] -> [0,1]
        tensors = []
        for img in numpy_imgs:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensors.append(img_tensor)
        return torch.stack(tensors).to(self.device)
    
    def sr_process(self, img_tensor):
        """
        전체 SR 처리: 다운샘플링 -> Real-ESRGAN
        Args:
            img_tensor: torch.Tensor (B, C, H, W), 범위 [0, 1]
        Returns:
            sr_result: torch.Tensor (B, C, H, W), 범위 [0, 1]
        """
        # 1. 다운샘플링
        downsampled = self.downsample(img_tensor)
        
        # 2. Tensor -> numpy 변환
        numpy_imgs = self.tensor_to_numpy(downsampled)
        
        # 3. Real-ESRGAN SR 처리
        sr_numpy_imgs = []
        for img in numpy_imgs:
            sr_img, _ = self.upsampler.enhance(img, outscale=self.scale)
            sr_numpy_imgs.append(sr_img)
        
        # 4. numpy -> Tensor 변환
        sr_tensor = self.numpy_to_tensor(sr_numpy_imgs)
        
        return sr_tensor
    
    def process_batch(self, batch_tensor):
        """배치 단위 처리"""
        return self.sr_process(batch_tensor)


def get_sr_processor(scale=4, model_name='RealESRGAN_x4plus', device='cuda'):
    """SR Processor 팩토리 함수"""
    return BasicSRProcessor(
        scale=scale,
        model_name=model_name,
        device=device
    )


if __name__ == "__main__":
    # 테스트
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # SR Processor 초기화
        processor = get_sr_processor(device=device)
        
        # 더미 이미지로 테스트 (224x224)
        dummy_img = torch.randn(2, 3, 224, 224).to(device)
        print(f"Input shape: {dummy_img.shape}")
        
        # SR 처리
        sr_result = processor.sr_process(dummy_img)
        print(f"SR result shape: {sr_result.shape}")
        print("SR processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Real-ESRGAN weights are downloaded")