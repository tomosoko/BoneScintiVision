"""
BoneScintiVision — ガンマカメラDICOM読み込みモジュール

骨シンチグラフィのDICOMファイルを読み込み、
YOLOモデル入力形式 (256×256 PNG) に変換する。

対応DICOM:
  - NM (Nuclear Medicine) modality
  - WB (Whole Body) scan
  - 前面・後面の両画像が含まれるDICOM

使い方:
  from synth.dicom_reader import BoneScintiDicom
  ds = BoneScintiDicom("/path/to/scan.dcm")
  img_ant, img_post = ds.get_views(size=256)
  img_dual = ds.get_dual_view(size=256)

注意:
  pydicom が必要: pip install pydicom
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional


def _check_pydicom():
    try:
        import pydicom
        return pydicom
    except ImportError:
        raise ImportError(
            "pydicom が必要です: pip install pydicom\n"
            "  または: pip install pydicom pillow"
        )


class BoneScintiDicom:
    """
    骨シンチグラフィDICOMファイルの読み込みと前処理。

    主なユースケース:
      1. gamma camera DICOM (NM modality) から前後面画像を抽出
      2. 256×256 または 512×256 (dual view) に標準化
      3. モデル推論のための前処理（ヒストグラム正規化）
    """

    def __init__(self, dcm_path: str):
        pydicom = _check_pydicom()
        self.dcm_path = Path(dcm_path)
        self.ds = pydicom.dcmread(str(dcm_path))
        self._pixel_array = None

    @property
    def modality(self) -> str:
        return getattr(self.ds, "Modality", "UNKNOWN")

    @property
    def pixel_array(self) -> np.ndarray:
        if self._pixel_array is None:
            self._pixel_array = self.ds.pixel_array.astype(np.float32)
        return self._pixel_array

    @property
    def n_frames(self) -> int:
        arr = self.pixel_array
        if arr.ndim == 3:
            return arr.shape[0]
        return 1

    def get_frame(self, frame_idx: int = 0) -> np.ndarray:
        """指定フレームを float32 [0,1] で返す"""
        arr = self.pixel_array
        if arr.ndim == 3:
            frame = arr[frame_idx]
        else:
            frame = arr

        max_val = frame.max()
        if max_val > 0:
            frame = frame / max_val
        return frame

    def get_anterior_frame(self) -> Optional[np.ndarray]:
        """
        前面（anterior）フレームを返す。
        NMシンチグラフィでは通常フレーム0が前面。
        """
        if self.n_frames >= 1:
            return self.get_frame(0)
        return None

    def get_posterior_frame(self) -> Optional[np.ndarray]:
        """
        後面（posterior）フレームを返す。
        NMシンチグラフィでは通常フレーム1が後面。
        """
        if self.n_frames >= 2:
            return self.get_frame(1)
        return None

    def _normalize_and_resize(
        self, frame: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """フレームを uint8 [0,255] に変換して指定サイズにリサイズ（パディング付き）"""
        # CLAHE で骨シンチ特有のダイナミックレンジを最適化
        frame_u8 = (frame * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        frame_u8 = clahe.apply(frame_u8)

        # アスペクト比維持リサイズ + パディング
        h, w = frame_u8.shape[:2]
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        img_r = cv2.resize(frame_u8, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        px = (target_w - nw) // 2
        py = (target_h - nh) // 2
        canvas[py:py + nh, px:px + nw] = img_r
        return canvas

    def get_views(self, size: int = 256) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        前面・後面を各 size×size に標準化して返す。

        Returns:
            (anterior_img, posterior_img)  uint8 グレースケール
            posterior は None の場合あり（シングルビューDICOM）
        """
        ant_frame = self.get_anterior_frame()
        post_frame = self.get_posterior_frame()

        ant_img = self._normalize_and_resize(ant_frame, size, size) if ant_frame is not None else None
        post_img = self._normalize_and_resize(post_frame, size, size) if post_frame is not None else None

        return ant_img, post_img

    def get_dual_view(self, size: int = 256) -> np.ndarray:
        """
        前面 + 後面 を横並び (size*2 × size) にして返す。
        EXP-002 デュアルビューモデルへの入力に使用。

        後面がない場合は前面を両側に複製する。
        """
        ant_img, post_img = self.get_views(size)
        if ant_img is None:
            raise ValueError("前面（anterior）フレームが見つかりません")

        if post_img is None:
            post_img = ant_img.copy()

        dual = np.hstack([ant_img, post_img])
        return cv2.cvtColor(dual, cv2.COLOR_GRAY2RGB)

    def get_single_view_rgb(self, size: int = 256) -> np.ndarray:
        """
        前面のみを size×size RGB で返す。
        EXP-001 単一ビューモデルへの入力に使用。
        """
        ant_img, _ = self.get_views(size)
        if ant_img is None:
            raise ValueError("前面（anterior）フレームが見つかりません")
        return cv2.cvtColor(ant_img, cv2.COLOR_GRAY2RGB)

    def __repr__(self):
        return (
            f"BoneScintiDicom({self.dcm_path.name}, "
            f"modality={self.modality}, frames={self.n_frames})"
        )


def load_dicom_for_inference(
    dcm_path: str,
    dual_view: bool = False,
    size: int = 256,
) -> np.ndarray:
    """
    DICOMファイルをモデル推論用のndarrayに変換するヘルパー。

    Args:
        dcm_path:  DICOMファイルパス
        dual_view: True=デュアルビュー(512×256), False=単一ビュー(256×256)
        size:      出力高さ（幅は dual_view 時に 2×size）

    Returns:
        RGB ndarray, shape (size, size, 3) or (size, size*2, 3)
    """
    ds = BoneScintiDicom(dcm_path)
    if dual_view:
        return ds.get_dual_view(size)
    else:
        return ds.get_single_view_rgb(size)
