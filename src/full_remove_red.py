#!/usr/bin/env python3
"""PDF -> [B0 remove_red] -> Surya layout -> PP-OCR det -> cut/sort/crop -> VietOCR -> rule extraction (STANDALONE).

Ban nay = 8_merge.py + B0 tien xu ly xoa dau do (remove_red.py V5, vendored verbatim
trong section B0, doi ten process_pdf -> rr_process_pdf de tranh nham lan).
Hai che do B0 (chon qua --rr-mode):
  - pdf (mac dinh): rewrite XObject anh RGB trong PDF thanh ban sach do, luu vao
    <run_dir>/<stem>_cleaned.pdf, pipeline render tu ban sach nay. File goc giu nguyen.
    Dung flate (lossless, mac dinh) hoac jpeg (--rr-encode jpeg, nho hon).
  - memory: khong ghi PDF moi; loc truc tiep tren anh render (RGB->BGR->loc->RGB)
    bang remove_red_from_bgr truoc khi dua vao layout/det.
Tat --no-remove-red de chay pipeline goc (tuong duong 8_merge.py).

Ban doc lap: B1-B4 giu nguyen logic mo hinh; B5 rule VBHC 2.1-pruned duoc dong bang
verbatim trong file (section RULE VBHC), KHONG exec notebook, KHONG doc json ben ngoai,
KHONG import rule_evaluation. Muon doi rule: sua truc tiep section rule + bump version.

Toi uu chay batch / song song tren GPU:

- B1 Surya layout: chunk nhieu trang theo --layout-batch-size trong 1 lan goi
  predictor (notebook goi 1 lan cho toan bo trang); render trang song song CPU.
  --layout-device cpu (mac dinh, giong notebook) hoac cuda de tang toc.
- B2 PP-OCR det: gop nhieu trang thanh 1 batch ORT duy nhat theo
  --det-batch-size (pad ve max rh/rw trong batch, cat prob ve kich thuoc that
  truoc khi postprocess — giong cach src/pipeline.py lam); preprocess va
  postprocess OpenCV chay song song bang ThreadPoolExecutor. Notebook chay
  batch=1 tung trang.
- B1/B2 co the chay song song tren 2 worker voi --parallel-layout-det
  (ca hai chi doc page_images da render; an toan nhat khi layout=cpu,
  det=CUDA vi khac device).
- B3 gan layout + sort + crop: song song theo trang (ThreadPoolExecutor),
  moi trang doc lap hoan toan.
- B4 VietOCR: gom crop cua TAT CA trang thanh 1 hang doi global theo
  (page, reading_order) roi suy luan theo --ocr-batch-size (notebook reset
  batch theo tung trang voi batch=32); preload anh song song, torch.inference_mode,
  tuy chon --ocr-amp (autocast fp16).
- Batch cap PDF: --pdf 1 file hoac --input-dir quet de quy *.pdf; model load
  1 lan cho ca batch; timing + timing_summary giong 8_1_remove_red.py.

Vi du:
    python3 notebooks_v1/8_merge.py --pdf demo_benchmark_data/data_1/a.pdf
    python3 notebooks_v1/8_merge.py --pdf demo_benchmark_data/data_1/a.pdf --pages 0
    python3 notebooks_v1/8_merge.py --input-dir demo_benchmark_data/data_1
    python3 notebooks_v1/8_merge_remove_red.py --pdf demo_benchmark_data/data_1/a.pdf
    python3 notebooks_v1/8_merge_remove_red.py --pdf demo_benchmark_data/data_1/a.pdf --rr-mode memory
    python3 notebooks_v1/8_merge_remove_red.py --pdf demo_benchmark_data/data_1/a.pdf --no-remove-red\n    python3 notebooks_v1/8_merge.py --input-dir demo_benchmark_data/data_1 \
        --layout-device cuda --det-batch-size 4 --ocr-batch-size 64 --parallel-layout-det
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterable, TypedDict

import cv2
import fitz
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Tham so mac dinh (khop notebook 8_merge.ipynb)
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_ROOT = "outputs/merge_remove_red"
DEFAULT_DPI = 150
DEFAULT_LAYOUT_THRESHOLD = 0.4
DEFAULT_LAYOUT_DEVICE = "cpu"   # giong notebook (FAST_DETECTOR_DEVICE=cpu)
DEFAULT_LAYOUT_BATCH = 8        # so trang / 1 lan goi predictor
DEFAULT_DET_MAX_SIDE = 1536
DEFAULT_DET_BATCH = 4           # so trang / 1 lan ORT session.run
DEFAULT_ROW_TOL = 0.5
DEFAULT_CROP_PADDING = 2
DEFAULT_OCR_BATCH = 64          # notebook=32 (reset theo trang); script gom global
DEFAULT_OCR_WORKERS = 4
DEFAULT_PAGE_BATCH = 0          # 0 = toan bo trang cung luc (giong notebook)
DEFAULT_RULE_VERSION = "2.1-pruned"
# B0 remove_red (V5) — mac dinh khop remove_red.py
DEFAULT_RR_MODE = "pdf"  # pdf | memory ( + --no-remove-red de tat han)
DEFAULT_RR_SUFFIX = "_cleaned"
DEFAULT_RR_ENCODE = "flate"
DEFAULT_RR_JPEG_QUALITY = 90
DEFAULT_RR_DENSE_THRESHOLD = 0.18
DEFAULT_RR_DARK_RED_MAX = 145
DEFAULT_RR_MIN_BLACK_NEIGHBORS = 8
DEFAULT_RR_WORKERS = 4


# ---------------------------------------------------------------------------
# Helpers dung chung (nguyen van tu notebook)
# ---------------------------------------------------------------------------
def render_page(page: fitz.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, alpha=False, colorspace=fitz.csRGB)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n).copy()


def order_quad(points) -> np.ndarray:
    ordered = np.zeros((4, 2), np.float32)
    points = np.asarray(points, np.float32)
    ordered[0], ordered[2] = points[np.argmin(points.sum(1))], points[np.argmax(points.sum(1))]
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[1], ordered[3] = points[np.argmin(diffs)], points[np.argmax(diffs)]
    return ordered


def parse_pages(spec: str | None, page_count: int) -> list[int]:
    if not spec:
        return list(range(page_count))
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        pages.append(int(part))
    if bad := [p for p in pages if not 0 <= p < page_count]:
        raise IndexError(f"Trang ngoai pham vi: {bad}; PDF co {page_count} trang")
    return sorted(set(pages))


def bbox_overlap(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def polygon_overlap(a, b):
    try:
        area, _ = cv2.intersectConvexConvex(np.asarray(a, np.float32), np.asarray(b, np.float32))
        return float(max(0., area))
    except cv2.error:
        return 0.


def bbox_distance(a, b):
    return float(np.hypot(max(b[0] - a[2], a[0] - b[2], 0.), max(b[1] - a[3], a[1] - b[3], 0.)))


def assign_layout(line, layouts):
    if not layouts:
        return None, "no_layout_fallback", 0.
    scored = [((polygon_overlap(line["polygon_points"], layout["polygon_points"])
                if bbox_overlap(line["bbox"], layout["bbox"]) > 0 else 0.), layout) for layout in layouts]
    overlap, layout = max(scored, key=lambda item: (item[0], -item[1]["order"]))
    if overlap > 0:
        return layout, "max_intersection", overlap
    layout = min(layouts, key=lambda item: (bbox_distance(line["bbox"], item["bbox"]), item["order"]))
    return layout, "nearest_layout", bbox_distance(line["bbox"], layout["bbox"])


def sort_inside_layout(lines, row_tol_ratio):
    if not lines:
        return []
    tol = max(1., float(np.median([max(1., line["bbox"][3] - line["bbox"][1]) for line in lines]))) * row_tol_ratio
    rows = []
    for line in sorted(lines, key=lambda item: ((item["bbox"][1] + item["bbox"][3]) / 2, item["bbox"][0])):
        cy = (line["bbox"][1] + line["bbox"][3]) / 2
        hit = [row for row in rows if abs(cy - row["cy"]) <= tol]
        if hit:
            row = min(hit, key=lambda item: abs(cy - item["cy"]))
            row["lines"].append(line)
            row["cy"] = float(np.mean([(item["bbox"][1] + item["bbox"][3]) / 2 for item in row["lines"]]))
        else:
            rows.append({"cy": cy, "lines": [line]})
    ordered = []
    for row_index, row in enumerate(sorted(rows, key=lambda item: item["cy"])):
        for line in sorted(row["lines"], key=lambda item: (item["bbox"][0], item["bbox"][1])):
            line["row_index"] = row_index
            ordered.append(line)
    return ordered


def crop_line_polygon(image, quad, padding=0):
    pts = order_quad(np.asarray(quad, np.float32))
    w = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
    h = max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))
    ow, oh = max(1, int(round(w))), max(1, int(round(h)))
    dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32)
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(pts, dst), (ow, oh),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return cv2.copyMakeBorder(crop, padding, padding, padding, padding, cv2.BORDER_REPLICATE) if padding > 0 else crop




# ---------------------------------------------------------------------------
# B0: TIEN XU LY XOA DAU DO (remove_red.py V5, vendored — chi doi ten PDF-API)
# ---------------------------------------------------------------------------
# Nguon: remove_red.py (SUFFIX="_processed_v5"). Toan bo logic loc mau giu nguyen
# verbatim; chi doi ten de tranh nham voi pipeline: process_pdf -> rr_process_pdf,
# _read/_write/_encode/_save -> _rr_*, SUFFIX -> RR_SUFFIX2 (hang so noi bo B0).
# Tham so dong bo voi DEFAULT_RR_* o tren.
RR_SUFFIX2 = "_cleaned"
RR_ENCODERS = ("flate", "jpeg")


def most_frequent_bgr_color(img, alpha=None):
    pixels = img.reshape(-1, 3) if alpha is None else img[alpha > 0]
    if not pixels.size:
        return None
    packed = ((pixels[:, 0].astype(np.uint32) << 16)
              | (pixels[:, 1].astype(np.uint32) << 8)
              | pixels[:, 2].astype(np.uint32))
    colors, counts = np.unique(packed, return_counts=True)
    color = int(colors[np.argmax(counts)])
    return np.array([(color >> 16) & 255, (color >> 8) & 255, color & 255],
                    dtype=np.uint8)


class _Analysis:
    __slots__ = ("visible", "gray", "red_channel", "diff_rg", "diff_rb",
                 "final_red_mask", "isolated_dark_red", "stroke_with_red",
                 "black_anchor")

    def __init__(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


def analyse(img, alpha=None):
    visible = (np.ones(img.shape[:2], dtype=bool) if alpha is None
               else alpha > 0)
    channels = img.astype(np.int16)
    blue, green, red = channels[..., 0], channels[..., 1], channels[..., 2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    diff_rg = red - green
    diff_rb = red - blue

    black_anchor = ((np.abs(diff_rg) < 25) & (np.abs(diff_rb) < 25)
                    & (gray < 130) & visible)
    neighborhood = cv2.dilate(black_anchor.astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0
    overlap = (red < 115) & (np.minimum(green, blue) < 75) & (gray < 90)
    blue_ink = (blue > red) & (blue > green)
    protected = (blue_ink | black_anchor | (neighborhood & (gray < 160)
                                 & ((diff_rg < 35) | overlap))) & visible

    hue = (hsv[..., 0] <= 15) | (hsv[..., 0] >= 145)
    saturation = hsv[..., 1]
    red_mask = ((hue & (saturation >= 25) & (diff_rg >= 18) & (diff_rb >= 12) & (red > 80))
                | ((lab[..., 1] >= 134) & (diff_rg >= 15) & (red > 80))
                | (hue & (saturation >= 15) & (diff_rg >= 10) & (red > 120)))
    red_mask &= ~protected & visible

    removal = (cv2.dilate(red_mask.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0)
    removal &= ~protected & visible

    return _Analysis(
        visible=visible, gray=gray, red_channel=red, diff_rg=diff_rg,
        diff_rb=diff_rb,
        final_red_mask=removal,
        isolated_dark_red=(overlap & (diff_rg > 15) & (diff_rb > 12)
                           & ~neighborhood & visible),
        stroke_with_red=(protected & (diff_rg > 15) & (diff_rb > 12)
                         & (gray < 110)),
        black_anchor=black_anchor,
    )


def clean_red(img, alpha=None, *, analysis=None):
    if analysis is None:
        analysis = analyse(img, alpha)
    result = img.copy()
    result[analysis.final_red_mask] = 255
    if analysis.isolated_dark_red.any():
        background = most_frequent_bgr_color(img, alpha)
        if background is not None:
            result[analysis.isolated_dark_red] = background
    if analysis.stroke_with_red.any():
        result[analysis.stroke_with_red] = analysis.gray[analysis.stroke_with_red, None]
    return result


def recovery_boxes(regions, shape):
    height, width = shape[:2]
    boxes = []
    for x, y, box_width, box_height, _ in regions:
        margin = max(10, round(0.25 * max(box_width, box_height)))
        boxes.append((max(0, y - margin), min(height, y + box_height + margin),
                      max(0, x - margin), min(width, x + box_width + margin)))
    return boxes


def apply_recovery(img, alpha, result, analysis, *, dark_red_max,
                   min_black_neighbors, boxes):
    if not boxes:
        return result
    height, width = img.shape[:2]
    top = max(0, min(box[0] for box in boxes) - 9)
    bottom = min(height, max(box[1] for box in boxes) + 9)
    left = max(0, min(box[2] for box in boxes) - 9)
    right = min(width, max(box[3] for box in boxes) + 9)
    if top >= bottom or left >= right:
        return result

    anchors = analysis.black_anchor[top:bottom, left:right].astype(np.uint8)
    if not anchors.any():
        return result
    count = cv2.boxFilter(anchors, cv2.CV_32S, (19, 19), normalize=False)
    region_red = analysis.red_channel[top:bottom, left:right]
    candidate = ((analysis.diff_rg[top:bottom, left:right] > 15)
                 & (analysis.diff_rb[top:bottom, left:right] > 12)
                 & (region_red > 45) & (region_red < dark_red_max)
                 & (count >= min_black_neighbors)
                 & analysis.visible[top:bottom, left:right])
    if not candidate.any():
        return result

    targets = np.zeros(candidate.shape, dtype=bool)
    for box_top, box_bottom, box_left, box_right in boxes:
        targets[max(0, box_top - top):box_bottom - top,
                max(0, box_left - left):box_right - left] = True
    candidate &= targets
    if not candidate.any():
        return result

    recovered = np.minimum(np.rint(region_red * 0.55), 100).astype(np.uint8)
    result[top:bottom, left:right][candidate] = recovered[candidate, None]
    return result


def find_red_regions(img, alpha=None):
    channels = img.astype(np.int16)
    blue, green, red = channels[..., 0], channels[..., 1], channels[..., 2]
    strong_red = (red - green > 50) & (red - blue > 50) & (red > 100)
    if alpha is not None:
        strong_red &= alpha > 0
    if not strong_red.any():
        return []

    grouped = cv2.dilate(strong_red.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    min_box_area = max(2500, round(img.shape[0] * img.shape[1] * 0.002))
    candidates = []
    for index in range(1, count):
        x, y, box_width, box_height, _ = (int(value) for value in stats[index])
        box_area = box_width * box_height
        if box_area < min_box_area:
            continue
        red_count = int(strong_red[y:y + box_height, x:x + box_width].sum())
        if red_count < 400:
            continue
        candidates.append((x, y, box_width, box_height, red_count / box_area))

    candidates.sort(key=lambda region: region[2] * region[3], reverse=True)
    regions = []
    for candidate in candidates:
        x, y, box_width, box_height, _ = candidate
        contained = any((x >= ax and y >= ay and x + box_width <= ax + aw
                         and y + box_height <= ay + ah)
                        for ax, ay, aw, ah, _ in regions)
        if not contained:
            regions.append(candidate)
    return regions


def remove_red_from_bgr(img, alpha=None, *, dense_threshold=DEFAULT_RR_DENSE_THRESHOLD,
                        dark_red_max=DEFAULT_RR_DARK_RED_MAX,
                        min_black_neighbors=DEFAULT_RR_MIN_BLACK_NEIGHBORS):
    if not 0 <= dense_threshold <= 1:
        raise ValueError("dense_threshold phải trong khoảng 0–1.")
    if not 1 <= dark_red_max <= 255:
        raise ValueError("dark_red_max phải trong khoảng 1–255.")
    if not 0 <= min_black_neighbors <= 361:
        raise ValueError("min_black_neighbors phải trong khoảng 0–361.")
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError("Ảnh đầu vào phải là BGR uint8 với 3 kênh.")
    if alpha is not None and alpha.shape != img.shape[:2]:
        raise ValueError("Kích thước alpha không khớp ảnh đầu vào.")

    analysis = analyse(img, alpha)
    result = clean_red(img, alpha, analysis=analysis)
    dense = [region for region in find_red_regions(img, alpha)
             if region[4] >= dense_threshold]
    if not dense:
        return result
    return apply_recovery(img, alpha, result, analysis,
                          dark_red_max=dark_red_max,
                          min_black_neighbors=min_black_neighbors,
                          boxes=recovery_boxes(dense, img.shape))


def _rr_read_alpha_and_masks(doc, xref, info, image_shape):
    entries = []
    for key in ("SMask", "Mask"):
        kind, value = doc.xref_get_key(xref, key)
        if kind != "null":
            entries.append(f"/{key} {value}")

    alpha = None
    if info.get("smask"):
        try:
            mask_info = doc.extract_image(info["smask"])
            alpha = cv2.imdecode(np.frombuffer(mask_info["image"], np.uint8),
                                 cv2.IMREAD_GRAYSCALE)
        except Exception:
            alpha = None
        if alpha is not None and alpha.shape != image_shape:
            alpha = cv2.resize(alpha, (image_shape[1], image_shape[0]),
                               interpolation=cv2.INTER_AREA)
    return alpha, entries


def _rr_write_image(doc, xref, samples, *, width, height, codec, mask_entries):
    filter_name = "/DCTDecode" if codec == "jpeg" else "/FlateDecode"
    entries = ["/Type /XObject", "/Subtype /Image",
               f"/Width {width}", f"/Height {height}",
               "/BitsPerComponent 8", "/ColorSpace /DeviceRGB",
               f"/Filter {filter_name}"]
    entries.extend(mask_entries)
    doc.update_object(xref, "<< " + " ".join(entries) + " >>")
    doc.update_stream(xref, samples, compress=(1 if codec == "flate" else 0))
    doc.xref_set_key(xref, "Filter", filter_name)


def _rr_encode(cleaned, codec, jpeg_quality):
    if codec == "jpeg":
        ok, buffer = cv2.imencode(".jpg", cleaned,
                                  [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise ValueError("Không mã hóa được ảnh JPEG.")
        return buffer.tobytes()
    return np.ascontiguousarray(cleaned[:, :, ::-1]).tobytes()


def _rr_save_new_pdf(document, output_path):
    identity = None
    try:
        with output_path.open("xb") as stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            document.save(stream, garbage=4, deflate=True)
    except BaseException:
        if identity is not None:
            try:
                stat = output_path.lstat()
                if (stat.st_dev, stat.st_ino) == identity:
                    output_path.unlink()
            except FileNotFoundError:
                pass
        raise


def rr_process_pdf(input_pdf, output_pdf=None, *, encode=DEFAULT_RR_ENCODE,
                   jpeg_quality=DEFAULT_RR_JPEG_QUALITY,
                   dense_threshold=DEFAULT_RR_DENSE_THRESHOLD,
                   dark_red_max=DEFAULT_RR_DARK_RED_MAX,
                   min_black_neighbors=DEFAULT_RR_MIN_BLACK_NEIGHBORS,
                   suffix=DEFAULT_RR_SUFFIX, verbose=True):
    """Ban sao PDF da xoa dau do (doi ten tu process_pdf cua remove_red.py)."""
    if encode not in RR_ENCODERS:
        raise ValueError(f"encode phải là một trong {RR_ENCODERS}.")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality phải trong khoảng 1–100.")
    if not 0 <= dense_threshold <= 1:
        raise ValueError("dense_threshold phải trong khoảng 0–1.")

    source_path = Path(input_pdf)
    if output_pdf is None:
        output_path = source_path.with_name(source_path.stem + suffix + ".pdf")
    else:
        output_path = Path(output_pdf)
    if not source_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {source_path}")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("File đầu ra phải khác file đầu vào.")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"File đầu ra đã tồn tại, không ghi đè: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục đầu ra: {output_path.parent}")

    changed = skipped = unreadable = 0
    with fitz.open(source_path) as doc:
        if not doc.is_pdf or doc.needs_pass or not doc.page_count:
            raise ValueError("Cần PDF có trang và không yêu cầu mật khẩu.")
        pages = doc.page_count
        for xref in range(1, doc.xref_length()):
            try:
                if not doc.xref_is_image(xref):
                    continue
                info = doc.extract_image(xref)
            except Exception as exc:
                unreadable += 1
                if verbose:
                    print(f"  Bỏ qua xref {xref}, không đọc được object: {exc}")
                continue
            if info.get("colorspace") != 3:
                continue
            img = cv2.imdecode(np.frombuffer(info["image"], np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                unreadable += 1
                if verbose:
                    print(f"  Bỏ qua ảnh xref {xref}, không giải mã được.")
                continue
            shape = img.shape[:2]
            alpha, mask_entries = _rr_read_alpha_and_masks(doc, xref, info, shape)
            cleaned = remove_red_from_bgr(
                img, alpha, dense_threshold=dense_threshold,
                dark_red_max=dark_red_max,
                min_black_neighbors=min_black_neighbors)
            if np.array_equal(cleaned, img):
                skipped += 1
                continue
            try:
                samples = _rr_encode(cleaned, encode, jpeg_quality)
                _rr_write_image(doc, xref, samples, width=info["width"],
                                height=info["height"], codec=encode,
                                mask_entries=mask_entries)
            except Exception as exc:
                unreadable += 1
                if verbose:
                    print(f"  Bỏ qua xref {xref}, không ghi được ảnh: {exc}")
                continue
            changed += 1
            if verbose:
                print(f"  Đã làm sạch ảnh xref {xref}"
                      f" ({info['width']}x{info['height']},"
                      f" giữ trong suốt: {alpha is not None})")
        _rr_save_new_pdf(doc, output_path)

    if verbose:
        print(f"Đã lưu bản sao: {output_path}")
    return dict(pages=pages, changed=changed, skipped=skipped,
                unreadable=unreadable, output=os.path.getsize(output_path))


def rr_clean_rgb_image(rgb: np.ndarray, *, dense_threshold=DEFAULT_RR_DENSE_THRESHOLD,
                       dark_red_max=DEFAULT_RR_DARK_RED_MAX,
                       min_black_neighbors=DEFAULT_RR_MIN_BLACK_NEIGHBORS) -> np.ndarray:
    """Loc do truc tiep tren anh render RGB (che do memory). Tra ve RGB uint8."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cleaned_bgr = remove_red_from_bgr(
        np.ascontiguousarray(bgr), None, dense_threshold=dense_threshold,
        dark_red_max=dark_red_max, min_black_neighbors=min_black_neighbors)
    return cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)


def rr_options_from_args(args) -> dict:
    return dict(encode=args.rr_encode, jpeg_quality=args.rr_jpeg_quality,
                dense_threshold=args.rr_dense_threshold,
                dark_red_max=args.rr_dark_red_max,
                min_black_neighbors=args.rr_min_black_neighbors)


# ---------------------------------------------------------------------------
# Do thoi gian (step / trang / tong), luu vao outputs/merge
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sync_cuda() -> None:
    """Dong bo CUDA (best-effort) de perf_counter phan anh dung GPU async."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _round3(value: float) -> float:
    return round(float(value), 3)


def _rss_mb() -> float | None:
    """RSS hien tai cua tien trinh (MB); thu psutil, roi /proc, roi resource."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        pass
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        import os
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 1e6, 1)
    except Exception:
        return None


def _peak_rss_mb() -> float | None:
    """Peak RSS (MB); Linux ru_maxrss tinh theo KB, macOS theo bytes."""
    try:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round((peak / 1e6 if sys.platform == "darwin" else peak / 1e3), 1)
    except Exception:
        return None


def _gpu_mem_mb() -> dict:
    """VRAM torch (MB); None khi khong co CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"alloc_mb": None, "reserved_mb": None, "peak_alloc_mb": None}
        return {"alloc_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
                "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)}
    except Exception:
        return {"alloc_mb": None, "reserved_mb": None, "peak_alloc_mb": None}


def _gpu_reset_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _res_snapshot() -> dict:
    snap = {"rss_mb": _rss_mb()}
    snap.update(_gpu_mem_mb())
    return snap


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _config_dict(args) -> dict:
    keys = ("output_root", "pages", "dpi", "layout_threshold", "layout_device",
            "layout_batch_size", "det_batch_size", "det_max_side", "page_batch_size",
            "parallel_layout_det", "row_tol", "crop_padding", "ocr_batch_size",
            "ocr_workers", "ocr_amp", "rule_version",
            "no_remove_red", "rr_mode", "rr_encode", "rr_jpeg_quality",
            "rr_dense_threshold", "rr_dark_red_max", "rr_min_black_neighbors",
            "rr_workers")
    return {key: getattr(args, key) for key in keys}


def _collect_env(models, args) -> dict:
    """Cau hinh moi truong + model de kem theo timing (thu 1 lan / batch)."""
    import platform
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda_ok else []
        torch_v, cuda_v = torch.__version__, torch.version.cuda
    except Exception:
        cuda_ok, cuda_devices, torch_v, cuda_v = False, [], _pkg_version("torch"), None
    ram_gb = None
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        try:
            for line in open("/proc/meminfo", encoding="utf-8"):
                if line.startswith("MemTotal:"):
                    ram_gb = round(int(line.split()[1]) / 1e6, 1)
                    break
        except Exception:
            pass
    smi = None
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            smi = [line.strip() for line in out.stdout.strip().splitlines()]
    except Exception:
        pass
    det_providers: list = []
    try:
        det_providers = list(models.det_session.get_providers())
    except Exception:
        pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(), "ram_total_gb": ram_gb,
            "torch": torch_v, "torch_cuda": cuda_v, "cuda_available": cuda_ok,
            "cuda_devices": cuda_devices, "nvidia_smi": smi,
            "onnx_providers": det_providers, "det_provider": det_providers[0] if det_providers else None,
            "vietocr_device": getattr(models, "device", None),
            "packages": {"surya-ocr": _pkg_version("surya-ocr"), "onnxruntime-gpu": _pkg_version("onnxruntime-gpu"),
                         "onnxruntime": _pkg_version("onnxruntime"), "PyMuPDF": _pkg_version("PyMuPDF"),
                         "opencv-python-headless": _pkg_version("opencv-python-headless"),
                         "pillow": _pkg_version("pillow"), "numpy": _pkg_version("numpy")},
            "models": {"surya_dir": str(args.surya_dir), "ppocr_dir": str(args.ppocr_dir),
                       "vietocr_root": str(args.vietocr_root)}}


# ---------------------------------------------------------------------------
# RULE VBHC 2.1-pruned (FROZEN, STANDALONE)
# ---------------------------------------------------------------------------
# Logic trich xuat rule duoc dong bang verbatim tu 10 cell code cua
# notebooks_v1/6_1_optimize_rule.ipynb tai thoi diem generate:
#   cells (3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
# File nay KHONG doc .ipynb, KHONG doc json ben ngoai, KHONG import
# rule_evaluation. Moi thay doi rule phai sua truc tiep tai day va bump
# RULE_VERSION_FROZEN + RULE_CODE_SHA256.
# ---------------------------------------------------------------------------
RULE_VERSION = "2.1-pruned"
RULE_VERSION_FROZEN = RULE_VERSION
RULE_CODE_SHA256 = "4b3e0d2e9a4c88ea1b65e88821460954e9d56230f1d6b770672d844af2a34870"
RULE_CODE_CELLS = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21)

RULE_LEXICON = json.loads('{\n  "anchors": {\n    "attachment": [\n      "Ban hành kèm theo",\n      "Kèm theo"\n    ],\n    "code": [\n      "Số:"\n    ],\n    "first_recipients": [\n      "Kính gửi:"\n    ],\n    "receiverDate": [\n      "ĐẾN",\n      "CÔNG VĂN ĐẾN",\n      "VĂN BẢN ĐẾN"\n    ],\n    "recipients": [\n      "Nơi nhận:"\n    ]\n  },\n  "authority_prefixes": [\n    "TM.",\n    "KT.",\n    "TL.",\n    "TUQ.",\n    "Q."\n  ],\n  "authority_titles": [\n    "CHỦ TỊCH",\n    "PHÓ CHỦ TỊCH",\n    "GIÁM ĐỐC",\n    "PHÓ GIÁM ĐỐC",\n    "BỘ TRƯỞNG",\n    "THỨ TRƯỞNG",\n    "THỦ TƯỚNG",\n    "PHÓ THỦ TƯỚNG",\n    "CỤC TRƯỞNG",\n    "PHÓ CỤC TRƯỞNG",\n    "CHÁNH VĂN PHÒNG",\n    "PHÓ CHÁNH VĂN PHÒNG",\n    "CHỦ NHIỆM",\n    "PHÓ CHỦ NHIỆM",\n    "TỔNG GIÁM ĐỐC",\n    "NGƯỜI ĐẠI DIỆN",\n    "NGƯỜI ỦY QUYỀN",\n    "NGƯỜI THỰC HIỆN CÔNG BỐ THÔNG TIN",\n    "THƯ KÝ",\n    "CHỦ TỌA"\n  ],\n  "document_types": [\n    {\n      "abbreviation": "NQ",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "NGHỊ QUYẾT",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "QĐ",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "QUYẾT ĐỊNH",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "CT",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "CHỈ THỊ",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "QC",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "QUY CHẾ",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "QyĐ",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "QUY ĐỊNH",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "TC",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "THÔNG CÁO",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "TB",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "THÔNG BÁO",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "HD",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "HƯỚNG DẪN",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "CTr",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "CHƯƠNG TRÌNH",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "KH",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "KẾ HOẠCH",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "PA",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "PHƯƠNG ÁN",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "ĐA",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "ĐỀ ÁN",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "DA",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "DỰ ÁN",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "BC",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "BÁO CÁO",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "BB",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "BIÊN BẢN",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "TTr",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "TỜ TRÌNH",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "HĐ",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "HỢP ĐỒNG",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": null,\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "CÔNG VĂN",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "CĐ",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "CÔNG ĐIỆN",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "BGN",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "BẢN GHI NHỚ",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "BTT",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "BẢN THỎA THUẬN",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "GUQ",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "GIẤY ỦY QUYỀN",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "GM",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "GIẤY MỜI",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "GGT",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "GIẤY GIỚI THIỆU",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "GNP",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "GIẤY NGHỈ PHÉP",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "PG",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "PHIẾU GỬI",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "PC",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "PHIẾU CHUYỂN",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": "PB",\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "PHIẾU BÁO",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": null,\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "THƯ CÔNG",\n      "family": "administrative",\n      "source": "Nghị định 30/2020/NĐ-CP, Điều 7 / Phụ lục III"\n    },\n    {\n      "abbreviation": null,\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "QUYẾT ĐỊNH LIÊN TỊCH",\n      "family": "extended",\n      "source": "Existing pipeline scope / explicitly named instrument"\n    },\n    {\n      "abbreviation": null,\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "CÔNG BỐ THÔNG TIN BẤT THƯỜNG",\n      "family": "extended",\n      "source": "Existing pipeline scope / explicitly named instrument"\n    },\n    {\n      "abbreviation": null,\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "VĂN BẢN HỢP NHẤT",\n      "family": "extended",\n      "source": "Existing pipeline scope / explicitly named instrument"\n    },\n    {\n      "abbreviation": null,\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "NGHỊ QUYẾT LIÊN TỊCH",\n      "family": "extended",\n      "source": "Existing pipeline scope / explicitly named instrument"\n    },\n    {\n      "abbreviation": null,\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "NGHỊ ĐỊNH",\n      "family": "extended",\n      "source": "Existing pipeline scope / explicitly named instrument"\n    },\n    {\n      "abbreviation": null,\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "THÔNG TƯ",\n      "family": "extended",\n      "source": "Existing pipeline scope / explicitly named instrument"\n    },\n    {\n      "abbreviation": null,\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "THÔNG TƯ LIÊN TỊCH",\n      "family": "extended",\n      "source": "Existing pipeline scope / explicitly named instrument"\n    },\n    {\n      "abbreviation": null,\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "LUẬT",\n      "family": "extended",\n      "source": "Existing pipeline scope / explicitly named instrument"\n    },\n    {\n      "abbreviation": null,\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "PHÁP LỆNH",\n      "family": "extended",\n      "source": "Existing pipeline scope / explicitly named instrument"\n    }\n  ],\n  "levels": {\n    "priority_level": {\n      "1_HỎA TỐC": [\n        "HỎA TỐC",\n        "HOẢ TỐC",\n        "OA TỐC",\n        "HA TỐC"\n      ],\n      "2_KHẨN": [\n        "KHẨN"\n      ],\n      "3_THƯỢNG KHẨN": [\n        "THƯỢNG KHẨN"\n      ]\n    },\n    "security_level": {\n      "1_MẬT": [\n        "MẬT"\n      ],\n      "2_TỐI MẬT": [\n        "TỐI MẬT"\n      ],\n      "3_TUYỆT MẬT": [\n        "TUYỆT MẬT"\n      ]\n    }\n  },\n  "ocr_variant_policy": "OA TỐC / HA TỐC only match in compact stamp zone, never a global replacement",\n  "policy": "Exact/folded lookup supports anchors and enum normalization. Entity/fuzzy replacement disabled.",\n  "source_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2020/03/30.signed.pdf",\n  "version": 1\n}')

DEVELOPMENT_LEXICON = json.loads('{\n  "entries": [\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "PHÒNG QUẢN LÝ NIÊM YẾT",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/+TB-QLNY_2018-12-11.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "Hà nội",\n      "entity_type": "location",\n      "field_hints": [\n        "province"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/+TB-QLNY_2018-12-11.json",\n        "preprocessed_data/116+2020+TT-BTC_2020-12-31.json",\n        "preprocessed_data/117+2020+TT-BTC_2020-12-31.json",\n        "preprocessed_data/118+2020+TT-BTC_2020-12-31.json",\n        "preprocessed_data/119+2020+TT-BTC_2020-12-31.json",\n        "preprocessed_data/1191+QĐ-TTg_2017-08-14.json",\n        "preprocessed_data/120+2020+TT-BTC_2020-12-31.json",\n        "preprocessed_data/121+2020+TT-BTC_2020-12-31.json",\n        "preprocessed_data/1432+QĐ-BXD_2026-08-14.json",\n        "preprocessed_data/15+MED+2020_2020-03-04.json",\n        "preprocessed_data/155+2020+NĐ-CP_2020-12-31.json",\n        "preprocessed_data/156+2020+NĐ-CP_2020-12-31.json",\n        "preprocessed_data/1666+QĐ-TTg_2026-08-27.json",\n        "preprocessed_data/1671+QĐ-TTg_2026-08-28.json",\n        "preprocessed_data/1679+QĐ-TTg_2026-08-28.json",\n        "preprocessed_data/1685+QĐ-TTg_2026-08-30.json",\n        "preprocessed_data/169+KH-UBND_2026-04-24.json",\n        "preprocessed_data/1972+UBND-KT_2026-05-09.json",\n        "preprocessed_data/2125+TB-ĐLTKV_2020-12-21.json",\n        "preprocessed_data/2246+QĐ-BTC_2026-08-17.json",\n        "preprocessed_data/2271+QĐ-TTg_2021-12-31.json",\n        "preprocessed_data/2399+QĐ-BTC_2017-11-21.json",\n        "preprocessed_data/255+NQ-CP_2026-08-27.json",\n        "preprocessed_data/258+NQ-CP_2026-08-31.json",\n        "preprocessed_data/27+2026+VBHN-ND-BTC_2026-08-27.json",\n        "preprocessed_data/27+CT-TTg_2026-06-25.json",\n        "preprocessed_data/29+CT-TTg_2026-07-16.json",\n        "preprocessed_data/340+2026+NĐ-CP_2026-08-28.json",\n        "preprocessed_data/341+2026+NĐ-CP_2026-09-01.json",\n        "preprocessed_data/37+2026+QĐ-UBND_2026-03-31.json",\n        "preprocessed_data/60+CĐ-TTg_2026-08-29.json",\n        "preprocessed_data/61+2026+QĐ-UBND_2026-06-01.json",\n        "preprocessed_data/61+CĐ-TTg_2026-08-31.json",\n        "preprocessed_data/62+CĐ-TTg_2026-08-31.json",\n        "preprocessed_data/66+2026+VBHN-TT-BXD_2026-08-27.json",\n        "preprocessed_data/69+2026+QĐ-UBND_2026-06-18.json",\n        "preprocessed_data/71+2017+NĐ-CP_2017-06-06.json",\n        "preprocessed_data/87+2017+TT-BTC_2017-08-15.json",\n        "preprocessed_data/95+2017+TT-BTC_2017-09-22.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "CÔNG TY CỔ PHẦN THỰC PHẨM BÍCH CHI",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/03+BC.HĐQT_2020-02-20.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "Đồng Tháp",\n      "entity_type": "location",\n      "field_hints": [\n        "province"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/03+BC.HĐQT_2020-02-20.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "CÔNG TY CỔ PHẦN DU LỊCH - THƯƠNG MẠI TN",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/03+NQ-ĐHCĐ_2017-04-25.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "Tây Ninh",\n      "entity_type": "location",\n      "field_hints": [\n        "province"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/03+NQ-ĐHCĐ_2017-04-25.json",\n        "preprocessed_data/138+DLTM_2017-04-28.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "BỘ TÀI CHÍNH",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/116+2020+TT-BTC_2020-12-31.json",\n        "preprocessed_data/117+2020+TT-BTC_2020-12-31.json",\n        "preprocessed_data/118+2020+TT-BTC_2020-12-31.json",\n        "preprocessed_data/119+2020+TT-BTC_2020-12-31.json",\n        "preprocessed_data/120+2020+TT-BTC_2020-12-31.json",\n        "preprocessed_data/121+2020+TT-BTC_2020-12-31.json",\n        "preprocessed_data/2246+QĐ-BTC_2026-08-17.json",\n        "preprocessed_data/2399+QĐ-BTC_2017-11-21.json",\n        "preprocessed_data/27+2026+VBHN-ND-BTC_2026-08-27.json",\n        "preprocessed_data/87+2017+TT-BTC_2017-08-15.json",\n        "preprocessed_data/95+2017+TT-BTC_2017-09-22.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "THỦ TƯỚNG CHÍNH PHỦ",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/1191+QĐ-TTg_2017-08-14.json",\n        "preprocessed_data/1666+QĐ-TTg_2026-08-27.json",\n        "preprocessed_data/1671+QĐ-TTg_2026-08-28.json",\n        "preprocessed_data/1679+QĐ-TTg_2026-08-28.json",\n        "preprocessed_data/1685+QĐ-TTg_2026-08-30.json",\n        "preprocessed_data/2271+QĐ-TTg_2021-12-31.json",\n        "preprocessed_data/27+CT-TTg_2026-06-25.json",\n        "preprocessed_data/29+CT-TTg_2026-07-16.json",\n        "preprocessed_data/60+CĐ-TTg_2026-08-29.json",\n        "preprocessed_data/61+CĐ-TTg_2026-08-31.json",\n        "preprocessed_data/62+CĐ-TTg_2026-08-31.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "BỘ TÀI CHÍNH SỞ GIAO DỊCH CHỨNG KHOÁN TP.HCM",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/131+QĐ-SGDHCM_2017-04-18.json",\n        "preprocessed_data/158+QĐ-SGDHCM_2015-05-12.json",\n        "preprocessed_data/237+QĐ-SGDHCM_2017-07-05.json",\n        "preprocessed_data/342+QĐ-SGDHCM_2016-08-22.json",\n        "preprocessed_data/346+QĐ-SGDHCM_2016-08-23.json",\n        "preprocessed_data/53+QĐ-SGDHCM_2015-11-25.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "Thành phố Hồ Chí Minh",\n      "entity_type": "location",\n      "field_hints": [\n        "province"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/131+QĐ-SGDHCM_2017-04-18.json",\n        "preprocessed_data/158+QĐ-SGDHCM_2015-05-12.json",\n        "preprocessed_data/1637+SGDHCM-NY_2017-11-22.json",\n        "preprocessed_data/237+QĐ-SGDHCM_2017-07-05.json",\n        "preprocessed_data/340+QĐ-SGDHCM_2016-08-19.json",\n        "preprocessed_data/342+QĐ-SGDHCM_2016-08-22.json",\n        "preprocessed_data/346+QĐ-SGDHCM_2016-08-23.json",\n        "preprocessed_data/53+QĐ-SGDHCM_2015-11-25.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "ỦY BAN NHÂN DÂN TỈNH NGHỆ AN",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/1321+QĐ-UBND_2025-05-12.json",\n        "preprocessed_data/17+BC-UBND_2025-01-09.json",\n        "preprocessed_data/2472+QĐ-UBND_2025-08-01.json",\n        "preprocessed_data/32+CĐ-UBND_2024-08-28.json",\n        "preprocessed_data/503+KH-UBND_2023-07-06.json",\n        "preprocessed_data/5324+UBND-TH_2025-06-11.json",\n        "preprocessed_data/819+KH-UBND_2026-08-31.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "Nghệ An",\n      "entity_type": "location",\n      "field_hints": [\n        "province"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/1321+QĐ-UBND_2025-05-12.json",\n        "preprocessed_data/17+BC-UBND_2025-01-09.json",\n        "preprocessed_data/2472+QĐ-UBND_2025-08-01.json",\n        "preprocessed_data/32+CĐ-UBND_2024-08-28.json",\n        "preprocessed_data/503+KH-UBND_2023-07-06.json",\n        "preprocessed_data/5324+UBND-TH_2025-06-11.json",\n        "preprocessed_data/819+KH-UBND_2026-08-31.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "CÔNG TY CỔ PHẦN DU LỊCH - THƯƠNG MẠI TÂY NINH",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/138+DLTM_2017-04-28.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "BỘ XÂY DỰNG",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/1432+QĐ-BXD_2026-08-14.json",\n        "preprocessed_data/66+2026+VBHN-TT-BXD_2026-08-27.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "CÔNG TY CỔ PHẦN DƯỢC TRUNG ƯƠNG MEDIPLANTEX",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/15+MED+2020_2020-03-04.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "CHÍNH PHỦ",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/155+2020+NĐ-CP_2020-12-31.json",\n        "preprocessed_data/156+2020+NĐ-CP_2020-12-31.json",\n        "preprocessed_data/255+NQ-CP_2026-08-27.json",\n        "preprocessed_data/258+NQ-CP_2026-08-31.json",\n        "preprocessed_data/340+2026+NĐ-CP_2026-08-28.json",\n        "preprocessed_data/341+2026+NĐ-CP_2026-09-01.json",\n        "preprocessed_data/71+2017+NĐ-CP_2017-06-06.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "BỘ TÀI CHÍNH SỞ GIAO DỊCH CHỨNG KHOÁN TPHCM",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/1637+SGDHCM-NY_2017-11-22.json",\n        "preprocessed_data/340+QĐ-SGDHCM_2016-08-19.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "ỦY BAN NHÂN DÂN THÀNH PHỐ HÀ NỘI",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/169+KH-UBND_2026-04-24.json",\n        "preprocessed_data/1972+UBND-KT_2026-05-09.json",\n        "preprocessed_data/37+2026+QĐ-UBND_2026-03-31.json",\n        "preprocessed_data/61+2026+QĐ-UBND_2026-06-01.json",\n        "preprocessed_data/69+2026+QĐ-UBND_2026-06-18.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "TẬP ĐOÀN CÔNG NGHIỆP THAN - KHOÁNG SẢN VIỆT NAM TỔNG CÔNG TY ĐIỆN LỰC - TKV",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/2125+TB-ĐLTKV_2020-12-21.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    },\n    {\n      "aliases": [],\n      "auto_correct": false,\n      "canonical_label": "QUỐC HỘI",\n      "entity_type": "organization",\n      "field_hints": [\n        "officeSender"\n      ],\n      "review_status": "reference_label_requires_review",\n      "sources": [\n        "preprocessed_data/36+2026+QH16_2026-08-24.json",\n        "preprocessed_data/41+2026+QH16_2026-08-24.json",\n        "preprocessed_data/43+2026+QH16_2026-08-24.json"\n      ],\n      "valid_from": null,\n      "valid_to": null\n    }\n  ],\n  "manifest_sha256": "26d52b73e0a9591f1dc6c0ab882f80b60a817100ec8cbc95b050c5cc905334da",\n  "purpose": "Candidate lookup only. Missing validity dates are not evidence of current/historical validity.",\n  "source_split": "development",\n  "version": 1\n}')


def _write_json(path, payload):
    """Atomic JSON write (thay the rule_evaluation.write_json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _dev_fold(text):
    """Fold dung cho lookup_entity_candidates (verbatim tu rule_evaluation.fold)."""
    text = unicodedata.normalize("NFD", str(text).casefold().replace("đ", "d"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def lookup_entity_candidates(text, field, lexicon):
    """Candidate lookup only; never overwrites extracted text (verbatim tu rule_evaluation)."""
    if not text:
        return []
    key = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    matches = []
    for entry in lexicon["entries"]:
        if field not in entry["field_hints"]:
            continue
        labels = [entry["canonical_label"], *entry["aliases"]]
        exact = any(unicodedata.normalize("NFKC", label).casefold().strip() == key for label in labels)
        if exact or any(_dev_fold(label) == _dev_fold(text) for label in labels):
            matches.append({"canonical_label": entry["canonical_label"], "entity_type": entry["entity_type"],
                             "match": "exact" if exact else "folded_candidate", "auto_correct": False,
                             "review_status": entry["review_status"], "sources": entry["sources"]})
    return matches


def save_ocr_debug(img, ocr_boxes, debug_dir, pn):
    """Giu chu ky goc cua notebook; pipeline nay luon truyen debug_dir=None."""
    raise NotImplementedError("save_ocr_debug khong duoc ho tro trong ban standalone (debug_dir phai la None)")


# ---- rule cell 3 (verbatim tu 6_1_optimize_rule.ipynb) ----
from typing import TypedDict

class OCRBox(TypedDict, total=False):
    bbox: list[float]
    bbox_row: list[float]
    quad: list[list[float]]
    text: str

class PageBlock(TypedDict, total=False):
    type: str
    bbox: list[float]
    score: float
    source_layout_id: int
    content_type: str
    content: str | None
    text_items: list[OCRBox]

from dataclasses import dataclass

@dataclass(frozen=True)
class LayoutLabelSchema:
    table_types: frozenset[str]
    skip_types: frozenset[str]
    title_types: frozenset[str]
    h2_types: frozenset[str]

config = SimpleNamespace(
    map_overlap_threshold=0.5, figure_drop_types=["image"], figure_drop_ratio=0.4,
    line_overlap_min=0.5, anchor_band_mult=1.0, anchor_min_width=100,
    single_page_ratio_max=0.85, spanning_min_ratio=0.15)
schema = LayoutLabelSchema(
    table_types=frozenset({"table"}),
    skip_types=frozenset({"chart", "formula", "formula_number", "algorithm", "number", "aside_text", "footnote"}),
    title_types=frozenset({"doc_title"}), h2_types=frozenset({"paragraph_title"}))
pipeline_adapter = SimpleNamespace(config=config, layout=SimpleNamespace(label_schema=schema))

# ---- rule cell 5 (verbatim tu 6_1_optimize_rule.ipynb) ----
def overlap_ratio(text_bbox: list[float], layout_bbox: list[float]) -> float:
    ax0, ay0, ax1, ay1 = text_bbox
    bx0, by0, bx1, by1 = layout_bbox
    inter = max(0., min(ax1, bx1) - max(ax0, bx0)) * max(0., min(ay1, by1) - max(ay0, by0))
    area = max(1., (ax1 - ax0) * (ay1 - ay0))
    return inter / area

def block_area(bbox: list[float]) -> float:
    return max(0., bbox[2] - bbox[0]) * max(0., bbox[3] - bbox[1])

def best_match_block(
    text_bbox: list[float],
    layout_blocks: list[PageBlock],
    skip_types: frozenset[str],
    overlap_threshold: float,
) -> PageBlock | None:
    best_non_skip, best_non_skip_s, best_non_skip_area = None, overlap_threshold, None
    best_skip, best_skip_s, best_skip_area = None, overlap_threshold, None

    for b in layout_blocks:
        s = overlap_ratio(text_bbox, b["bbox"])
        if s <= overlap_threshold:
            continue
        area = block_area(b["bbox"])
        is_skip = b["type"].lower() in skip_types

        if is_skip:
            if best_skip is None or s > best_skip_s + 1e-6:
                best_skip, best_skip_s, best_skip_area = b, s, area
            elif abs(s - best_skip_s) <= 1e-6 and area < best_skip_area:
                best_skip, best_skip_s, best_skip_area = b, s, area
        else:
            if best_non_skip is None or s > best_non_skip_s + 1e-6:
                best_non_skip, best_non_skip_s, best_non_skip_area = b, s, area
            elif abs(s - best_non_skip_s) <= 1e-6 and area < best_non_skip_area:
                best_non_skip, best_non_skip_s, best_non_skip_area = b, s, area

    # Non-skip always wins if any candidate clears the threshold, even if its
    # overlap is lower than the best skip-type candidate.
    return best_non_skip if best_non_skip is not None else best_skip

def dedup_nested_blocks(blocks: list[PageBlock], containment_thr: float = 0.98) -> list[PageBlock]:
    """Keep one duplicate; drop strictly larger same-content containers."""
    removed = set()
    for i, a in enumerate(blocks):
        if i in removed:
            continue
        for j in range(i + 1, len(blocks)):
            if j in removed:
                continue
            b = blocks[j]
            if a["content_type"] != b["content_type"]:
                continue
            ab = overlap_ratio(b["bbox"], a["bbox"])
            ba = overlap_ratio(a["bbox"], b["bbox"])
            if ab >= containment_thr and ba >= containment_thr:
                if float(b.get("score", 0)) > float(a.get("score", 0)):
                    removed.add(i)
                    break
                removed.add(j)
            elif ab >= containment_thr and block_area(a["bbox"]) > block_area(b["bbox"]):
                removed.add(i)
                break
            elif ba >= containment_thr and block_area(b["bbox"]) > block_area(a["bbox"]):
                removed.add(j)
    return [b for i, b in enumerate(blocks) if i not in removed]


def map_text_to_blocks(
    ocr_boxes: list[OCRBox],
    layout_blocks: list[PageBlock],
    skip_types: frozenset[str],
    overlap_threshold: float,
) -> list[OCRBox]:
    """First mapping pass for ALL ocr_boxes into layout_blocks (resets
    text_items on every block first). Returns the OCR boxes that matched no
    layout block -- likely regions the layout model missed (the OCR detector
    still found text there, but no layout bbox covers it)."""
    for b in layout_blocks:
        b["text_items"] = []
    unmatched = []
    for o in ocr_boxes:
        best = best_match_block(o["bbox"], layout_blocks, skip_types, overlap_threshold)
        if best is not None:
            best["text_items"].append(o)
        else:
            unmatched.append(o)
    return unmatched

def reassign_ocr_boxes(
    ocr_boxes: list[OCRBox],
    layout_blocks: list[PageBlock],
    skip_types: frozenset[str],
    overlap_threshold: float,
) -> list[OCRBox]:
    """Re-maps a SUBSET of ocr_boxes (e.g. boxes freed after dropping a
    'figure' block that absorbed too many OCR boxes) into the REMAINING
    layout_blocks, using the exact same overlap + tie-break rule as
    map_text_to_blocks() above.

    UNLIKE map_text_to_blocks(): this does NOT reset text_items on
    layout_blocks -- it only appends, so it doesn't lose the correct mapping
    already established for other blocks (text/table/...)."""
    unmatched = []
    for o in ocr_boxes:
        best = best_match_block(o["bbox"], layout_blocks, skip_types, overlap_threshold)
        if best is not None:
            best["text_items"].append(o)
        else:
            unmatched.append(o)
    return unmatched

def unmatched_to_blocks(unmatched: list[OCRBox]) -> list[PageBlock]:
    """Turns each unmatched OCR box into its own independent block, type =
    'text' like a normal block (no separate type) -- no clustering, each box
    = 1 block, bbox = that OCR box's exact bbox. Goes through the same
    content-building / sort_reading_order steps as any other 'text' block,
    no special-casing needed downstream.

    score = 0.0 since this isn't a layout model confidence (the layout model
    never detected this region at all)."""
    blocks: list[PageBlock] = []
    for o in unmatched:
        blocks.append({
            "type": "text",
            "bbox": list(o["bbox"]),
            "score": 0.0,
            "content_type": "text",
            "content": None,
            "text_items": [o],
        })
    return blocks

# ---- rule cell 7 (verbatim tu 6_1_optimize_rule.ipynb) ----
import math

def is_double_page(img_width: float, img_height: float,
                   single_page_ratio_max: float = 0.85, blocks=None) -> bool:
    """Require two distinct header clusters, not merely landscape aspect ratio.

    Ambiguous wide pages retain one-page reading order; no guessed seam.
    """
    if img_width / max(img_height, 1) <= single_page_ratio_max or not blocks:
        return False
    sides = set()
    for block in blocks:
        text = _fold(str(block.get("content", "")))
        if "cong hoa xa hoi chu nghia viet nam" not in text:
            continue
        x0, y0, x1, y1 = block["bbox"]
        if y0 > img_height * 0.45 or x1 - x0 > img_width * 0.55:
            continue
        sides.add("left" if (x0+x1)/2 < img_width/2 else "right")
    return sides == {"left", "right"}


def _quad_geom(b: dict, anchor_min_width: float = 100):
    """Returns (cx, cy, w, h, theta) from the raw 'quad' (4 points, NOT yet
    rotated) if present. theta = None if the box has no quad or is too
    narrow to measure a reliable angle (short boxes get their corners
    rounded toward 0 by the detector, so a measured angle wouldn't be
    trustworthy). No 'quad' (e.g. a layout block in sort_reading_order, or a
    box built manually elsewhere) -> derive from 'bbox'/'bbox_row', theta is
    always None."""
    q = b.get("quad")
    if q:
        p0, p1, p2, p3 = q
        cx = sum(p[0] for p in q) / 4
        cy = sum(p[1] for p in q) / 4
        w = p1[0] - p0[0]
        h = (math.hypot(p3[0] - p0[0], p3[1] - p0[1]) + math.hypot(p2[0] - p1[0], p2[1] - p1[1])) / 2
        theta = None
        if w >= anchor_min_width:
            t_top = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            t_bot = math.atan2(p2[1] - p3[1], p2[0] - p3[0])
            theta = (t_top + t_bot) / 2
        return cx, cy, w, h, theta
    x0, y0, x1, y1 = b.get("bbox_row", b["bbox"])
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, None

def _geo(b: dict) -> list[float]:
    """Axis-aligned bbox used for the fallback path (no reliable
    quad/theta) -- prefers 'bbox_row' (already deskewed by the page's
    overall angle) if present, falls back to the raw 'bbox'."""
    return b.get("bbox_row", b["bbox"])

def _tb_rows_fallback(lst: list[dict], overlap_min: float = 0.5) -> list[list[dict]]:
    """Groups into rows via axis-aligned y-overlap (best-fit, fixed anchor)
    -- used for boxes that no local anchor picked up (e.g. every box in a
    block is too short to self-measure an angle, or called for LAYOUT
    blocks in sort_reading_order, which have no 'quad')."""
    lst_sorted = sorted(lst, key=lambda b: _geo(b)[1])
    rows = []
    for item in lst_sorted:
        y0, y1 = _geo(item)[1], _geo(item)[3]
        best_row, best_ratio = None, overlap_min
        for row in rows:
            ry0, ry1 = row["anchor_y0"], row["anchor_y1"]
            inter = max(0., min(y1, ry1) - max(y0, ry0))
            min_h = min(y1 - y0, ry1 - ry0)
            if min_h > 0:
                ratio = inter / min_h
                if ratio >= overlap_min and ratio > best_ratio:
                    best_ratio, best_row = ratio, row
        if best_row is not None:
            best_row["items"].append(item)
        else:
            rows.append({"anchor_y0": y0, "anchor_y1": y1, "items": [item]})
    rows.sort(key=lambda r: r["anchor_y0"])
    return [sorted(r["items"], key=lambda b: _geo(b)[0]) for r in rows]

def tb_rows(
    lst: list[dict],
    overlap_min: float = 0.5,
    band_mult: float = 1.0,
    anchor_min_width: float = 100,
) -> list[list[dict]]:
    if not lst:
        return []

    infos = [(_quad_geom(b, anchor_min_width), b) for b in lst]
    infos.sort(key=lambda x: -x[0][2])  # widest/most-reliable boxes considered as anchors first

    n = len(infos)
    assigned = [False] * n
    lines = []  # list of (cy_anchor, [items left->right])

    for i in range(n):
        if assigned[i]:
            continue
        cx, cy, w, h, theta = infos[i][0]
        if theta is None:
            continue  # not reliable enough to be an anchor, leave for the fallback
        c, s = math.cos(theta), math.sin(theta)
        half_h = max(h, 1.0) / 2
        picked = []
        for j in range(n):
            if assigned[j]:
                continue
            cx2, cy2, *_ = infos[j][0]
            dx, dy = cx2 - cx, cy2 - cy
            perp = -dx * s + dy * c  # perpendicular distance from the anchor's skew direction
            along = dx * c + dy * s  # position along that direction (for left->right sort)
            if abs(perp) <= half_h * band_mult:
                picked.append((along, infos[j][1]))
                assigned[j] = True
        picked.sort(key=lambda x: x[0])
        lines.append((cy, [b for _, b in picked]))

    leftover = [infos[i][1] for i in range(n) if not assigned[i]]
    if leftover:
        for row in _tb_rows_fallback(leftover, overlap_min):
            fy = sum(_geo(b)[1] for b in row) / len(row)
            lines.append((fy, row))

    lines.sort(key=lambda x: x[0])
    return [items for _, items in lines]

def _tb(
    lst: list[dict],
    overlap_min: float = 0.5,
    band_mult: float = 1.0,
    anchor_min_width: float = 100,
) -> list[dict]:
    out = []
    for row in tb_rows(lst, overlap_min, band_mult, anchor_min_width):
        out.extend(row)
    return out

def split_side(bbox: list[float], split_x: float, min_spanning_ratio: float = 0.15) -> str:
    """Determines whether a block belongs to the left / right column, or is
    'spanning' (crosses the seam between 2 pages) based on the RATIO of the
    bbox width on each side of split_x. Only 'spanning' when BOTH sides
    carry a meaningful share (>= min_spanning_ratio) -- avoids mistaking a
    long line of text from one side that just slightly overflows the seam
    for something genuinely straddling both sheets."""
    x0, y0, x1, y1 = bbox
    width = max(1e-6, x1 - x0)

    left_w = max(0., min(x1, split_x) - x0)
    right_w = max(0., x1 - max(x0, split_x))

    left_ratio = left_w / width
    right_ratio = right_w / width

    if left_ratio >= min_spanning_ratio and right_ratio >= min_spanning_ratio:
        return "spanning"
    return "left" if left_ratio >= right_ratio else "right"

def _merge_by_y0(ordered: list[PageBlock], extra: list[PageBlock]) -> list[PageBlock]:
    """Splices `extra` (sorted by bbox y0) into `ordered` at the position
    matching its own y0, without disturbing `ordered`'s existing relative
    order."""
    if not extra:
        return ordered
    result, ei = [], 0
    for b in ordered:
        while ei < len(extra) and extra[ei]["bbox"][1] <= b["bbox"][1]:
            result.append(extra[ei])
            ei += 1
        result.append(b)
    result.extend(extra[ei:])
    return result

def sort_reading_order(
    blocks: list[PageBlock],
    img_width: float,
    img_height: float,
    overlap_min: float = 0.5,
    band_mult: float = 1.0,
    anchor_min_width: float = 100,
    single_page_ratio_max: float = 0.85,
    spanning_min_ratio: float = 0.15,
) -> list[PageBlock]:
    """Orders blocks for reading on a single page.

    'skip' blocks (image/figure/formula/seal/... -- never have content, see
    label_schema.skip_types) are pulled out before row-grouping and spliced
    back in by their own y0 afterward: a page-spanning image/background
    block would otherwise Y-overlap with nearly every other block and drag
    them all into a single reading-order "row", collapsing the whole page's
    order down to a left-to-right sort by x0 alone.
    """
    skip_blocks = [b for b in blocks if b.get("content_type") == "skip"]
    orderable = [b for b in blocks if b.get("content_type") != "skip"]

    if not is_double_page(img_width, img_height, single_page_ratio_max, orderable):
        ordered = _tb(orderable, overlap_min, band_mult, anchor_min_width)
    else:
        split_x = img_width / 2
        left, right, spanning = [], [], []
        for b in orderable:
            side = split_side(b["bbox"], split_x, spanning_min_ratio)
            if side == "spanning":
                spanning.append(b)
            elif side == "left":
                left.append(b)
            else:
                right.append(b)

        col_ordered = _tb(left, overlap_min, band_mult, anchor_min_width) + _tb(right, overlap_min, band_mult, anchor_min_width)
        spanning_sorted = sorted(spanning, key=lambda b: b["bbox"][1])
        ordered = _merge_by_y0(col_ordered, spanning_sorted)

    skip_sorted = sorted(skip_blocks, key=lambda b: b["bbox"][1])
    return _merge_by_y0(ordered, skip_sorted)

# ---- rule cell 9 (verbatim tu 6_1_optimize_rule.ipynb) ----
import re
ALPHA_ITEM_RE = re.compile('^\\s*[a-zđ][\\.\\)]\\s+')
NUM_ITEM_RE = re.compile('^\\s*\\d+[\\.\\)]\\s+')
MIN_LIST_ITEMS = 2

def _list_item_type(line: str) -> str | None:
    """Trả về 'alpha' nếu khớp marker chữ cái (a) b) c)...), 'num' nếu khớp
    marker số (1. 2. 3)...), None nếu không phải list item."""
    if NUM_ITEM_RE.match(line):
        return 'num'
    if ALPHA_ITEM_RE.match(line):
        return 'alpha'
    return None

def build_block_content(line_texts: list[str], preserve_breaks: bool=False, preserve_format: bool=False) -> str:
    """
    Ghép các "hàng" (mỗi hàng = 1 dòng chữ thật, đã gom theo tb_rows) thành
    1 chuỗi content dạng HTML-trong-Markdown:
      - Các dòng THƯỜNG liên tiếp -> nối bằng ' ' (space) thành 1 đoạn văn
        liên tục -- dòng OCR chỉ là chỗ bị ngắt do word-wrap theo khổ trang
        gốc, không phải ranh giới đoạn thật, nên không giữ thành xuống dòng
        cứng trong output. Trừ khi `preserve_breaks=True` (dùng cho block
        title/header -- xem document_pipeline.py's call site): ở đó mỗi
        dòng THƯỜNG là 1 đơn vị hiển thị có chủ đích (vd tên công ty 1
        dòng, địa chỉ 1 dòng khác), nên nối bằng '<br>
' để giữ xuống dòng
        thay vì gộp.
      - Các dòng LIST ITEM liên tiếp, CÙNG NHÓM marker (toàn 'alpha' hoặc
        toàn 'num') -> gom thành 1 khối <ul>, mỗi dòng là 1 <li>...</li> --
        chỉ khi số dòng liên tiếp cùng nhóm >= MIN_LIST_ITEMS.
      - Nếu marker đổi nhóm giữa chừng (ví dụ '1. ...' rồi 'a) ...') ->
        đóng khối list hiện tại (theo đúng rule MIN_LIST_ITEMS ở trên) và
        mở khối MỚI cho nhóm marker khác, KHÔNG gộp chung 1 <ul>.
      - Nếu chỉ có 1 dòng lẻ khớp marker (dù đứng riêng hay bị đổi nhóm
        ngay sau đó), coi như text thường, không bọc <ul>.
      - 1 khối <ul> vẫn nối với text-run liền kề bằng '
' (ranh giới
        block-level HTML thật sự, không thể gộp bằng space/<br>) -- chỉ 2
        text-run liền kề nhau mới áp dụng rule ' ' hoặc '<br>
' ở trên.
    """
    if not preserve_format:
        plain_sep = (' ', '\n')[int(preserve_breaks)]
        return plain_sep.join(line_texts)
    plain_sep = '<br>\n' if preserve_breaks else ' '
    segments = []
    buffer = []
    list_buffer = []
    list_type = None

    def flush_buffer():
        if buffer:
            segments.append(('text', plain_sep.join(buffer)))
            buffer.clear()

    def flush_list():
        nonlocal list_type
        if len(list_buffer) >= MIN_LIST_ITEMS:
            items = '\n'.join((f'<li>{t}</li>' for t in list_buffer))
            segments.append(('ul', f'<ul>\n{items}\n</ul>'))
        elif len(list_buffer) == 1:
            segments.append(('text', list_buffer[0]))
        list_buffer.clear()
        list_type = None
    for line in line_texts:
        t = _list_item_type(line)
        if t is not None:
            if list_buffer and t != list_type:
                flush_list()
            if not list_buffer:
                flush_buffer()
            list_buffer.append(line)
            list_type = t
        elif list_buffer:
            list_buffer[-1] = list_buffer[-1] + ' ' + line
        else:
            buffer.append(line)
    flush_buffer()
    flush_list()
    if not segments:
        return ''
    result = segments[0][1]
    for i in range(1, len(segments)):
        (prev_kind, _) = segments[i - 1]
        (cur_kind, cur_content) = segments[i]
        sep = '\n' if prev_kind == 'ul' or cur_kind == 'ul' else plain_sep
        result += sep + cur_content
    return result

# ---- rule cell 11 (verbatim tu 6_1_optimize_rule.ipynb) ----
logger = logging.getLogger("vbhc_rules")

def assemble_page(self, pn, prepped, blocks, ocr_boxes, debug_dir=None):
    """Reading-order/content assembly for one page. `blocks` (from
        process_pdf()'s Phase 2.5/2.6) already has table content filled
        in, and `ocr_boxes` (from process_pdf()'s Phase 3a/3b) already has
        every text box detected+recognized -- both batched across the
        WHOLE document beforehand, so nothing in this method calls a
        model anymore."""
    cfg = self.config
    label_schema = self.layout.label_schema
    img = prepped.img
    (w, h) = img.size
    timings = prepped.timings
    n_raw_blocks = len(blocks)
    t_stage = time.time()

    def _mark(name: str) -> None:
        nonlocal t_stage
        now = time.time()
        timings[name] = now - t_stage
        t_stage = now
    if debug_dir:
        save_ocr_debug(img, ocr_boxes, debug_dir, pn)
    _mark('ocr_debug_save')
    blocks = dedup_nested_blocks(blocks)
    unmatched = map_text_to_blocks(ocr_boxes, blocks, label_schema.skip_types, cfg.map_overlap_threshold)
    n_ocr_total = len(ocr_boxes)
    if n_ocr_total > 0:
        figure_to_drop = [b for b in blocks if b['type'].lower() in cfg.figure_drop_types and len(b.get('text_items', [])) / n_ocr_total > cfg.figure_drop_ratio]
        if figure_to_drop:
            reclaim_boxes = []
            for b in figure_to_drop:
                reclaim_boxes.extend(b['text_items'])
                logger.debug("Page %d: dropping block '%s' bbox=%s (absorbed %d/%d = %.0f%% of OCR boxes)", pn + 1, b['type'], [round(v) for v in b['bbox']], len(b['text_items']), n_ocr_total, 100 * len(b['text_items']) / n_ocr_total)
            drop_ids = {id(b) for b in figure_to_drop}
            blocks = [b for b in blocks if id(b) not in drop_ids]
            unmatched.extend(reassign_ocr_boxes(reclaim_boxes, blocks, label_schema.skip_types, cfg.map_overlap_threshold))
    if unmatched:
        extra_blocks = unmatched_to_blocks(unmatched)
        blocks.extend(extra_blocks)
        logger.debug('Page %d: %d OCR boxes unmatched -> %d extra blocks', pn + 1, len(unmatched), len(extra_blocks))
    _mark('mapping')
    preserve_breaks_types = label_schema.title_types | label_schema.h2_types | {'header'}
    for b in blocks:
        if b['content_type'] == 'text':
            rows = tb_rows(b['text_items'], cfg.line_overlap_min, cfg.anchor_band_mult, cfg.anchor_min_width)
            line_texts = []
            for row in rows:
                line = ' '.join((t['text'] for t in row if t['text'].strip()))
                if line:
                    line_texts.append(line)
            b['extraction_text'] = '\n'.join(line_texts)
            b['ocr_lines'] = [dict(item) for row in rows for item in row]
            preserve_breaks = b['type'].lower() in preserve_breaks_types
            b['content'] = build_block_content(line_texts, preserve_breaks=preserve_breaks)
    _mark('content_build')
    blocks = sort_reading_order(blocks, img_width=w, img_height=h, overlap_min=cfg.line_overlap_min, band_mult=cfg.anchor_band_mult, anchor_min_width=cfg.anchor_min_width, single_page_ratio_max=cfg.single_page_ratio_max, spanning_min_ratio=cfg.spanning_min_ratio)
    _mark('reading_order')
    elapsed = sum(timings.values())
    n_tbl = sum((1 for b in blocks if b['content_type'] == 'table'))
    n_text = sum((1 for b in blocks if b['content_type'] == 'text'))
    timings_str = ' '.join((f'{k}={v:.2f}s' for (k, v) in timings.items()))
    logger.info('Page %d done in %.1fs: layout=%d ocr=%d table=%d text=%d | %s', pn + 1, elapsed, n_raw_blocks, len(ocr_boxes), n_tbl, n_text, timings_str)
    return blocks

# ---- rule cell 13 (verbatim tu 6_1_optimize_rule.ipynb) ----
"""Deterministic VBHC metadata extraction from OCR/layout page blocks.

This module deliberately has no model dependency.  It consumes the ordered
``PageBlock`` objects produced by :mod:`module.pipeline.document_pipeline`
and applies the anchors/relative layout rules documented in
``docs/vbhc_administrative_layout_rules.md``.

Matching uses folded text (NFKC, case-insensitive, accent-insensitive) so
common OCR accent errors do not hide anchors.  Returned values always come
from the original OCR text, apart from the required date/enum canonical forms
and the inferred ``Công văn`` type.
"""

import html

import re

import unicodedata

from dataclasses import dataclass

from datetime import date

from typing import TYPE_CHECKING, Any, Iterable

TEXT_FIELDS = (
    "type",
    "title",
    "code",
    "documentDate",
    "officeSender",
    "recipients",
    "signer",
    "priority_level",
    "security_level",
    "first_recipients",
    "signer_title",
    "province",
    "receiverDate",
)

_TAG_RE = re.compile(r"<[^>]+>")

_SPACE_RE = re.compile(r"[ \t\r\f\v]+")

_BULLET_RE = re.compile(r"(?:^|\s)[\-–—•]\s+")

_LONG_DATE_RE = re.compile(
    r"ng[aà]y\s*[:.]?\s*(\d{1,2})\s*th[aá]ng\s*(\d{1,2})\s*n[aă]m\s*(\d{4})",
    re.IGNORECASE,
)

_SLASH_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})(?!\d)")

_CODE_RE = re.compile(
    r"(?:^|\n|\b)\s*S[oố]\s*[:：.]?\s*([0-9A-Za-zÀ-ỹĐđ./\-]+(?:(?!\s*(?:ng[aà]y|v/v|k[ií]nh|n[oơ]i|c[oộ]ng))\s+[0-9A-Za-zÀ-ỹĐđ./\-]+)*)",
    re.IGNORECASE | re.MULTILINE,
)

_INLINE_TYPE_CODE_RE = re.compile(
    r"(?:^|\n|\b)\s*(?:LUẬT|NGHỊ\s+ĐỊNH|NGHỊ\s+QUYẾT|QUYẾT\s+ĐỊNH|"
    r"THÔNG\s+TƯ|CHỈ\s+THỊ)\s+S[oố]\s*[:：.]?\s*"
    r"([0-9]*[A-Za-zÀ-ỹĐđ0-9./\-]+(?:\s*[/\-.]\s*[0-9A-Za-zÀ-ỹĐđ]+)*)",
    re.IGNORECASE | re.MULTILINE,
)

_VV_RE = re.compile(r"(?<!\w)V\s*/\s*v\s*:?")

_FIRST_RECIPIENT_RE = re.compile(r"K[ií]nh\s+g[uử]i\s*[:：]?", re.IGNORECASE)

_RECIPIENT_RE = re.compile(r"N[oơ]i\s+nh[aậ]n\s*[:：]?", re.IGNORECASE)

_SAVE_RE = re.compile(r"(?:^|[;\n])\s*(?:[-–—•]\s*)?L[uư]u\b\s*[:：]?", re.IGNORECASE)

_SAVE_SUFFIX_RE = re.compile(
    r"[\s,;]\(?(?:[-–—•]\s*)?L[uư]u\s*[:：]?\s*(?:VT|VP|HS|HĐ|HD|TK|HC|QT)\b.*$",
    re.IGNORECASE,
)

_DOCUMENT_TYPES = tuple(sorted(
    {entry["canonical_label"] for entry in RULE_LEXICON["document_types"]},
    key=len, reverse=True,
))

_TYPE_RE = re.compile(
    r"(?:^|\n)\s*(" + "|".join(re.escape(v) for v in _DOCUMENT_TYPES) + r")\b",
    re.IGNORECASE,
)

_TYPE_FOLDED_TO_CANONICAL: dict[str, str] = {}

_TYPE_FOLDED_RE: re.Pattern[str] | None = None

_AUTHORITY_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(
        re.escape(value).replace(r"\ ", r"\s+")
        for value in sorted(RULE_LEXICON["authority_prefixes"] + RULE_LEXICON["authority_titles"], key=len, reverse=True)
    ) + r")(?!\w)", re.IGNORECASE,
)

_ORG_RE = re.compile(
    r"\b(?:ỦY\s+BAN|UỶ\s+BAN|BỘ|SỞ|CỤC|VỤ|BAN|PHÒNG|VĂN\s+PHÒNG|CHÍNH\s+PHỦ|"
    r"THỦ\s+TƯỚNG|QUỐC\s+HỘI|HỘI\s+ĐỒNG|TÒA\s+ÁN|VIỆN\s+KIỂM\s+SÁT|TRUNG\s+TÂM|"
    r"NGÂN\s+HÀNG|CÔNG\s+TY|TỔNG\s+CÔNG\s+TY)\b",
    re.IGNORECASE,
)

_ORG_FOLDED_RE = re.compile(
    r"\b(?:uy ban|bo|so|cuc|vu|ban(?! hanh)|phong|van phong|chinh phu|"
    r"thu tuong|quoc hoi|hoi dong|toa an|vien kiem sat|trung tam|"
    r"ngan hang|cong ty|tong cong ty)\b"
)

_ANNOTATION_FOLDED_MARKERS = (
    "nguoi ky",
    "nguoi ki",
    "thoi gian ky",
    "co quan phat hanh",
    "cong bao",
    "ky boi",
    "chu ky so",
)

_TITLE_WORDS_FOLDED = {
    "tm",
    "kt",
    "tl",
    "tuq",
    "q",
    "chu",
    "tich",
    "pho",
    "giam",
    "doc",
    "bo",
    "truong",
    "nguoi",
    "dai",
    "dien",
    "uy",
    "quyen",
    "ban",
    "hoi",
    "dong",
    "tong",
    "thuc",
    "hien",
    "cong",
    "thong",
    "tin",
    "cuc",
    "chanh",
    "nhiem",
    "toa",
}

def _has_org_keyword(text: str, folded: str | None = None) -> bool:
    """Match issuing-authority keywords on raw or folded (accent-insensitive) text."""
    if _ORG_RE.search(text):
        return True
    return bool(_ORG_FOLDED_RE.search(folded if folded is not None else _fold(text)))

def _is_annotation_header(text: str, folded: str | None = None) -> bool:
    """Detect digital-signature / PDF annotation blocks, never an officeSender."""
    folded_text = folded if folded is not None else _fold(text)
    if any(marker in folded_text for marker in _ANNOTATION_FOLDED_MARKERS):
        return True
    return "@" in text or "mail:" in folded_text

@dataclass(frozen=True)
class _Block:
    page: int
    index: int
    type: str
    content_type: str
    text: str
    folded: str
    bbox: tuple[float, float, float, float]
    bbox_pixels: tuple[float, float, float, float]
    source_layout_id: int | None
    geometry_reliable: bool
    source_lines: tuple[dict, ...] = ()

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

def _plain_text(value: str | None) -> str:
    """Remove internal HTML wrappers without otherwise normalizing output."""
    if not value:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</li\s*>", "\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub("", text)
    lines = [_SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()

def _fold(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("đ", "d")
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return _SPACE_RE.sub(" ", value).strip()

def _canonical_date(day: str, month: str, year: str) -> str:
    try:
        parsed = date(int(year), int(month), int(day))
    except ValueError:
        return ""
    return parsed.strftime("%d/%m/%Y")

def _find_date(text: str) -> str:
    for regex in (_LONG_DATE_RE, _SLASH_DATE_RE):
        for match in regex.finditer(unicodedata.normalize("NFKC", text)):
            value = _canonical_date(*match.groups())
            if value:
                return value
    return ""

def _build_folded_type_matcher() -> None:
    """Build the accent-insensitive document-type matcher (needs _fold)."""
    global _TYPE_FOLDED_TO_CANONICAL, _TYPE_FOLDED_RE
    _TYPE_FOLDED_TO_CANONICAL = {_fold(v): v for v in _DOCUMENT_TYPES}
    _TYPE_FOLDED_RE = re.compile(
        r"(?:^|\n)\s*(" + "|".join(sorted(_TYPE_FOLDED_TO_CANONICAL, key=len, reverse=True)) + r")\b"
    )

_build_folded_type_matcher()

def _match_document_type(block: _Block) -> tuple[str, str, int] | None:
    """Match a line-start document-type heading; return (canonical, tail, line).

    Matching runs on folded text so a single damaged diacritic in the heading
    (e.g. "CHÌ THỊ") still resolves to the canonical label ("CHỈ THỊ").
    ``tail`` is the remainder of the same original line after the heading.
    """
    assert _TYPE_FOLDED_RE is not None
    match = _TYPE_FOLDED_RE.search(block.folded)
    if not match:
        return None
    canonical = _TYPE_FOLDED_TO_CANONICAL[match.group(1)]
    # Group 1 starts at the heading itself (the leading (?:^|\n)\s* may span
    # lines), so its line index maps back to the original line 1:1.
    line_index = block.folded.count("\n", 0, match.start(1))
    original_lines = block.text.splitlines()
    if line_index >= len(original_lines):
        return canonical, "", line_index
    words = original_lines[line_index].split()
    drop = len(canonical.split())
    tail_first = " ".join(words[drop:])
    remaining = "\n".join([tail_first, *original_lines[line_index + 1 :]]).strip(" \n:-–—")
    return canonical, remaining, line_index

def _normalise_code(value: str) -> str:
    """Keep meaningful punctuation. Never synthesize a hyphen from whitespace."""
    value = unicodedata.normalize("NFC", value).strip()
    value = re.sub(r"\s*([/\-])\s*", r"\1", value)
    value = re.sub(r"[ \t]+", " ", value)
    return value.rstrip(";,:")


def _field(value: str) -> dict[str, str]:
    return {"value": value, "type": "string"}

def _zone(block: _Block, *, x0: float = 0.0, x1: float = 1.0, y0: float = 0.0, y1: float = 1.0) -> bool:
    # When crop/rotation transforms cannot be mapped back reliably, anchors
    # remain authoritative and geometry must not hard-reject a candidate.
    if not block.geometry_reliable:
        return True
    return block.x1 >= x0 and block.x0 <= x1 and block.y1 >= y0 and block.y0 <= y1

def _make_blocks(
    pages_blocks: list[list[PageBlock]],
    original_page_sizes: list[tuple[int, int]],
    processed_page_sizes: list[tuple[int, int]],
    crop_offsets: list[tuple[float, float]],
    geometry_reliable: bool,
) -> list[list[_Block]]:
    pages: list[list[_Block]] = []
    for page_index, blocks in enumerate(pages_blocks):
        original_w, original_h = original_page_sizes[page_index]
        processed_w, processed_h = processed_page_sizes[page_index]
        offset_x, offset_y = crop_offsets[page_index]
        converted: list[_Block] = []
        for block_index, block in enumerate(blocks):
            display_text = _plain_text(block.get("content"))
            # Preserve original signature assembly; raw OCR lines remain available in evidence.
            # A short title/name fragment split by the detector must not become a new name.
            text = (display_text if _has_authority_anchor(display_text)
                    or str(block.get("type", "")).lower() in {"image", "seal"}
                    else _plain_text(block.get("extraction_text", block.get("content"))))
            if not text:
                continue
            # A signature image can include stamp fragments after the title.
            # Only isolate authority lines when the existing parser found no name;
            # never replace a recognized signer using a partial OCR line.
            if (
                str(block.get("type", "")).lower() == "image"
                and _has_authority_anchor(text)
                and not _split_signature(text)[1]
            ):
                items = sorted(block.get("text_items", []), key=lambda item: (item["bbox"][1], item["bbox"][0]))
                authority_lines = []
                for item in items:
                    line = str(item.get("text", "")).strip()
                    if _has_authority_anchor(line):
                        authority_lines.append(line)
                    elif authority_lines:
                        break
                if authority_lines:
                    text = "\n".join(authority_lines)
            raw_bbox = block.get("bbox", [0.0, 0.0, 0.0, 0.0])
            if geometry_reliable:
                denom_w, denom_h = max(original_w, 1), max(original_h, 1)
                bbox = (
                    (raw_bbox[0] + offset_x) / denom_w,
                    (raw_bbox[1] + offset_y) / denom_h,
                    (raw_bbox[2] + offset_x) / denom_w,
                    (raw_bbox[3] + offset_y) / denom_h,
                )
            else:
                denom_w, denom_h = max(processed_w, 1), max(processed_h, 1)
                bbox = (
                    raw_bbox[0] / denom_w,
                    raw_bbox[1] / denom_h,
                    raw_bbox[2] / denom_w,
                    raw_bbox[3] / denom_h,
                )
            converted.append(
                _Block(
                    page=page_index,
                    index=block_index,
                    type=str(block.get("type", "")).lower(),
                    content_type=str(block.get("content_type", "")),
                    text=text,
                    folded=_fold(text),
                    bbox=bbox,
                    bbox_pixels=tuple(float(value) for value in raw_bbox),
                    source_layout_id=block.get("source_layout_id"),
                    geometry_reliable=geometry_reliable,
                    source_lines=tuple(block.get("ocr_lines", block.get("text_items", []))),
                )
            )
        pages.append(converted)
    return pages

def _header_score(blocks: Iterable[_Block]) -> int:
    score = 0
    folded = "\n".join(block.folded for block in blocks if _zone(block, y1=0.45))
    if "cong hoa xa hoi chu nghia viet nam" in folded:
        score += 2
    if "doc lap - tu do - hanh phuc" in folded or "doc lap tu do hanh phuc" in folded:
        score += 1
    if any(_CODE_RE.search(block.text) for block in blocks if _zone(block, y1=0.35)):
        score += 1
    if any(_LONG_DATE_RE.search(block.text) for block in blocks if _zone(block, y1=0.35)):
        score += 1
    if any((_match_document_type(block) or _VV_RE.search(block.text)) for block in blocks if _zone(block, y1=0.45)):
        score += 1
    return score

def _document_start(pages: list[list[_Block]]) -> int:
    for page_index, blocks in enumerate(pages):
        if _header_score(blocks) >= 2:
            return page_index
    return 0

def _next_document_start(pages: list[list[_Block]], start: int) -> int | None:
    for page_index in range(start + 1, len(pages)):
        blocks = pages[page_index]
        folded = "\n".join(block.folded for block in blocks if _zone(block, y1=0.45))
        has_country = "cong hoa xa hoi chu nghia viet nam" in folded
        if has_country and _header_score(blocks) >= 3:
            return page_index
    return None

def _looks_like_person_name(value: str) -> bool:
    value = value.strip(" ,.;:-")
    words = value.split()
    if not 2 <= len(words) <= 6:
        return False
    folded = _fold(value)
    # Reject complete role/organization phrases, not syllables such as Trường.
    if _contains_authority_compound(folded.split()):
        return False
    if any(re.search(r"\b" + re.escape(token) + r"\b", folded)
           for token in ("trung tam", "cong ty", "uy ban", "chung khoan", "thu ky", "chu toa", "truong phong")):
        return False
    role_words = _TITLE_WORDS_FOLDED | {"tuong", "phu", "trach"}
    if sum(_fold(word) in role_words for word in words) >= 2:
        return False
    # Two all-capital OCR fragments are not sufficient evidence of a person.
    if value.isupper():
        return len(words) >= 3 and all(word.isalpha() and len(word) >= 2 for word in words)
    return all(word.isalpha() and word.istitle() for word in words)


def _has_authority_anchor(value: str) -> bool:
    """Match the configured role phrases, without an unbounded OCR wildcard."""
    return bool(_AUTHORITY_RE.search(value))


_AUTHORITY_COMPOUNDS_FOLDED = (
    ("chinh", "phu"),
    ("thu", "tuong"),
    ("bo", "truong"),
    ("thu", "truong"),
    ("pho", "thu"),
    ("uy", "ban"),
    ("quoc", "hoi"),
    ("tong", "giam"),
    ("chanh", "van"),
    ("chu", "tich"),
)

def _contains_authority_compound(folded_words: list[str]) -> bool:
    for first, second in _AUTHORITY_COMPOUNDS_FOLDED:
        for index in range(len(folded_words) - 1):
            if folded_words[index] == first and folded_words[index + 1] == second:
                return True
    return False

def _looks_like_signature_name(value: str) -> bool:
    """Person-name check hardened against all-uppercase authority phrases.

    Title-case names always use the base check.  All-uppercase candidates
    additionally must not contain an authority compound ("CHÍNH PHỦ",
    "THỦ TƯỚNG", ...): those are titles, never signers.
    """
    if not _looks_like_person_name(value):
        return False
    # Note: [a-zà-ỹ] also matches UPPERCASE Vietnamese (U+1E00 block mixes
    # cases inside à-ỹ), so case must be tested with islower() instead.
    if any(ch.islower() for ch in value):
        return True
    return not _contains_authority_compound(_fold(value).split())

def _split_signature(text: str) -> tuple[str, str]:
    """Return (title, signer) while keeping the OCR spelling/punctuation."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        for index in range(len(lines) - 1, -1, -1):
            if _looks_like_signature_name(lines[index]):
                return " ".join(lines[:index]).strip(), lines[index].strip()

    # Layout sometimes merges the entire signature image into one line.
    words = text.strip().split()
    for width in range(min(6, len(words)), 1, -1):
        suffix = " ".join(words[-width:]).strip(" ,.;:-")
        if _looks_like_signature_name(suffix):
            title = " ".join(words[:-width]).strip()
            return title, suffix
    return (" ".join(text.split()).strip(), "") if _has_authority_anchor(text) else ("", "")

def _signature_candidates(blocks: Iterable[_Block]) -> list[tuple[_Block, str, str]]:
    candidates = []
    for block in blocks:
        if block.geometry_reliable:
            width = block.x1 - block.x0
            # Signature/authority blocks are compact and centred in the
            # right column.  They can occur high on a sparse closing page, so
            # vertical position is deliberately not a hard constraint.
            # Rectangle overlap is too permissive here:
            # a full-width body paragraph mentioning a minister/chairperson
            # otherwise becomes a convincing false signature.
            if block.cx < 0.55 or width > 0.62:
                continue
        if len(block.text) > 350:
            continue
        title, signer = _split_signature(block.text)
        has_authority = _has_authority_anchor(block.text)
        if has_authority and len(block.text) > 120:
            # A body paragraph can mention an authority mid-sentence
            # ("... giao cho Hội đồng ... Ban Tổng Giám đốc Công ty thực
            # hiện ..."): only a block LED by the anchor is a signature.
            anchor_pos = min([match.start() for match in _AUTHORITY_RE.finditer(block.text)] or [0])
            if anchor_pos > 40:
                has_authority = False
                title, signer = "", ""
        visual_signature = (
            block.type in {"image", "seal"}
            and signer
            and (block.x1 - block.x0) >= 0.15
            and block.x0 < 0.88
            # All-uppercase suffixes cut out of a company seal are commonly
            # organization-name fragments, not the signer's printed name.
            and not (_ORG_RE.search(block.text) and signer.isupper())
            # Seal-stamp remnants ("ONG HOA VII" clipped from "CỘNG HÒA...") are
            # all-caps fragments built from seal vocabulary; a printed signer
            # name either keeps Title case or avoids that vocabulary.
            and not _is_seal_name_fragment(signer)
        )
        # A person-like suffix in an ordinary body paragraph is not enough:
        # require an authority/title anchor, or visual signature/seal layout.
        if has_authority or visual_signature:
            candidates.append((block, title, signer))
    return candidates

_SEAL_NAME_TOKENS = ("doc lap", "xa hoi", "viet nam", "chung thuc", "sao y")

_SEAL_ROMAN_TOKENS = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}

def _is_seal_name_fragment(value: str) -> bool:
    if any(ch.islower() for ch in value):
        return False
    folded_words = _fold(value).split()
    folded = " ".join(folded_words)
    if any(token in folded for token in _SEAL_NAME_TOKENS):
        return True
    if "hoa" in folded_words and (
        any(word in _SEAL_ROMAN_TOKENS or any(ch.isdigit() for ch in word) for word in folded_words)
        or "cong" in folded_words
    ):
        return True
    return False

def _closing_page(pages: list[list[_Block]], start: int, limit: int) -> int:
    for page_index in range(start, limit + 1):
        blocks = pages[page_index]
        # The "Nơi nhận" anchor itself is a strong closing signal.  No lower
        # vertical bound: crop-normalized coordinates shift with the detected
        # margins (e.g. a stamp-strip crop maps the closing block above 0.45
        # in original-page space).  The left-column bound stays to avoid
        # right-column quotations.
        recipients = [block for block in blocks if _RECIPIENT_RE.search(block.text) and _zone(block, x1=0.62)]
        signatures = _signature_candidates(blocks)
        right_seal = any(block.type in {"seal", "image"} and _zone(block, x0=0.40, y0=0.40) for block in blocks)
        strong_signature = any(
            title and signer and (not block.geometry_reliable or block.y0 >= 0.40 or block.type in {"image", "seal"})
            for block, title, signer in signatures
        )
        if (recipients and (signatures or right_seal)) or strong_signature:
            return page_index
        # ``Nơi nhận`` is itself a strong closing anchor when it is in the
        # left column; many scans do not OCR the signature reliably.
        if recipients:
            return page_index
    return limit

def _evidence(block: _Block, rule: str, value: str) -> dict[str, Any]:
    return {
        "page": block.page + 1,
        "block": block.index + 1,
        "block_id": block.index + 1,
        "block_type": block.type,
        "source_layout_id": block.source_layout_id,
        "rule": rule,
        "value": value,
        "source_text": block.text,
        "bbox": [round(v, 1) for v in block.bbox_pixels],
        "bbox_normalized": [round(v, 4) for v in block.bbox],
        "geometry_reliable": block.geometry_reliable,
        "source_line_ids": [item.get("source_line_order") for item in block.source_lines],
        "source_lines": [{"text": item.get("text"), "bbox": item.get("bbox"),
                          "recognition_confidence": item.get("recognition_confidence"),
                          "detection_confidence": item.get("detection_confidence")}
                         for item in block.source_lines],
        "confidence_note": "Recognizer scores are uncalibrated; block evidence may span multiple lines.",
    }

def _is_address_or_noise(text: str) -> bool:
    folded = _fold(text)
    if any(token in folded for token in ("dia chi", "tru so", "dien thoai", "website", "email", "fax", "dt:", "tel:")):
        return True
    # Street / ward / house number patterns indicating postal addresses
    if re.search(r"\b(?:đường|duong|ngõ|ngo|ngách|ngach|hẻm|hem|ấp|ap|khu phố|khu pho)\b", text, re.IGNORECASE):
        return True
    # "Số <digits>" followed by comma or address words (e.g. "Số 1253, CMT8" or "Số 10 đường")
    if re.search(r"(?:s[oố]\s+\d+\s*,|s[oố]\s+\d+\s+(?:đường|duong|phố|pho))\b", text, re.IGNORECASE):
        return True
    return False

def _is_valid_code_value(raw: str) -> bool:
    val = raw.strip(" ;,:")
    if not val or len(val) < 2:
        return False
    if "," in val:
        return False
    # All statutory Vietnamese administrative codes contain a slash '/'
    # (e.g. 12/QĐ-UBND, 03/NQ-ĐHCĐ, 4977/BVHTTDL-VP, /TB-QLNY, 36/2026/QH16)
    if "/" in val:
        return True
    # If no slash, must contain digits AND uppercase letters, not just plain numbers
    has_alpha = bool(re.search(r"[A-Za-zÀ-ỸĐđ]", val))
    has_digit = bool(re.search(r"\d", val))
    return has_alpha and has_digit

@dataclass(frozen=True)
class _HeaderFrame:
    """Default header geometry anchored on the Quốc hiệu/Tiêu ngữ block.

    The country header (top-right) is the reference milestone: the issuing
    authority sits to its LEFT, the code sits BELOW the authority (still left
    of the country), and the place/date line sits to the RIGHT under the
    country.  All relations are relative overlaps, never absolute cutoffs, so
    crops and narrow/wide layouts keep working.

    This is the DEFAULT flow.  Documents without a country header (corporate
    letterheads, foreign forms) get ``country=None`` and callers fall back to
    the absolute-zone behavior — that is the edge-case path, not a failure.
    """

    country: _Block | None
    office: _Block | None

def _find_country_block(blocks: list[_Block]) -> _Block | None:
    candidates = [
        block
        for block in blocks
        if _zone(block, y1=0.22)
        and (
            "cong hoa xa hoi chu nghia viet nam" in block.folded
            or "doc lap - tu do - hanh phuc" in block.folded
            or "doc lap tu do hanh phuc" in block.folded
        )
    ]
    if not candidates:
        return None
    # Rightmost country block is the true right column; a merged
    # "office + country" block further left is handled by the office fallback.
    return max(candidates, key=lambda block: (block.x0, block.y0))

def _header_frame(blocks: list[_Block]) -> _HeaderFrame:
    country = _find_country_block(blocks)
    office: _Block | None = None
    if country is not None:
        left_orgs = [
            block
            for block in blocks
            if _zone(block, y1=0.30)
            and _has_org_keyword(block.text, block.folded)
            and (not block.geometry_reliable or block.x1 <= country.x0 + 0.05)
            and (not block.geometry_reliable or block.y1 >= 0.04)
            and not _is_annotation_header(block.text, block.folded)
            and "cong bao" not in block.folded
            and re.match(r"^\s*mau\s+\d+", block.folded) is None
        ]
        if left_orgs:
            office = min(left_orgs, key=lambda block: (block.y0, block.x0))
        elif _has_org_keyword(country.text, country.folded):
            # Merged "office + country" single block: the authority is the
            # country block's left part (split later by _clean_office_sender_text).
            office = country
    return _HeaderFrame(country=country, office=office)

def _code_match_reaches_line_end(block_text: str, match_end: int) -> bool:
    """True when nothing but whitespace/punctuation follows the code value.

    Statutory codes close their line ("Số: 8871/VPCP-CN" and nothing after);
    body citations ("... số 81/2013/NĐ-CP ngày 19 tháng 7 ...") continue.
    """
    return re.match(r"[\s;:,.]*(\n|$)", block_text[match_end:]) is not None

# ---- rule cell 15 (verbatim tu 6_1_optimize_rule.ipynb) ----
def _extract_code(
    blocks: list[_Block],
    evidence: dict[str, list[dict[str, Any]]],
    *,
    office_block: _Block | None = None,
    date_block_hint: _Block | None = None,
    country_block: _Block | None = None,
) -> tuple[str, _Block | None]:
    # QH-anchored flow: the caller resolves the header frame ONCE
    # (country -> office -> date/province -> code) and passes the anchors in.
    # Direct calls without anchors (unit tests) fall back to local resolution.
    if office_block is None or country_block is None:
        frame = _header_frame(blocks)
        if country_block is None:
            country_block = frame.country
        if office_block is None:
            office_block = frame.office
    country_block = country_block
    if office_block is None:
        # Edge-case path: legacy absolute-zone office resolution.
        office_blocks = [
            block
            for block in blocks
            if _zone(block, y1=0.30)
            and _has_org_keyword(block.text, block.folded)
            and (not block.geometry_reliable or block.cx <= 0.52)
            and (not block.geometry_reliable or country_block is None or block.cx < country_block.cx)
            and "cong bao" not in block.folded
            and re.match(r"^\s*mau\s+\d+", block.folded) is None
            and not _is_annotation_header(block.text, block.folded)
        ]
        office_block = min(office_blocks, key=lambda block: (block.y0, block.x0)) if office_blocks else None

    date_block = date_block_hint
    if date_block is None:
        date_blocks = [b for b in blocks if _zone(b, x0=0.40, y1=0.40) and "ngay" in b.folded]
        date_block = date_blocks[0] if date_blocks else None

    # Filter candidate blocks by relative region and exclusion rules
    candidates: list[_Block] = []
    for block in blocks:
        if block.geometry_reliable:
            # Code region: upper portion of page, left/center column
            if block.cx > 0.60 or block.y0 > 0.35:
                continue
            if country_block and block.x0 >= country_block.cx:
                continue
            if date_block and block.x0 >= date_block.x0 + 0.15:
                continue
        # Exclusion rules: ignore form templates, citations, body articles,
        # cong bao, addresses.  A "Mẫu 08/..." template block carries a cited
        # decision serial ("... số 600190-SGDHN ..."), never the document code.
        if "cong bao" in block.folded or block.folded.startswith("can cu") or "căn cứ" in block.text.lower():
            continue
        if re.match(r"^\s*mau\s+\d+", block.folded):
            continue
        if (
            block.folded.startswith("xet ")
            or block.folded.startswith("dieu ")
            or re.search(r"\bdieu\s+\d+", block.folded)
        ):
            continue
        match = _CODE_RE.search(block.text)
        if _is_address_or_noise(block.text):
            continue
        candidates.append(block)

    # Prefer candidates below the identified issuing-authority region. Keep
    # other header candidates as a fallback because OCR/layout may merge or
    # vertically overlap the authority and code blocks.
    candidates.sort(
        key=lambda block: (
            (
                0
                if office_block is None or block.index == office_block.index or block.y0 >= office_block.y1 - 0.04
                else 1
            ),
            block.y0,
            block.x0,
        )
    )

    # Pass 0: Consolidated-heading code without a "Số" anchor
    # (e.g. "VĂN BẢN HỢP NHẤT 05/2026/VBHN-TT-BTP" at the very top of the
    # header).  This must run before the generic "Số" passes: body citations
    # such as "Thông tư số 07/2022/..." would otherwise win and cascade into
    # wrong date/province skips.  Restricted to the top header strip so body
    # references can never match.
    consolidated_re = re.compile(
        r"VĂN\s+BẢN\s+HỢP\s+NHẤT\s*:?\s*" r"([0-9][0-9A-Za-zÀ-ỹĐđ./\-]*(?:\s*[0-9A-Za-zÀ-ỹĐđ./\-]+)*)",
        re.IGNORECASE,
    )
    for block in blocks:
        if block.geometry_reliable and (block.y0 > 0.20 or block.cx > 0.60):
            continue
        if "cong bao" in block.folded:
            continue
        if re.match(r"^\s*mau\s+\d+", block.folded):
            continue
        match = consolidated_re.search(block.text)
        if match:
            raw_val = match.group(1).strip()
            if _is_valid_code_value(raw_val):
                value = _normalise_code(raw_val)
                evidence["code"].append(_evidence(block, "consolidated-heading-code", value))
                return value, block

    # Pass 1: Primary regex: Line starts with Số / Số:
    # Collect every match, then prefer the value that closes its line:
    # statutory codes end their line while body citations continue.
    line_start_code_re = re.compile(
        r"(?:^|\n)\s*S[oố]\s*[:：.]?\s*([0-9A-Za-zÀ-ỹĐđ./\-]+(?:(?!\s*(?:ng[aà]y|v/v|k[ií]nh|n[oơ]i|c[oộ]ng))\s+[0-9A-Za-zÀ-ỹĐđ./\-]+)*)",
        re.IGNORECASE,
    )
    pass1: list[tuple[int, int, _Block, str]] = []
    for order, block in enumerate(candidates):
        match = line_start_code_re.search(block.text)
        if match:
            raw_val = match.group(1).strip()
            if _is_valid_code_value(raw_val):
                eol = 0 if _code_match_reaches_line_end(block.text, match.end()) else 1
                pass1.append((eol, order, block, raw_val))
    if pass1:
        pass1.sort(key=lambda item: (item[0], item[1]))
        _, _, block, raw_val = pass1[0]
        value = _normalise_code(raw_val)
        evidence["code"].append(_evidence(block, "number-anchor", value))
        return value, block

    # Pass 2: Inline document type code (e.g. QUỐC HỘI Nghị quyết số: 36/2026/QH16)
    pass2: list[tuple[int, int, _Block, str]] = []
    for block in blocks:
        if block.geometry_reliable and block.y0 > 0.42:
            continue
        if (
            "cong bao" in block.folded
            or re.match(r"^\s*mau\s+\d+", block.folded)
            or block.folded.startswith("can cu")
            or block.folded.startswith("xet ")
            or block.folded.startswith("v/v")
            or block.folded.startswith("ve viec")
        ):
            continue
        match = _INLINE_TYPE_CODE_RE.search(block.text)
        if match:
            raw_val = match.group(1).strip()
            if _is_valid_code_value(raw_val):
                eol = 0 if _code_match_reaches_line_end(block.text, match.end()) else 1
                pass2.append((eol, block.y0, block, raw_val))
    if pass2:
        pass2.sort(key=lambda item: (item[0], item[1]))
        _, _, block, raw_val = pass2[0]
        value = _normalise_code(raw_val)
        evidence["code"].append(_evidence(block, "document-type-number-anchor", value))
        return value, block

    # Pass 3: General Số regex inside candidates (not preceded by address/noise)
    pass3: list[tuple[int, int, _Block, str]] = []
    for order, block in enumerate(candidates):
        match = _CODE_RE.search(block.text)
        if match:
            prefix = block.text[: match.start()]
            if not _is_address_or_noise(prefix):
                raw_val = match.group(1).strip()
                if _is_valid_code_value(raw_val):
                    eol = 0 if _code_match_reaches_line_end(block.text, match.end()) else 1
                    pass3.append((eol, order, block, raw_val))
    if pass3:
        pass3.sort(key=lambda item: (item[0], item[1]))
        _, _, block, raw_val = pass3[0]
        value = _normalise_code(raw_val)
        evidence["code"].append(_evidence(block, "number-anchor", value))
        return value, block

    # Pass 4: Fallback for geometry_reliable=False
    if any(not b.geometry_reliable for b in blocks[:10]):
        for block in blocks[:10]:
            if "cong bao" in block.folded or block.folded.startswith("can cu") or block.folded.startswith("xet "):
                continue
            if re.match(r"^\s*mau\s+\d+", block.folded):
                continue
            if _is_address_or_noise(block.text):
                continue
            match = _CODE_RE.search(block.text) or _INLINE_TYPE_CODE_RE.search(block.text)
            if match:
                raw_val = match.group(1).strip()
                if _is_valid_code_value(raw_val):
                    value = _normalise_code(raw_val)
                    evidence["code"].append(_evidence(block, "fallback-text-code-anchor", value))
                    return value, block

    return "", None

def _extract_date_province(
    blocks: list[_Block],
    code_block: _Block | None,
    evidence: dict[str, list[dict[str, Any]]],
    *,
    country_block: _Block | None = None,
) -> tuple[str, str]:
    # QH-anchored flow: province + documentDate live RIGHT of / UNDER the
    # country header.  Without a country anchor (edge case) the absolute
    # right-column zone applies.  Province is resolved INDEPENDENTLY of the
    # date parse: a place prefix ("Hà Nam, ngày ...") yields province even
    # when OCR destroys the date itself.
    # documentDate only accepts the long "ngày ... tháng ... năm ..." form
    # (accented or not); a bare slash/dot date belongs to an arrival stamp,
    # never to the issuance line.
    scored: list[tuple[int, float, _Block, str, str]] = []
    place_hits: list[tuple[float, _Block, str]] = []
    for block in blocks:
        if block.geometry_reliable:
            if country_block is not None:
                if block.x1 < country_block.x0 - 0.10:
                    continue
                if block.y0 < country_block.y0 - 0.05 or block.y0 > country_block.y1 + 0.14:
                    continue
            elif block.cx < 0.56 or block.y0 > 0.24:
                continue
        if code_block and block.geometry_reliable and block.y0 < code_block.y0 - 0.04:
            continue
        # Body citations ("Căn cứ Luật ... ngày ...", "Xét ...", "Điều N")
        # carry calendar dates but are never the issuance line.
        if block.folded.startswith("can cu") or block.folded.startswith("xet "):
            continue
        if re.search(r"\bdieu\s+\d+", block.folded):
            continue
        if "ngay" not in block.folded:
            continue
        issuance_text = re.split(
            r"(?:CÔNG\s+VĂN|VĂN\s+BẢN)?\s*ĐẾN\b",
            block.text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        place_match = re.search(r"^\s*(.+?)\s*,\s*ng[aà]y\b", issuance_text, re.IGNORECASE)
        place = place_match.group(1).strip() if place_match else ""
        place_valid = bool(place and 1 <= len(place.split()) <= 6 and not _has_org_keyword(place, _fold(place)))
        value = ""
        for long_match in _LONG_DATE_RE.finditer(unicodedata.normalize("NFKC", issuance_text)):
            value = _canonical_date(*long_match.groups())
            if value:
                break
        if value:
            scored.append((0 if place_valid else 1, block.y0, block, value, place if place_valid else ""))
        if place_valid:
            place_hits.append((block.y0, block, place))
    value = ""
    date_block: _Block | None = None
    if scored:
        scored.sort(key=lambda item: (item[0], item[1]))
        _, _, date_block, value, _ = scored[0]
        evidence["documentDate"].append(_evidence(date_block, "issuance-date", value))
    province = ""
    province_block: _Block | None = None
    if place_hits:
        # Prefer the place on the chosen date line; otherwise earliest place.
        same_line = [hit for hit in place_hits if date_block is not None and hit[1].index == date_block.index]
        candidates = same_line or sorted(place_hits, key=lambda item: item[0])
        _, province_block, province = candidates[0]
        evidence["province"].append(_evidence(province_block, "place-before-date", province))
    # OCR can damage every digit/keyword in the issuance date while leaving
    # the leading place intact.  Keep province independently when a short
    # place-like prefix remains at the start of a top metadata block.
    if not province:
        for block in blocks:
            if block.geometry_reliable:
                if country_block is not None:
                    if block.x1 < country_block.x0 - 0.10:
                        continue
                    if block.y0 < country_block.y0 - 0.05 or block.y0 > country_block.y1 + 0.30:
                        continue
                elif block.y0 > 0.32:
                    continue
            if code_block and block.geometry_reliable and block.y0 < code_block.y0 - 0.04:
                continue
            # Postal-address lines ("Thành phố Sa Đéc, Tỉnh Đồng Tháp") also
            # match the "Place," shape but are addresses, never the province:
            # the issuance line always keeps its "ngày" keyword even when the
            # digits are unreadable.
            if _is_address_or_noise(block.text):
                continue
            if "ngay" not in block.folded:
                continue
            match = re.match(r"^\s*([A-ZÀ-ỸĐ][^,\n]{1,40})\s*,", block.text)
            if not match:
                continue
            candidate_place = match.group(1).strip()
            if any(token in _fold(candidate_place) for token in ("dia chi", "website", "fax", "dien thoai", "dt:")):
                continue
            if 1 <= len(candidate_place.split()) <= 6 and not _ORG_RE.search(candidate_place):
                province = candidate_place
                evidence["province"].append(_evidence(block, "place-prefix-date-unreadable", province))
                break
    return value, province

def _split_country_office(text: str) -> str:
    """Keep the part before the country header in a merged office block.

    Matched on folded text so OCR variants ("CỘNG HOÀ", "CỘNG HÒA") split the
    same way; the cut maps back by word position, preserving original spelling.
    """
    folded_words = _fold(text).split()
    for index in range(len(folded_words) - 3):
        if folded_words[index : index + 4] == ["cong", "hoa", "xa", "hoi"]:
            original_words = text.split()
            return " ".join(original_words[: min(index, len(original_words))])
    return text

def _clean_office_sender_text(text: str) -> str:
    # Layout may merge the issuing authority and the country header in one
    # block (e.g. "THỦ TƯỚNG CHÍNH PHỦ CỘNG HÒA XÃ HỘI..."); keep the part
    # before the country anchor.
    text = _split_country_office(text)
    # A single layout line can merge the authority with its postal address
    # ("CÔNG TY ... TN Số 1253, CMT8, ..."): cut the address tail.  A comma
    # after "Số <digits>" never occurs in a statutory code.
    text = re.split(r"\bS[oố]\s+\d+\s*,", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    # Strip any inline code anchor if merged in same block
    code_m = _CODE_RE.search(text)
    if code_m:
        text = text[: code_m.start()].strip()
    # Layout may merge the issuing authority with the first-recipient block.
    text = _FIRST_RECIPIENT_RE.split(text, maxsplit=1)[0].strip()
    # A subject line ("V/v: ...") riding in the same block is body content,
    # never part of the issuing authority.
    text = _VV_RE.split(text, maxsplit=1)[0].strip()
    # Strip any document type
    text = re.sub(
        r"\b(?:" + "|".join(re.escape(v) for v in _DOCUMENT_TYPES) + r")\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    # Strip address lines
    lines = []
    for line in text.splitlines():
        if _is_address_or_noise(line):
            break
        lines.append(line)
    cleaned = " ".join(" ".join(lines).split())
    # Certified-copy headers embed the authority between a "SAO Y" prefix and a
    # signing-time suffix; drop those wrappers while keeping the authority.
    cleaned = re.sub(
        r"^(?:sao\s+y[\s,.:;]*)+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.split(
        r"[,;]?\s*thời\s+gian\s+k[ýy]\s*[:：]?.*$",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return cleaned

def _extract_office_sender(
    blocks: list[_Block],
    code_block: _Block | None,
    evidence: dict[str, list[dict[str, Any]]],
    *,
    country_block: _Block | None = None,
) -> tuple[str, _Block | None]:
    # QH-anchored flow: the office lives LEFT of the country header and its
    # vertical band is anchored on the country (not on the code — the code is
    # resolved AFTER the office, so a wrong code must never mask the office).
    # ``code_block`` is kept for backward compatibility and ignored.
    if country_block is None:
        country_block = _header_frame(blocks).country
    country = country_block
    if country is not None and country.geometry_reliable:
        max_y = max(0.25, country.y1 + 0.10)
    else:
        max_y = 0.25

    def _left_of_country(block: _Block) -> bool:
        if not block.geometry_reliable or country is None:
            return True
        return block.x1 <= country.x0 + 0.05

    candidates = []
    for block in blocks:
        if block.geometry_reliable and block.y1 > max_y:
            continue
        if block.geometry_reliable:
            if country is None:
                # Edge case (no country header): absolute left-column bound.
                if block.cx > 0.52:
                    continue
            elif not _left_of_country(block):
                continue
        # Top-strip digital-signature annotations (portal name, signer, time)
        # sit entirely above the statutory letterhead; their mangled OCR can
        # still contain an "org keyword" (e.g. "... điện tử Chính phủ").
        if block.geometry_reliable and block.y1 < 0.04:
            continue
        # Form-template headers ("Mẫu 08/CBTT-SGDHN...") describe the paper
        # form, never the issuing authority.
        if re.match(r"^\s*mau\s+\d+", block.folded):
            continue
        # Country text is stripped inside _clean_office_sender_text, so merged
        # "office + country" blocks still contribute their office part while
        # pure country blocks clean to "" and are dropped below.
        if _is_annotation_header(block.text, block.folded):
            continue
        # Arrival-stamp blocks ("SỞ GIAO DỊCH ... HÌNH ĐẾN", "VĂN BẢN ĐẾN
        # Ngày: ...") name another institution, never the issuing authority.
        if "van ban den" in block.folded or "cong van den" in block.folded:
            continue
        if any(token in block.folded for token in ("ky boi", "cong bao")):
            continue
        if _has_org_keyword(block.text, block.folded):
            candidates.append(block)

    if candidates:
        # Sort candidates top-to-bottom by y0 to preserve reading order
        candidates = sorted(candidates, key=lambda b: (b.y0, b.x0))
        cluster: list[_Block] = []
        for c in candidates:
            if not cluster or (c.y0 - cluster[-1].y1 <= 0.06):
                cluster.append(c)
            else:
                break

        parts: list[tuple[str, _Block]] = []
        for b in cluster:
            cleaned = _clean_office_sender_text(b.text)
            if cleaned:
                parts.append((cleaned, b))

        if parts:
            val = " ".join(p[0] for p in parts)
            for _, b in parts:
                evidence["officeSender"].append(_evidence(b, "upper-left-organization", val))
            return val, parts[-1][1]

    # "Tên công ty:" is the authoritative corporate letterhead anchor: prefer
    # it over a loose centered-org guess (arrival stamps name other
    # institutions, e.g. "SỞ GIAO DỊCH ... (HNX)", and would otherwise win by
    # reading order).
    for block in blocks:
        match = re.search(r"\bT[eê]n\s+c[oô]ng\s+ty\s*[:：]\s*(.+)$", block.text, re.IGNORECASE)
        if match:
            raw_val = match.group(1).strip()
            clean_val = " ".join(raw_val.split())
            evidence["officeSender"].append(_evidence(block, "company-name-anchor", clean_val))
            return clean_val, block

    # Corporate letterheads commonly center the issuing company rather than using statutory left column
    for block in blocks:
        if block.geometry_reliable and block.y1 > max_y:
            continue
        if block.geometry_reliable and block.y1 < 0.04:
            continue
        if re.match(r"^\s*mau\s+\d+", block.folded):
            continue
        if _is_annotation_header(block.text, block.folded):
            continue
        if any(token in block.folded for token in ("ky boi", "dia chi", "cong bao", "van ban den", "cong van den")):
            continue
        if re.search(r"(?:^|\W)den(?:$|\W)", block.folded):
            continue
        if _has_org_keyword(block.text, block.folded):
            cleaned = _clean_office_sender_text(block.text)
            if cleaned:
                evidence["officeSender"].append(_evidence(block, "upper-left-organization", cleaned))
                return cleaned, block

    return "", None

def _first_content_heading(
    blocks: list[_Block],
    after_index: int,
    evidence: dict[str, list[dict[str, Any]]],
) -> str:
    """Title of a consolidated (VBHN) document from its instrument heading.

    Returns the tail (or continuation) of the first non-VBHN named heading
    below the VBHN label, e.g. "Quy định ..." after "NGHỊ ĐỊNH".
    """
    for block_index in range(after_index + 1, len(blocks)):
        block = blocks[block_index]
        if not _zone(block, y0=0.08, y1=0.60):
            continue
        matched = _match_document_type(block)
        if not matched:
            continue
        _, tail, _ = matched
        if re.fullmatch(r"s[oố]\s*[:.]?\s*[0-9A-Za-zÀ-ỹĐđ./\-\s]+", tail, re.IGNORECASE):
            tail = ""
        title = " ".join(tail.split())
        if title:
            evidence["title"].append(_evidence(block, "content-heading-title", title))
            return title
        following: list[str] = []
        for candidate in blocks[block_index + 1 : block_index + 5]:
            if _FIRST_RECIPIENT_RE.search(candidate.text):
                break
            if candidate.y0 - block.y1 > 0.14:
                break
            if re.fullmatch(r"\d+", candidate.text.strip()):
                continue
            if any(anchor in candidate.folded for anchor in ("van ban den", "cong van den")):
                continue
            if "cong hoa xa hoi chu nghia viet nam" in candidate.folded:
                continue
            if re.match(r"^\s*S[oố]\s*[:：.]", candidate.text):
                continue
            if len(candidate.text) < 80 and _find_date(candidate.text):
                continue
            if candidate.folded.startswith("can cu"):
                break
            if candidate.type in {"doc_title", "paragraph_title", "text"} and _zone(
                candidate, x0=0.16, x1=0.88, y1=0.62
            ):
                following.append(candidate.text)
                evidence["title"].append(_evidence(candidate, "content-heading-continuation", candidate.text))
        if following:
            return " ".join(" ".join(following).split())
        return ""
    return ""

def _clean_type_tail(tail: str) -> str:
    """Strip arrival-stamp fragments from the same-line title tail.

    Layout merges the heading with stamp lines ("NGHỊ QUYẾT Ngày: 27 04-
    2017 ĐẠI HỘI... 2017 10247 Số:"): a colon-led "Ngày:" stamp prefix, a
    serial glued to a trailing "Số:" stub, and a letter-less "Số:" stub are
    stamp noise, never title text.  The colon requirement keeps legitimate
    prose dates ("... ngày 30/4 ...") intact.
    """
    cleaned = re.sub(r"\bNg[aà]y\s*[:：]\s*[\d\s/\-.]+", " ", tail, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d+\s+S[oố]\s*[:：]?\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bS[oố]\s*[:：]?\s*[^A-Za-zÀ-ỹĐđ]*$", "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())

def _extract_type_title(blocks: list[_Block], evidence: dict[str, list[dict[str, Any]]]) -> tuple[str, str]:
    # A named heading in the header zone is the authoritative document type,
    # even when a "V/v" subject line appears earlier on the page (corporate
    # letterheads put the subject above the heading).  ``V/v`` only identifies
    # a Công văn when no named heading exists.
    # Exception: when the subject rides inside the code line itself
    # ("Số: 270/HĐ V/v: ..."), the document is a Công văn whose body may quote
    # a named instrument ("Nghị quyết Đại hội..."); a later line-start type
    # word is then quoted content, not the heading.
    subject_in_code_line = any(
        _CODE_RE.search(block.text) and _VV_RE.search(block.text) for block in blocks if _zone(block, y0=0.08, y1=0.52)
    )
    if not subject_in_code_line:
        def heading_rank(index_block):
            _, candidate = index_block
            matched = _match_document_type(candidate)
            if not matched:
                return 2
            kind, _, line = matched
            raw_line = candidate.text.splitlines()[line].strip(" :.-–—")
            return 0 if _fold(raw_line) == _fold(kind) else 1
        for block_index, block in sorted(enumerate(blocks), key=heading_rank):
            if not _zone(block, y0=0.08, y1=0.52):
                continue
            matched = _match_document_type(block)
            if not matched:
                continue
            document_type, tail, line_index = matched
            raw_heading_line = block.text.splitlines()[line_index].strip(" :.-–—")
            standalone_heading = _fold(raw_heading_line) == _fold(document_type)
            earlier_subject = any(_VV_RE.search(candidate.text) for candidate in blocks
                                  if candidate.y0 < block.y0 and _zone(candidate, y1=0.52))
            if earlier_subject and not standalone_heading:
                continue  # Wrapped body citations after V/v are not named headings.
            # A "type word" starting a wrapped line of a V/v subject sentence
            # (e.g. "V/v: ... thông tin\nNghị quyết Đại hội...") is a subject
            # continuation, not the document heading: the heading never shares
            # its block with an earlier V/v anchor.
            if _VV_RE.search("\n".join(block.text.splitlines()[:line_index])):
                continue
            if _fold(document_type) == "van ban hop nhat":
                # Edge case: the VBHN label is the container, not the subject.
                # The title belongs to the consolidated instrument's own
                # heading (the first non-VBHN named heading below).
                content = _first_content_heading(blocks, block_index, evidence)
                evidence["type"].append(_evidence(block, "named-document-type", document_type))
                return document_type, content
            if re.fullmatch(r"s[oố]\s*[:.]?\s*[0-9A-Za-zÀ-ỹĐđ./\-\s]+", tail, re.IGNORECASE):
                tail = ""
            title = _clean_type_tail(tail)
            title_evidence: list[tuple[_Block, str, str]] = []
            if title:
                title = " ".join(title.split())
                title_evidence.append((block, "text-after-document-type", title))
            if not title:
                following: list[str] = []
                for candidate in blocks[max(0, block_index - 2) : block_index]:
                    vv_match = _VV_RE.search(candidate.text)
                    if vv_match:
                        candidate_value = candidate.text[vv_match.start() :].strip()
                        following.append(candidate_value)
                        title_evidence.append((candidate, "v-v-title-before-document-type", candidate_value))
                for candidate in blocks[block_index + 1 : block_index + 5]:
                    if _FIRST_RECIPIENT_RE.search(candidate.text):
                        break
                    if candidate.y0 - block.y1 > 0.14:
                        break
                    if re.fullmatch(r"\d+", candidate.text.strip()):
                        continue
                    if any(anchor in candidate.folded for anchor in ("van ban den", "cong van den")):
                        continue
                    # The country header, the code line and the place/date line
                    # sit between the heading and the body; none of them is a
                    # title continuation.
                    if "cong hoa xa hoi chu nghia viet nam" in candidate.folded:
                        continue
                    if re.match(r"^\s*S[oố]\s*[:：.]", candidate.text):
                        continue
                    # A date-led line ("Ngày: 27 04- 2017 ...") is stamp/header
                    # metadata, never a title continuation — even when OCR
                    # damage makes the date itself unparseable.
                    if re.match(r"^\s*Ng[aà]y\s*[:：]", candidate.text, re.IGNORECASE):
                        continue
                    # Short date lines ("Hà Nội, ngày ...") are metadata, never
                    # title text; long sentences merely mentioning a date keep
                    # flowing into the title.
                    if len(candidate.text) < 80 and _find_date(candidate.text):
                        continue
                    # Citation lines ("Căn cứ Luật...") follow the heading in
                    # statutory documents and mark the start of the body;
                    # they are never the title, and neither is anything below.
                    if candidate.folded.startswith("can cu"):
                        break
                    if candidate.type in {"doc_title", "paragraph_title", "text"} and _zone(
                        candidate, x0=0.16, x1=0.88, y1=0.58
                    ):
                        following.append(candidate.text)
                        title_evidence.append((candidate, "title-continuation", candidate.text))
                title = "\n".join(following).strip()
            # API output joins a wrapped title with a single space (contract
            # 3.1), whichever path produced it; raw breaks stay in evidence.
            title = " ".join(title.split())
            evidence["type"].append(_evidence(block, "named-document-type", document_type))
            for title_block, rule, matched_value in title_evidence:
                evidence["title"].append(_evidence(title_block, rule, matched_value))
            return document_type, title

    for block in blocks:
        if not _zone(block, y0=0.08, y1=0.52):
            continue
        match = _VV_RE.search(block.text)
        if match:
            title = block.text[match.start() :].strip()
            evidence["type"].append(_evidence(block, "v-v-structure", "Công văn"))
            evidence["title"].append(_evidence(block, "v-v-title-preserved", title))
            return "Công văn", title
    return "", ""

# ---- rule cell 17 (verbatim tu 6_1_optimize_rule.ipynb) ----
def _clean_recipient_value(value: str) -> str:
    value = _SAVE_RE.split(value, maxsplit=1)[0]
    value = _SAVE_SUFFIX_RE.sub("", value)
    value = value.replace("\n", "; ")
    # Preserve punctuation inside values; item boundaries come from anchors/bullets.
    value = _BULLET_RE.sub("; ", value)
    value = re.sub(r"\s*;\s*", "; ", value)
    value = re.sub(r"(?:;\s*)+", "; ", value)
    value = re.sub(r"^[.;,:\s]+", "", value)
    value = re.sub(r"[;:\s]+$", "", value)
    return value.strip(" ;")

def _extract_anchor_value(
    blocks: list[_Block],
    regex: re.Pattern[str],
    field_name: str,
    rule: str,
    evidence: dict[str, list[dict[str, Any]]],
) -> str:
    for block_index, block in enumerate(blocks):
        match = regex.search(block.text)
        if not match:
            continue
        if field_name == "first_recipients" and any(
            anchor in block.folded for anchor in ("van ban den", "cong van den")
        ):
            # A "Kính ..." fragment inside an arrival-stamp paragraph is stamp
            # noise — UNLESS the anchor is immediately followed by a real
            # addressee on the same block (the stamp overlaps the Kính gửi
            # line instead of replacing it).
            remainder = block.text[match.end() :].strip()
            if not remainder or re.match(r"^(ngay|so)\s*[:：]", remainder, re.IGNORECASE):
                continue
        value = block.text[match.end() :]
        if field_name == "first_recipients":
            value = re.sub(
                r"\bNg[aà]y\s*[:：]\s*\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{4}",
                " ",
                value,
                flags=re.IGNORECASE,
            )
            # A trailing arrival-stamp stub ("Số:................ A03/",
            # "Số: 10608") is stamp noise, never an addressee: cut at "Số:"
            # when no real word (3+ letters) follows it.  A genuine reference
            # ("... Số: 123/BQP") keeps its code and survives.
            stub = re.search(r"\bS[oố]\s*[:：]", value, flags=re.IGNORECASE)
            if stub and not any(
                sum(ch.isalpha() for ch in word) >= 3 for word in value[stub.end() :].split()
            ):
                value = value[: stub.start()]
        if field_name == "recipients" and (not value.strip() or value.strip().isdigit()) and block.geometry_reliable:
            # Layout can split the label and each bullet into independent blocks.
            # Follow the same column by geometry, not layout reading-order indices.
            parts = []
            previous_y = block.y1
            for candidate in sorted(blocks, key=lambda item: (item.y0, item.x0)):
                if candidate.index == block.index or candidate.y0 < block.y0:
                    continue
                if abs(candidate.x0 - block.x0) > 0.06 or candidate.cx > 0.62:
                    continue
                if candidate.y0 - previous_y > 0.045:
                    break
                if re.match(r"^\s*[-–—•]?\s*L[uư]u\b", candidate.text, re.IGNORECASE):
                    break
                if not parts and not re.match(r"^\s*[-–—•]", candidate.text):
                    break
                if _RECIPIENT_RE.search(candidate.text):
                    break
                parts.append(candidate.text)
                previous_y = candidate.y1
                evidence[field_name].append(_evidence(candidate, "recipient-column-continuation", candidate.text))
            if parts:
                value = "\n".join(parts)
        value = _clean_recipient_value(value)
        continuation_block: _Block | None = None
        continuation_value = ""
        if field_name == "first_recipients" and value and ";" not in value:
            for candidate in blocks[block_index + 1 : block_index + 5]:
                if candidate.geometry_reliable and candidate.y0 > block.y1 + 0.06:
                    break
                if any(anchor in candidate.folded for anchor in ("van ban den", "cong van den")):
                    continue
                if len(candidate.text) > 160 or re.search(
                    r"\b(?:tru so|ten cong ty|ten to chuc|dia chi|noi dung|" r"can cu|thuc hien|dieu\s+\d+)\b",
                    candidate.folded,
                ):
                    break
                # Stamp serial fragments ("Số: 10608") are not organizations,
                # even though folded "số" collides with the "sở" keyword.
                if re.match(r"^\s*S[oố]\s*[:：]", candidate.text, re.IGNORECASE):
                    break
                if _has_org_keyword(candidate.text, candidate.folded):
                    continuation = _clean_recipient_value(candidate.text)
                    if continuation:
                        value = f"{value}; {continuation}"
                        continuation_block = candidate
                        continuation_value = continuation
                    break
        if value:
            evidence[field_name].append(_evidence(block, rule, value))
            if continuation_block is not None:
                evidence[field_name].append(_evidence(continuation_block, f"{rule}-continuation", continuation_value))
            return value
    return ""

def _extract_levels(blocks: list[_Block], evidence: dict[str, list[dict[str, Any]]]) -> tuple[str, str]:
    candidates = {"priority_level": [], "security_level": []}
    for block in blocks:
        if block.geometry_reliable and (block.cx > 0.30 or block.y0 < 0.08 or block.y1 > 0.48):
            continue
        if len(block.text) > 80:
            continue
        for field, labels in RULE_LEXICON["levels"].items():
            hits = []
            for label, aliases in labels.items():
                for alias in aliases:
                    folded_alias = _fold(alias)
                    if re.search(r"(?<!\w)" + re.escape(folded_alias) + r"(?!\w)", block.folded):
                        hits.append((len(folded_alias), label))
            if field == "priority_level" and not hits and re.search(
                r"(?<!\w)\w?hoa\s+toc(?!\w)", block.folded
            ):
                # One spurious leading OCR character, scoped to this compact stamp zone.
                hits.append((len("hoa toc"), "1_HỎA TỐC"))
            if hits:
                # Match longest phrase: THƯỢNG KHẨN must not become KHẨN.
                _, label = max(hits)
                candidates[field].append((block, label))
                evidence[field].append(_evidence(block, field + "-stamp", label))
    values = {}
    for field, hits in candidates.items():
        labels = {label for _, label in hits}
        # API-compatible default on unresolved conflicts; debug explicitly marks ambiguity.
        values[field] = next(iter(labels)) if len(labels) == 1 else "0_BÌNH THƯỜNG"
    return values["priority_level"], values["security_level"]


def _extract_receiver_date(blocks: list[_Block], evidence: dict[str, list[dict[str, Any]]]) -> str:
    for block in blocks:
        has_den = bool(re.search(r"(?:^|\W)den(?:$|\W)", block.folded))
        has_stamp_phrase = any(anchor in block.folded for anchor in ("cong van den", "van ban den"))
        has_stamp_fields = has_den and "ngay:" in block.folded and len(block.text) <= 180
        starts_with_den = bool(re.match(r"^\s*ĐẾN\b", block.text, re.IGNORECASE)) and len(block.text) <= 80
        if not (has_stamp_phrase or has_stamp_fields or starts_with_den):
            continue
        if len(block.text) > 220 and not any(anchor in block.folded for anchor in ("cong van den", "van ban den")):
            continue
        value = _find_date(block.text)
        if "ngay" in block.folded:
            if value:
                evidence["receiverDate"].append(_evidence(block, "arrival-stamp-date", value))
                return value
            # The stamp has its own date slot but OCR did not produce a valid
            # calendar value.  Do not substitute a nearby body/issuance date.
            continue
        # Arrival stamp text is often split into adjacent layout blocks.
        nearby_blocks = [
            candidate
            for candidate in blocks
            if candidate.index != block.index
            and re.match(r"^\s*ngay\s*[:：]", candidate.folded)
            and len(candidate.text) <= 80
            and candidate.x0 <= block.x1 + 0.12
            and candidate.x1 >= block.x0 - 0.12
            and candidate.y0 <= block.y1 + 0.14
            and candidate.y1 >= block.y0 - 0.14
        ]
        joined = "\n".join([block.text, *(candidate.text for candidate in nearby_blocks)])
        if "ngay" in _fold(joined):
            value = _find_date(joined)
            if value:
                evidence["receiverDate"].append(_evidence(block, "split-arrival-stamp-date", value))
                for candidate in nearby_blocks:
                    evidence["receiverDate"].append(_evidence(candidate, "split-arrival-stamp-date-part", value))
                return value
    # Arrival stamps often wrap across two layout blocks: the previous block
    # ends with "Đến" and the next block holds "Ngày: <date>" (the issuance
    # line and the stamp share the header strip, so reading order links them).
    for index, block in enumerate(blocks):
        if index == 0 or "ngay" not in block.folded:
            continue
        if block.geometry_reliable and (block.y0 > 0.35 or block.x0 < 0.40):
            continue
        value = _find_date(block.text)
        if not value:
            continue
        previous = blocks[index - 1]
        if previous.folded.rstrip(" :.-").endswith("den"):
            evidence["receiverDate"].append(_evidence(block, "wrapped-arrival-stamp-date", value))
            evidence["receiverDate"].append(_evidence(previous, "wrapped-arrival-stamp-den-part", value))
            return value
    return ""

def _extract_signature(blocks: list[_Block], evidence: dict[str, list[dict[str, Any]]]) -> tuple[str, str]:
    titles: list[str] = []
    signers: list[str] = []
    candidates = sorted(_signature_candidates(blocks), key=lambda item: (item[0].y0, item[0].x0))
    for block, title, signer in candidates:
        clean_title = " ".join(title.split()).strip() if title else ""
        if clean_title and not signer:
            # Authority wrapped onto a second block ("TM. ĐOÀN CHỦ TỊCH" /
            # "CHỦ TỌA"): the following, column-aligned short block continues
            # the SAME title with a space (contract 3.1), unless it is a
            # person name, another anchor, or list content.  Recipient bullets
            # interleaved between the two lines are stepped over.
            follower = None
            try:
                position = blocks.index(block) + 1
            except ValueError:
                position = None
            if position is not None:
                for candidate_block in blocks[position : position + 4]:
                    candidate_text = " ".join(candidate_block.text.split()).strip()
                    if re.match(r"^\s*[-–—•]", candidate_text) or re.match(r"^\s*l[uư]u\b", candidate_block.folded):
                        continue
                    follower = candidate_block
                    break
            if follower is not None and follower is not block:
                follower_text = " ".join(follower.text.split()).strip()
                follower_words = set(follower.folded.split())
                if (
                    follower_text
                    and len(follower_text) <= 60
                    # A wrapped title line still speaks the authority's
                    # language: stamp serials ("0010843") and OCR crumbs
                    # ("SK") carry no title word and must not join.
                    and (follower_words & _TITLE_WORDS_FOLDED)
                    and not _has_authority_anchor(follower_text)
                    and not _has_org_keyword(follower_text, follower.folded)
                    and not _looks_like_signature_name(follower_text)
                    and not _RECIPIENT_RE.search(follower_text)
                    and not re.match(r"^\s*[-–—•]", follower_text)
                    and not re.match(r"^\s*(?:luu|dieu\s+\d+)\b", follower.folded)
                    and (
                        not follower.geometry_reliable
                        or (follower.y0 <= block.y1 + 0.06 and abs(follower.cx - block.cx) <= 0.24)
                    )
                ):
                    clean_title = f"{clean_title} {follower_text}"
                    evidence["signer_title"].append(
                        _evidence(follower, "signature-authority-continuation", clean_title)
                    )
        if clean_title and clean_title not in titles:
            titles.append(clean_title)
            evidence["signer_title"].append(_evidence(block, "signature-authority", clean_title))
        if signer and signer not in signers:
            signers.append(signer)
            evidence["signer"].append(_evidence(block, "signature-name", signer))
        if title and not signer:
            # Prefer a clear mixed-case printed name over all-capital seal fragments.
            for name_block in sorted(blocks, key=lambda b: (not any(ch.islower() for ch in b.text), b.y0)):
                if name_block.type == "seal" or name_block.y0 < block.y0:
                    continue
                if name_block.geometry_reliable:
                    if name_block.y0 > block.y1 + 0.22 or abs(name_block.cx - block.cx) > 0.24:
                        continue
                if (
                    _looks_like_signature_name(name_block.text)
                    and not _is_seal_name_fragment(name_block.text.strip())
                    and name_block.text not in signers
                ):
                    signers.append(name_block.text)
                    evidence["signer"].append(_evidence(name_block, "name-below-signature-authority", name_block.text))
                    break
    if not signers:
        for block in sorted(blocks, key=lambda b: (-b.y0, b.x0)):
            if block.type in {"seal", "image"}:
                continue
            if block.geometry_reliable and (block.cx < 0.50 or block.y0 < 0.35):
                continue
            cleaned = block.text.strip()
            lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
            for candidate_line in reversed(lines):
                if (
                    _looks_like_signature_name(candidate_line)
                    and not _is_seal_name_fragment(candidate_line)
                    and candidate_line not in signers
                ):
                    signers.append(candidate_line)
                    evidence["signer"].append(_evidence(block, "fallback-signature-zone-name", candidate_line))
                    break
            if signers:
                break
    return "; ".join(titles), "; ".join(signers)

def _extract_cong_dien_recipients(
    blocks: list[_Block],
    evidence: dict[str, list[dict[str, Any]]],
) -> str:
    """Addressees of a CÔNG ĐIỆN: the distribution list after "[...] ĐIỆN:".

    Điện dispatches carry no "Kính gửi"; the addressee list follows the
    dispatch line ("THỦ TƯỚNG CHÍNH PHỦ ĐIỆN:") as bullet/continuation blocks.
    """
    for block_index, block in enumerate(blocks):
        if block.geometry_reliable and block.y0 > 0.55:
            continue
        dispatch = re.search(r"ĐIỆN\s*:\s*(.*)$", block.text, re.IGNORECASE)
        if not dispatch:
            continue
        parts = [dispatch.group(1).strip()] if dispatch.group(1).strip() else []
        anchor_block = block
        for candidate in blocks[block_index + 1 : block_index + 8]:
            if candidate.geometry_reliable and candidate.y0 > anchor_block.y1 + 0.20:
                break
            text = " ".join(candidate.text.split()).strip()
            if not text or len(text) > 220:
                break
            if _RECIPIENT_RE.search(text) or _FIRST_RECIPIENT_RE.search(text):
                break
            if re.search(r"\b(?:can cu|dieu\s+\d+|thuc hien)\b", candidate.folded):
                break
            parts.append(text)
            evidence["first_recipients"].append(_evidence(candidate, "dien-dispatch-continuation", text))
        value = _clean_recipient_value("; ".join(part for part in parts if part))
        if value:
            evidence["first_recipients"].append(_evidence(block, "dien-dispatch-anchor", value))
            return value
    return ""

# ---- rule cell 19 (verbatim tu 6_1_optimize_rule.ipynb) ----
def extract_vbhc(
    pages_blocks: list[list[PageBlock]],
    original_page_sizes: list[tuple[int, int]],
    processed_page_sizes: list[tuple[int, int]],
    crop_offsets: list[tuple[float, float]],
    *,
    geometry_reliable: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract the first VBHC document and return ``(prediction, debug)``.

    ``processing_time`` is initialized to zero; the caller owns the full
    pipeline timer and overwrites it immediately before returning/persisting
    the final response.
    """
    lengths = {len(pages_blocks), len(original_page_sizes), len(processed_page_sizes), len(crop_offsets)}
    if len(lengths) != 1:
        raise ValueError("pages_blocks, page sizes and crop_offsets must have equal lengths")

    evidence: dict[str, list[dict[str, Any]]] = {field: [] for field in TEXT_FIELDS}
    if not pages_blocks:
        values = {field: "" for field in TEXT_FIELDS}
        values["priority_level"] = "0_BÌNH THƯỜNG"
        values["security_level"] = "0_BÌNH THƯỜNG"
        information = {field: _field(values[field]) for field in TEXT_FIELDS}
        return {"information": [information], "processing_time": 0.0}, {
            "document_start_page": None,
            "document_end_page": None,
            "geometry_reliable": geometry_reliable,
            "evidence": evidence,
        }

    pages = _make_blocks(
        pages_blocks,
        original_page_sizes,
        processed_page_sizes,
        crop_offsets,
        geometry_reliable,
    )
    start = _document_start(pages)
    next_start = _next_document_start(pages, start)
    search_limit = (next_start - 1) if next_start is not None else len(pages) - 1
    end = _closing_page(pages, start, search_limit)

    header_blocks = pages[start]
    closing_blocks = pages[end]
    # QH-anchored header flow (contract 3.3): resolve the frame once, then
    # office LEFT of country -> province+date RIGHT/UNDER country ->
    # code BELOW office and LEFT of the date.  A wrong code must never mask
    # the office, so the office no longer depends on the code.
    frame = _header_frame(header_blocks)
    office_sender, office_block = _extract_office_sender(header_blocks, None, evidence, country_block=frame.country)
    document_date, province = _extract_date_province(header_blocks, None, evidence, country_block=frame.country)
    date_hint = None
    # Re-resolve the issuance block as a code anchor when present.
    if evidence["documentDate"]:
        date_page = evidence["documentDate"][0]["page"]
        date_bid = evidence["documentDate"][0]["block"]
        for block in header_blocks:
            if block.page + 1 == date_page and block.index + 1 == date_bid:
                date_hint = block
                break
    code, code_block = _extract_code(
        header_blocks,
        evidence,
        office_block=office_block,
        date_block_hint=date_hint,
        country_block=frame.country,
    )
    document_type, title = _extract_type_title(header_blocks, evidence)
    first_recipients = _extract_anchor_value(
        header_blocks, _FIRST_RECIPIENT_RE, "first_recipients", "kinh-gui-anchor", evidence
    )
    if not first_recipients and _fold(document_type) == "cong dien":
        first_recipients = _extract_cong_dien_recipients(header_blocks, evidence)
    recipients = _extract_anchor_value(closing_blocks, _RECIPIENT_RE, "recipients", "noi-nhan-anchor", evidence)
    signer_title, signer = _extract_signature(closing_blocks, evidence)
    priority, security = _extract_levels(header_blocks, evidence)
    receiver_date = _extract_receiver_date(header_blocks, evidence)

    values = {
        "type": document_type,
        "title": title,
        "code": code,
        "documentDate": document_date,
        "officeSender": office_sender,
        "recipients": recipients,
        "signer": signer,
        "priority_level": priority,
        "security_level": security,
        "first_recipients": first_recipients,
        "signer_title": signer_title,
        "province": province,
        "receiverDate": receiver_date,
    }
    information = {field: _field(values[field]) for field in TEXT_FIELDS}
    prediction = {"information": [information], "processing_time": 0.0}
    debug = {
        "document_start_page": start + 1,
        "document_end_page": end + 1,
        "next_document_start_page": next_start + 1 if next_start is not None else None,
        "geometry_reliable": geometry_reliable,
        "evidence": evidence,
        "field_status": {
            field: ("ambiguous" if field in {"priority_level", "security_level"}
                    and len({item["value"] for item in evidence[field]}) > 1
                    else "found" if evidence[field] else "not_found")
            for field in TEXT_FIELDS
        },
        "rule_version": RULE_VERSION,
        "lexicon_candidates": {
            field: lookup_entity_candidates(values[field], field, DEVELOPMENT_LEXICON)
            for field in ("officeSender", "province")
        },
    }
    return prediction, debug

__all__ = ["extract_vbhc"]

# ---- rule cell 21 (verbatim tu 6_1_optimize_rule.ipynb) ----
LABEL_MAP = {
    "Text": "text", "SectionHeader": "paragraph_title", "Title": "doc_title",
    "PageHeader": "header", "PageFooter": "footer", "Picture": "image",
    "Figure": "image", "Table": "table", "Caption": "figure_title",
    "ListItem": "text", "Footnote": "footnote", "Formula": "formula",
}
def make_ocr_boxes(lines):
    # Same skew geometry as finish_ocr_page; all recognized nonempty lines retained.
    quads = [line["polygon_points"] for line in lines]
    angles = []
    for line, q in zip(lines, quads):
        if any(c.isalpha() for c in line["text"]) and q[1][0] - q[0][0] >= 100:
            angles.append((math.atan2(q[1][1]-q[0][1], q[1][0]-q[0][0])
                           + math.atan2(q[2][1]-q[3][1], q[2][0]-q[3][0])) / 2)
    theta = sorted(angles)[len(angles)//2] if angles else 0.0
    points = [p for q in quads for p in q]
    cx = sum(p[0] for p in points)/len(points) if points else 0
    cy = sum(p[1] for p in points)/len(points) if points else 0
    c, s = math.cos(-theta), math.sin(-theta)
    boxes = []
    for line, q in zip(lines, quads):
        if not line["text"].strip():
            continue
        rotated = [(cx+(x-cx)*c-(y-cy)*s, cy+(x-cx)*s+(y-cy)*c) for x,y in q]
        boxes.append({
            "bbox": [min(p[0] for p in q), min(p[1] for p in q),
                     max(p[0] for p in q), max(p[1] for p in q)],
            "bbox_row": [min(p[0] for p in rotated), min(p[1] for p in rotated),
                         max(p[0] for p in rotated), max(p[1] for p in rotated)],
            "quad": q, "text": line["text"], "source_line_order": line["line_order"],
            "detection_confidence": line.get("detection_confidence"),
            "recognition_confidence": line.get("recognition_confidence"),
        })
    return boxes


def _config_hash(args) -> str:
    payload = _config_dict(args)
    payload["models"] = {"surya_dir": str(args.surya_dir), "ppocr_dir": str(args.ppocr_dir),
                          "vietocr_root": str(args.vietocr_root)}
    payload["rule_code_sha256"] = RULE_CODE_SHA256
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Tai model mot lan cho ca batch
# ---------------------------------------------------------------------------
def load_models(args):
    import importlib.util

    surya_dir = Path(args.surya_dir)
    ppocr_dir = Path(args.ppocr_dir)
    vietocr_root = Path(args.vietocr_root)

    # B1 — Surya layout (device cau hinh duoc; notebook mac dinh cpu)
    if importlib.util.find_spec("surya") is None:
        raise ModuleNotFoundError("Thieu surya-ocr.")
    layout_device = args.layout_device
    if layout_device == "auto":
        try:
            import torch
            layout_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            layout_device = "cpu"
    os.environ.update({"FAST_LAYOUT_MODEL_CHECKPOINT": str(surya_dir),
                       "FAST_ORDER_MODEL_CHECKPOINT": str(surya_dir / "order"),
                       "FAST_LAYOUT_USE_ORDER": "true", "FAST_DETECTOR_DEVICE": layout_device})
    from surya.fast_layout import FastLayoutPredictor
    layout_predictor = FastLayoutPredictor(checkpoint=str(surya_dir), use_order=True)
    print(f"Surya layout device: {layout_device}", flush=True)

    # B2 — PP-OCR det (ORT CUDA, batch nhieu trang / 1 lan run)
    import onnxruntime as ort
    try:
        ort.preload_dlls(cuda=True, cudnn=True, directory=None)
    except Exception:
        pass
    import yaml
    post = yaml.safe_load((ppocr_dir / "inference.yml").read_text(encoding="utf-8"))["PostProcess"]
    det_cfg = {"thresh": float(post.get("thresh", .2)), "box_thresh": float(post.get("box_thresh", .45)),
               "unclip_ratio": float(post.get("unclip_ratio", 1.4)),
               "max_candidates": int(post.get("max_candidates", 3000))}
    providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
    except Exception:
        pass
    det_session = ort.InferenceSession(str(ppocr_dir / "inference.onnx"), sess_options=opts, providers=providers)
    print("PP-OCR provider:", det_session.get_providers()[0], flush=True)

    # B4 — VietOCR
    import torch
    torch.backends.cudnn.benchmark = True
    if str(vietocr_root) not in sys.path:
        sys.path.insert(0, str(vietocr_root))
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = Cfg.load_config_from_file(vietocr_root / "config/base.yml")
    cfg.update(Cfg.load_config_from_file(vietocr_root / "config/vgg-transformer.yml"))
    cfg["device"] = device
    cfg["cnn"]["pretrained"] = False
    cfg["predictor"]["beamsearch"] = False
    recognizer = Predictor(cfg)
    recognizer.model.eval()
    print("VietOCR device:", device, flush=True)

    # B5 — rule VBHC 2.1-pruned lay tu section RULE VBHC frozen trong file (khong exec notebook).
    if getattr(args, "rule_version", RULE_VERSION_FROZEN) != RULE_VERSION_FROZEN:
        print(f"Canh bao: --rule-version {getattr(args, 'rule_version')!r} khac rule dong bang "
              f"{RULE_VERSION_FROZEN!r}; van chay rule dong bang.", flush=True)

    return SimpleNamespace(layout_predictor=layout_predictor, layout_device=layout_device,
                           det_session=det_session, det_cfg=det_cfg,
                           recognizer=recognizer, device=device)


# ---------------------------------------------------------------------------
# B1: layout inference theo batch trang (chunk de gioi han RAM/VRAM)
# ---------------------------------------------------------------------------
def _layout_infer(models, images: list[Image.Image], threshold: float, batch_size: int):
    """Chay predictor theo chunk; tra ve predictions giu nguyen thu tu trang."""
    batch_size = max(1, batch_size)
    out = []
    for start in range(0, len(images), batch_size):
        out.extend(models.layout_predictor(images[start:start + batch_size],
                                           threshold=threshold, use_order=True,
                                           batch_size=batch_size))
    return out


def _layout_boxes(pred) -> list[dict]:
    boxes = sorted(
        ({"label": b.label, "score": round(float(b.confidence), 6),
          "bbox": [round(float(v), 3) for v in b.bbox],
          "polygon_points": [[round(float(x), 3), round(float(y), 3)] for x, y in b.polygon],
          "order": int(b.position)} for b in pred.bboxes),
        key=lambda d: d["order"])
    for layout_id, box in enumerate(boxes):
        box["layout_id"] = layout_id
    return boxes


# ---------------------------------------------------------------------------
# B2: det preprocess (thread) -> batched ORT infer -> postprocess (thread)
# ---------------------------------------------------------------------------
_DET_MEAN = np.array([.485, .456, .406], np.float32).reshape(1, 1, 3)
_DET_STD = np.array([.229, .224, .225], np.float32).reshape(1, 1, 3)
# Mau pad cho vung thua khi gop batch (pixel trang sau normalize — giong src/pipeline.py)
_DET_PAD = ((np.ones(3, np.float32) - _DET_MEAN.reshape(3)) / _DET_STD.reshape(3))


def _det_preprocess_one(image: np.ndarray, det_max_side: int):
    h, w = image.shape[:2]
    scale = min(1., det_max_side / max(h, w))
    rh, rw = max(32, round(h * scale / 32) * 32), max(32, round(w * scale / 32) * 32)
    resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (rw, rh), interpolation=cv2.INTER_LINEAR)
    tensor = ((resized.astype(np.float32) / 255 - _DET_MEAN) / _DET_STD).transpose(2, 0, 1)
    return tensor, {"w": w, "h": h, "rw": rw, "rh": rh, "sx": rw / w, "sy": rh / h}


def _det_score(prob, contour):
    x, y, w, h = cv2.boundingRect(contour)
    local = contour.copy().astype(np.int32)
    local[:, :, 0] -= x
    local[:, :, 1] -= y
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [local], 1)
    return float(cv2.mean(prob[y:y + h, x:x + w], mask=mask)[0])


def _det_postprocess_one(prob, meta, det_cfg) -> list[dict]:
    binary = cv2.dilate(((prob >= det_cfg["thresh"]).astype(np.uint8) * 255),
                        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:det_cfg["max_candidates"]]:
        if len(contour) < 3:
            continue
        score, area, perim = _det_score(prob, contour), abs(cv2.contourArea(contour)), cv2.arcLength(contour, True)
        if score < det_cfg["box_thresh"] or area <= 0 or perim <= 0:
            continue
        center, size, angle = cv2.minAreaRect(contour)
        dist = area * det_cfg["unclip_ratio"] / perim
        pts = order_quad(cv2.boxPoints((center, (size[0] + 2 * dist, size[1] + 2 * dist), angle)).astype(np.float32))
        pts[:, 0] = np.clip(pts[:, 0] / meta["sx"], 0, meta["w"] - 1)
        pts[:, 1] = np.clip(pts[:, 1] / meta["sy"], 0, meta["h"] - 1)
        w = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
        h = max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))
        if w < 3 or h < 3:
            continue
        found.append({"score": round(float(score), 6),
                      "bbox": [round(float(v), 3) for v in (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())],
                      "polygon_points": [[round(float(x), 3), round(float(y), 3)] for x, y in pts]})
    return found


def det_infer_pages(models, images: list[np.ndarray], det_max_side: int,
                    det_batch_size: int, workers: int | None = None) -> list[list[dict]]:
    """Suy luan det cho nhieu trang: preprocess song song -> 1 lan ORT run /
    det-batch -> postprocess song song. Tra ve lines cho tung trang."""
    det_session = models.det_session
    det_input, det_output = det_session.get_inputs()[0].name, det_session.get_outputs()[0].name
    det_cfg = models.det_cfg
    det_batch_size = max(1, det_batch_size)
    n_threads = workers or min(8, max(1, (os.cpu_count() or 1)))

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        prepped = list(ex.map(lambda im: _det_preprocess_one(im, det_max_side), images))

    prob_maps: list[np.ndarray] = []
    for start in range(0, len(prepped), det_batch_size):
        jobs = prepped[start:start + det_batch_size]
        bh = max(t.shape[1] for t, _ in jobs)
        bw = max(t.shape[2] for t, _ in jobs)
        batch = np.empty((len(jobs), 3, bh, bw), np.float32)
        for c in range(3):
            batch[:, c] = _DET_PAD[c]
        for i, (tensor, _) in enumerate(jobs):
            batch[i, :, :tensor.shape[1], :tensor.shape[2]] = tensor
        maps = det_session.run([det_output], {det_input: np.ascontiguousarray(batch)})[0]
        for prob, (_, meta) in zip(maps[:, 0], jobs):
            prob_maps.append(np.array(prob[:meta["rh"], :meta["rw"]]))

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        all_lines = list(ex.map(lambda pm: _det_postprocess_one(pm[0], pm[1], det_cfg),
                                zip(prob_maps, [m for _, m in prepped])))
    return all_lines


# ---------------------------------------------------------------------------
# B3: gan layout + sort + crop cho 1 trang (chay song song theo trang)
# ---------------------------------------------------------------------------
def _sort_crop_one_page(job: dict) -> dict:
    page, det_lines, image, run_dir, row_tol, crop_padding = (
        job["page"], job["det_lines"], job["image"], job["run_dir"],
        job["row_tol"], job["crop_padding"])
    groups, fallback = {}, max([b["order"] for b in page["boxes"]], default=-1) + 1
    for line_id, raw in enumerate(det_lines):
        line = {"line_id": line_id, "score": raw["score"], "bbox": raw["bbox"], "polygon_points": raw["polygon_points"]}
        layout, method, metric = assign_layout(line, page["boxes"])
        line.update({"layout_id": layout["layout_id"] if layout else None,
                     "layout_order": layout["order"] if layout else fallback,
                     "layout_label": layout["label"] if layout else "unassigned",
                     "assignment_method": method, "assignment_metric": round(float(metric), 3)})
        groups.setdefault(line["layout_order"], []).append(line)
    ordered = [line for order in sorted(groups) for line in sort_inside_layout(groups[order], row_tol)]
    for reading_order, line in enumerate(ordered):
        line["reading_order"] = reading_order
    lines_dir = run_dir / f"page_{page['page_number']:03d}" / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    for stale in lines_dir.glob("line_*.png"):  # chay lai khong de crop cu doi so luong
        stale.unlink()
    for line in ordered:
        crop = crop_line_polygon(image, line["polygon_points"], crop_padding)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Khong encode duoc crop")
        path = lines_dir / f"line_{line['reading_order']:04d}_layout_{line['layout_order']:03d}_row_{line['row_index']:03d}.png"
        path.write_bytes(buf.tobytes())
        line["crop"] = {"path": str(path.relative_to(REPO_ROOT)), "width": int(crop.shape[1]),
                        "height": int(crop.shape[0]), "bytes": int(path.stat().st_size)}
    assert [line["reading_order"] for line in ordered] == list(range(len(ordered)))
    return {"page_index": page["page_index"], "page_number": page["page_number"],
            "width": page["width"], "height": page["height"], "lines": ordered}


# ---------------------------------------------------------------------------
# B4: VietOCR global batch (gom crop moi trang) + preload song song
# ---------------------------------------------------------------------------
def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def recognize_global(models, crop_paths: list[Path], batch_size: int,
                     workers: int, use_amp: bool) -> list[tuple[str, float | None]]:
    """Suy luan VietOCR tren toan bo crop (moi trang), giu nguyen thu tu dau vao."""
    import torch
    recognizer = models.recognizer
    use_cuda = str(getattr(models, "device", "cpu")).startswith("cuda") and torch.cuda.is_available()
    out: list[tuple[str, float | None]] = []
    batch_size = max(1, batch_size)
    with torch.inference_mode():
        for start in range(0, len(crop_paths), batch_size):
            chunk = crop_paths[start:start + batch_size]
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                images = list(ex.map(_load_rgb, chunk))
            try:
                if use_amp and use_cuda:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        texts, probs = recognizer.predict_batch(images, return_prob=True)
                else:
                    texts, probs = recognizer.predict_batch(images, return_prob=True)
            finally:
                for im in images:
                    im.close()
            out.extend((text.strip(), None if prob is None else round(float(prob), 6))
                       for text, prob in zip(texts, probs))
    return out


# ---------------------------------------------------------------------------
# Chay 1 file PDF (B1 -> B5)
# ---------------------------------------------------------------------------
def run_one_pdf(pdf_path: Path, args, models) -> Path:
    pdf_path = pdf_path.resolve()
    run_dir = (Path(args.output_root).resolve() if Path(args.output_root).is_absolute()
               else (REPO_ROOT / args.output_root).resolve()) / pdf_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and (run_dir / "prediction.json").is_file():
        try:
            _prev = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
            if _prev.get("config_hash") == _config_hash(args) and _prev.get("rule_code_sha256") == RULE_CODE_SHA256:
                print(f"Bo qua (da co prediction.json khop config): {run_dir.relative_to(REPO_ROOT)}", flush=True)
                return run_dir
            print("Cau hinh/rule da doi -> chay lai:", run_dir.relative_to(REPO_ROOT), flush=True)
        except Exception:
            print("Khong doc duoc provenance -> chay lai:", run_dir.relative_to(REPO_ROOT), flush=True)

    t_total0 = time.perf_counter()
    started_at = _now_iso()
    steps: dict[str, float] = {}
    res_steps: dict[str, dict] = {}  # snapshot RAM/VRAM sau moi step
    page_time: dict[int, dict] = {}  # page_number -> cac moc thoi gian + so luong
    _gpu_reset_peak()

    def _page_entry(page_number: int) -> dict:
        return page_time.setdefault(page_number, {"page_number": page_number})

    n_workers = min(8, max(1, (os.cpu_count() or 1)))
    ocr_workers = max(1, args.ocr_workers)
    rr_enabled = not getattr(args, "no_remove_red", False)
    rr_mode = getattr(args, "rr_mode", DEFAULT_RR_MODE)
    orig_pdf_path = pdf_path  # giu de ghi provenance/timing
    active_pdf_path = pdf_path
    rr_stats: dict | None = None
    rr_cleaned_rel: str | None = None

    # ---- B0: tien xu ly xoa dau do (pdf-mode: rewrite XObject truoc khi render) ----
    if rr_enabled and rr_mode == "pdf":
        _sync_cuda()
        t_b0 = time.perf_counter()
        cleaned_pdf = run_dir / (orig_pdf_path.stem + args.rr_suffix + ".pdf")
        if cleaned_pdf.is_file() and not args.overwrite:
            active_pdf_path = cleaned_pdf.resolve()
            rr_stats = {"reused": True, "output": cleaned_pdf.stat().st_size}
            print(f"B0 remove_red: tai dung {cleaned_pdf.name} (da co, --overwrite de lam lai)", flush=True)
        else:
            if cleaned_pdf.exists() or cleaned_pdf.is_symlink():
                cleaned_pdf.unlink()
            try:
                stats = rr_process_pdf(
                    str(orig_pdf_path), str(cleaned_pdf), verbose=False, **rr_options_from_args(args))
                rr_stats = {"reused": False, **stats}
                active_pdf_path = cleaned_pdf.resolve()
                print(f"B0 remove_red (pdf): {stats['changed']} anh sach / {stats['skipped']} giu nguyen / "
                      f"{stats['unreadable']} loi -> {cleaned_pdf.name}", flush=True)
            except Exception as exc:  # noqa: BLE001 - fallback ve PDF goc
                print(f"B0 remove_red THAT BAI ({type(exc).__name__}: {exc}); dung PDF goc.", flush=True)
                rr_stats = {"failed": f"{type(exc).__name__}: {exc}"}
                if cleaned_pdf.is_file():
                    try:
                        cleaned_pdf.unlink()
                    except OSError:
                        pass
        steps["b0_remove_red"] = time.perf_counter() - t_b0
        res_steps["b0_remove_red"] = _res_snapshot()
        if active_pdf_path != orig_pdf_path:
            rr_cleaned_rel = str(cleaned_pdf.relative_to(REPO_ROOT))\
                if cleaned_pdf.is_relative_to(REPO_ROOT) else str(cleaned_pdf)

    # ---- B1+B2+B3 theo page-batch (0 = toan bo trang) ----
    page_images: dict[int, np.ndarray] = {}
    layout_pages: list[dict] = []
    ordered_pages: list[dict] = []

    _sync_cuda()
    t_step0 = time.perf_counter()
    with fitz.open(active_pdf_path) as doc:
        pages = parse_pages(args.pages, doc.page_count)
        # render song song CPU (I/O + decode MuPDF)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            images = list(ex.map(lambda i: render_page(doc[i], args.dpi), pages))
    steps["b1_render"] = time.perf_counter() - t_step0
    res_steps["b1_render"] = _res_snapshot()
    try:
        _pdf_label = pdf_path.relative_to(REPO_ROOT)
    except ValueError:
        _pdf_label = pdf_path
    print(f"PDF: {_pdf_label} | {len(pages)} trang" +
          (f" (B0 pdf -> {Path(rr_cleaned_rel).name})" if rr_cleaned_rel else ""), flush=True)

    # ---- B0 (memory-mode): loc do tren anh render, song song theo trang ----
    if rr_enabled and rr_mode == "memory":
        _sync_cuda()
        t_b0 = time.perf_counter()
        rr_workers = max(1, getattr(args, "rr_workers", DEFAULT_RR_WORKERS))
        rr_kwargs = dict(dense_threshold=args.rr_dense_threshold,
                         dark_red_max=args.rr_dark_red_max,
                         min_black_neighbors=args.rr_min_black_neighbors)
        with ThreadPoolExecutor(max_workers=min(rr_workers, max(1, len(images)))) as ex:
            images = list(ex.map(lambda im: rr_clean_rgb_image(im, **rr_kwargs), images))
        steps["b0_remove_red"] = time.perf_counter() - t_b0
        res_steps["b0_remove_red"] = _res_snapshot()
        rr_stats = {"mode": "memory", "n_images": len(images)}
        print(f"B0 remove_red (memory): da loc {len(images)} anh render", flush=True)

    page_batch = args.page_batch_size if args.page_batch_size and args.page_batch_size > 0 else len(pages)
    chunks = [pages[i:i + page_batch] for i in range(0, len(pages), page_batch)]
    img_by_page = dict(zip(pages, images))

    # ---- B1: surya layout ----
    _sync_cuda()
    t_step0 = time.perf_counter()
    t_infer_total = 0.0

    def _run_layout(chunk_pages: list[int]) -> dict[int, list[dict]]:
        """Tra ve {page_index: boxes} cho 1 chunk trang."""
        chunk_pils = [Image.fromarray(img_by_page[i]) for i in chunk_pages]
        preds = _layout_infer(models, chunk_pils, args.layout_threshold, args.layout_batch_size)
        return {idx: _layout_boxes(pred) for idx, pred in zip(chunk_pages, preds)}

    # ---- B2: ppocr det (batched GPU) ----
    def _run_det(chunk_pages: list[int]) -> dict[int, list[dict]]:
        chunk_images = [img_by_page[i] for i in chunk_pages]
        all_lines = det_infer_pages(models, chunk_images, args.det_max_side,
                                   args.det_batch_size, n_workers)
        return dict(zip(chunk_pages, all_lines))

    for chunk_pages in chunks:
        if args.parallel_layout_det:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_layout = ex.submit(_run_layout, chunk_pages)
                fut_det = ex.submit(_run_det, chunk_pages)
                _sync_cuda()
                t_inf = time.perf_counter()
                layout_map = fut_layout.result()
                _sync_cuda()
                t_infer_total += time.perf_counter() - t_inf
                det_map = fut_det.result()
        else:
            _sync_cuda()
            t_inf = time.perf_counter()
            layout_map = _run_layout(chunk_pages)
            _sync_cuda()
            t_infer_total += time.perf_counter() - t_inf
            det_map = _run_det(chunk_pages)

        for page_index in chunk_pages:
            image = img_by_page[page_index]
            page_images[page_index] = image
            boxes = layout_map[page_index]
            layout_pages.append({"page_index": page_index, "page_number": page_index + 1,
                                 "width": image.shape[1], "height": image.shape[0], "boxes": boxes})

        # ---- B3: gan layout + sort + crop (song song theo trang) ----
        jobs = [{"page": next(p for p in layout_pages if p["page_index"] == idx),
                 "det_lines": det_map[idx], "image": img_by_page[idx],
                 "run_dir": run_dir, "row_tol": args.row_tol,
                 "crop_padding": args.crop_padding} for idx in chunk_pages]
        with ThreadPoolExecutor(max_workers=min(n_workers, max(1, len(jobs)))) as ex:
            chunk_ordered = list(ex.map(_sort_crop_one_page, jobs))
        by_idx = {p["page_index"]: p for p in chunk_ordered}
        for idx in chunk_pages:
            page = by_idx[idx]
            ordered_pages.append(page)
            _page_entry(page["page_number"]).update(
                {"n_layout_boxes": len(next(p for p in layout_pages if p["page_index"] == idx)["boxes"]),
                 "n_det_lines": len(det_map[idx]), "n_lines": len(page["lines"])})
            print(f"Trang {page['page_number']}: "
                  f"{len(next(p for p in layout_pages if p['page_index'] == idx)['boxes'])} vung layout | "
                  f"{len(page['lines'])} dong da sort + crop", flush=True)

    layout_pages.sort(key=lambda p: p["page_index"])
    ordered_pages.sort(key=lambda p: p["page_index"])
    for page in layout_pages:
        assert [b["order"] for b in page["boxes"]] == list(range(len(page["boxes"]))), \
            f"reading order khong lien tuc o trang {page['page_number']}"
    _sync_cuda()
    steps["b1_layout_infer"] = t_infer_total
    steps["b123_layout_det_crop"] = time.perf_counter() - t_step0 - steps["b1_render"]
    res_steps["b123_layout_det_crop"] = _res_snapshot()
    print(f"OK: {len(layout_pages)} trang layout + reading order; "
          f"{sum(len(p['lines']) for p in ordered_pages)} dong, khong mat line.", flush=True)
    # Giai phong anh render goc (crop da luu dia); giu layout/ocr nhe
    page_images.clear()

    # ---- B4: vietocr (global batch moi trang) ----
    _sync_cuda()
    t_step0 = time.perf_counter()
    all_crops: list[Path] = []
    owner: list[tuple[int, int]] = []  # (pos trang trong ordered_pages, reading_order)
    for pos, page in enumerate(ordered_pages):
        crops = sorted((run_dir / f"page_{page['page_number']:03d}" / "lines").glob("line_*.png"),
                       key=lambda p: int(p.stem.split("_")[1]))  # prefix line_%04d = reading_order
        if len(crops) != len(page["lines"]):
            raise RuntimeError(f"Trang {page['page_number']}: {len(crops)} crop nhung co {len(page['lines'])} line")
        for crop in crops:
            owner.append((pos, len(all_crops)))
            all_crops.append(crop)
    texts = recognize_global(models, all_crops, args.ocr_batch_size, ocr_workers, args.ocr_amp) \
        if all_crops else []
    by_page: dict[int, list] = {pos: [] for pos in range(len(ordered_pages))}
    for (pos, _), (text, conf) in zip(owner, texts):
        by_page[pos].append((text, conf))

    ocr_pages = []
    for pos, page in enumerate(ordered_pages):
        t_page0 = time.perf_counter()
        page_crops = sorted((run_dir / f"page_{page['page_number']:03d}" / "lines").glob("line_*.png"),
                            key=lambda p: int(p.stem.split("_")[1]))
        lines = [{"line_order": src["reading_order"], "layout_order": src["layout_order"],
                  "row_order": src["row_index"], "layout_id": src["layout_id"], "layout_label": src["layout_label"],
                  "text": text, "recognition_confidence": conf, "bbox": src["bbox"],
                  "polygon_points": src["polygon_points"], "detection_confidence": src["score"],
                  "assignment_method": src["assignment_method"], "assignment_metric": src["assignment_metric"],
                  "crop_file": str(crop.relative_to(REPO_ROOT))}
                 for src, crop, (text, conf) in zip(page["lines"], page_crops, by_page[pos])]
        _page_entry(page["page_number"]).update(
            {"ocr_seconds": _round3(time.perf_counter() - t_page0)})
        text_path = run_dir / f"page_{page['page_number']:03d}.txt"
        text_path.write_text("\n".join(line["text"] for line in lines) + ("\n" if lines else ""), encoding="utf-8")
        assert [line["line_order"] for line in lines] == list(range(len(lines)))
        ocr_pages.append({"page_index": page["page_index"], "page_number": page["page_number"],
                          "width": page["width"], "height": page["height"],
                          "text_file": text_path.name, "lines": lines})
        print(f"Trang {page['page_number']}: {len(lines)} dong OCR -> {text_path.name}", flush=True)
    (run_dir / "full_ocr_result.json").write_text(
        json.dumps({"source_pdf": str(pdf_path.relative_to(REPO_ROOT)), "pages": ocr_pages},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {sum(len(p['lines']) for p in ocr_pages)} dong OCR.", flush=True)
    _sync_cuda()
    steps["b4_vietocr"] = time.perf_counter() - t_step0
    res_steps["b4_vietocr"] = _res_snapshot()

    # ---- B5: rule extraction (frozen v2.1-pruned, standalone) ----
    _sync_cuda()
    t_step0 = time.perf_counter()
    text_fields = TEXT_FIELDS

    layout_by_page = {p["page_number"]: p for p in layout_pages}
    pages_blocks, page_sizes, warnings = [], [], []
    for page in sorted(ocr_pages, key=lambda p: p["page_number"]):
        regions = sorted(layout_by_page[page["page_number"]]["boxes"], key=lambda b: b["order"])
        if (page["width"], page["height"]) != (layout_by_page[page["page_number"]]["width"], layout_by_page[page["page_number"]]["height"]):
            raise ValueError(f"Layout va OCR khong cung he toa do o trang {page['page_number']}")
        boxes = make_ocr_boxes(sorted(page["lines"], key=lambda line: line["line_order"]))
        blocks = []
        for i, region in enumerate(regions, 1):
            btype = LABEL_MAP.get(region["label"], region["label"].lower())
            if region["label"] not in LABEL_MAP and btype not in {"text", "header", "footer", "table", "image", "doc_title", "paragraph_title"}:
                warnings.append(f"Trang {page['page_number']}: label {region['label']!r} -> {btype!r}")
            kind = "table" if btype == "table" else "skip" if btype in schema.skip_types else "text"
            block = {"type": btype, "bbox": region["bbox"], "score": region["score"],
                     "source_layout_id": i, "source_fullocr_layout_id": region["layout_id"],
                     "content_type": kind, "content": None}
            if kind == "table":
                warnings.append(f"Trang {page['page_number']}: bang dung OCR text fallback, thieu TSR")
                table_lines = [b for b in boxes if overlap_ratio(b["bbox"], block["bbox"]) > 0.5]
                block["content"] = "\n".join(" ".join(b["text"] for b in row) for row in tb_rows(table_lines, 0.5, 1.0, 100))
            blocks.append(block)
        prepped = SimpleNamespace(img=SimpleNamespace(size=(page["width"], page["height"])), timings={})
        pages_blocks.append(assemble_page(pipeline_adapter, page["page_number"] - 1, prepped, blocks, boxes, debug_dir=None))
        page_sizes.append((page["width"], page["height"]))
    print("Pages:", len(pages_blocks), "| Blocks:", sum(map(len, pages_blocks)), "| Warnings:", len(warnings), flush=True)

    t0 = time.perf_counter()
    prediction, extraction_debug = extract_vbhc(
        pages_blocks, original_page_sizes=page_sizes, processed_page_sizes=page_sizes,
        crop_offsets=[(0.0, 0.0)] * len(page_sizes), geometry_reliable=True)
    _sync_cuda()
    steps["b5_rule_assemble_extract"] = time.perf_counter() - t_step0
    res_steps["b5_rule_assemble_extract"] = _res_snapshot()
    steps["b5_rule_extract_only"] = time.perf_counter() - t0
    prediction["processing_time"] = steps["b5_rule_extract_only"]
    assert tuple(prediction["information"][0]) == text_fields
    assert all(v["type"] == "string" and isinstance(v["value"], str)
               for v in prediction["information"][0].values())
    _write_json(run_dir / "prediction.json", prediction)
    _write_json(run_dir / "extraction_debug.json", extraction_debug)
    _write_json(run_dir / "page_blocks.json", {
        "pages_blocks": pages_blocks, "original_page_sizes": page_sizes,
        "processed_page_sizes": page_sizes, "crop_offsets": [(0, 0)] * len(page_sizes), "geometry_reliable": True})
    _write_json(run_dir / "provenance.json", {
        "rule_version_frozen": RULE_VERSION_FROZEN, "rule_version_requested": args.rule_version,
        "rule_code_sha256": RULE_CODE_SHA256, "rule_code_cells": list(RULE_CODE_CELLS),
        "lexicon_version": RULE_LEXICON.get("version"),
        "development_manifest_sha256": DEVELOPMENT_LEXICON.get("manifest_sha256"),
        "label_map": LABEL_MAP, "warnings": warnings,
        "config_hash": _config_hash(args), "config": _config_dict(args),
        "remove_red": {"enabled": rr_enabled, "mode": (rr_mode if rr_enabled else "off"),
                       "options": (rr_options_from_args(args) if rr_enabled else {}),
                       "stats": rr_stats, "cleaned_pdf": rr_cleaned_rel,
                       "source": "remove_red.py V5 vendored (rr_process_pdf / remove_red_from_bgr)"},
        "selected_pages": sorted(p["page_number"] for p in ocr_pages),
        "processing_time_scope": "assemble_plus_extraction_measured_separately",
        "standalone": True,
        "fidelity": "Rule dong bang verbatim tu 6_1_optimize_rule.ipynb cells (3,5,7,9,11,13,15,17,19,21)"})

    replayed, replay_debug = extract_vbhc(**json.loads((run_dir / "page_blocks.json").read_text(encoding="utf-8")))
    assert replayed["information"] == prediction["information"] and replay_debug == extraction_debug
    for field, items in extraction_debug["evidence"].items():
        for item in items:
            assert 1 <= item["page"] <= len(pages_blocks)
            assert 1 <= item["block"] <= len(pages_blocks[item["page"] - 1])
    for field, value in prediction["information"][0].items():
        print(f"{field}: {value['value'][:80]}", flush=True)
    print("13-field contract, evidence va replay: OK ->", (run_dir / "prediction.json").relative_to(REPO_ROOT), flush=True)

    # ---- timing: tong + step + trang -> outputs/merge_remove_red/<stem>/timing.json ----
    _sync_cuda()
    total_seconds = time.perf_counter() - t_total0
    steps_done = {name: _round3(val) for name, val in steps.items()}
    pages_done = [page_time[key] for key in sorted(page_time)]
    try:
        _src_rel = str(orig_pdf_path.relative_to(REPO_ROOT))
    except ValueError:
        _src_rel = str(orig_pdf_path)
    timing = {"source_pdf": _src_rel,
              "active_pdf": (rr_cleaned_rel or _src_rel),
              "remove_red": {"enabled": rr_enabled, "mode": (rr_mode if rr_enabled else "off"),
                             "stats": rr_stats},
              "run_dir": str(run_dir.relative_to(REPO_ROOT)),
              "started_at": started_at, "ended_at": _now_iso(),
              "total_seconds": _round3(total_seconds),
              "n_pages": len(pages_done),
              "n_lines": sum(len(p["lines"]) for p in ordered_pages),
              "steps_seconds": steps_done,
              "resource": {"peak_rss_mb": _peak_rss_mb(), "peak_gpu_alloc_mb": _gpu_mem_mb()["peak_alloc_mb"],
                           "steps": res_steps},
              "env": getattr(models, "env", {}),
              "config": _config_dict(args),
              "pages": pages_done}
    (run_dir / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = " | ".join(f"{name}={val:.1f}s" for name, val in steps_done.items())
    print(f"[timing] total={total_seconds:.1f}s | {summary}", flush=True)
    return run_dir


# ---------------------------------------------------------------------------
def discover_pdfs(args) -> list[Path]:
    if args.pdf and args.input_dir:
        raise ValueError("Chi dung 1 trong --pdf hoac --input-dir.")
    if args.pdf:
        p = Path(args.pdf)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Khong tim thay PDF: {p}")
        return [p]
    if args.input_dir:
        root = Path(args.input_dir)
        if not root.is_absolute():
            root = (REPO_ROOT / root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Khong tim thay thu muc: {root}")
        pdfs = sorted(p for p in root.rglob("*.pdf") if p.is_file())
        if not pdfs:
            raise FileNotFoundError(f"Khong co PDF nao trong: {root}")
        return pdfs
    raise ValueError("Can --pdf FILE hoac --input-dir DIR.")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="PDF -> [B0 remove_red] -> Surya layout + PP-OCR det + VietOCR -> rule (batch, toi uu GPU).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="Duong dan 1 file PDF (tuong doi repo root hoac tuyet doi).")
    src.add_argument("--input-dir", help="Thu muc chua PDF; quet de quy *.pdf.")
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Thu muc ket qua (mac dinh: outputs/merge_remove_red).")
    ap.add_argument("--pages", default=None, help="Chi chay cac trang, vi du '0,2'. Mac dinh: toan bo file.")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ap.add_argument("--layout-threshold", type=float, default=DEFAULT_LAYOUT_THRESHOLD)
    ap.add_argument("--layout-device", choices=("cpu", "cuda", "auto"), default=DEFAULT_LAYOUT_DEVICE,
                    help="Device cho Surya layout (mac dinh: cpu giong notebook; cuda nhanh hon nhung ton VRAM).")
    ap.add_argument("--layout-batch-size", type=int, default=DEFAULT_LAYOUT_BATCH,
                    help="So trang / 1 lan goi Surya predictor (mac dinh: 8).")
    ap.add_argument("--det-max-side", type=int, default=DEFAULT_DET_MAX_SIDE)
    ap.add_argument("--det-batch-size", type=int, default=DEFAULT_DET_BATCH,
                    help="So trang / 1 lan ORT inference (mac dinh: 4).")
    ap.add_argument("--page-batch-size", type=int, default=DEFAULT_PAGE_BATCH,
                    help="So trang xu ly B1-B3 cung luc; 0 = toan bo file (mac dinh: 0).")
    ap.add_argument("--parallel-layout-det", action="store_true",
                    help="Chay B1 (layout) va B2 (det) song song tren 2 worker.")
    ap.add_argument("--row-tol", type=float, default=DEFAULT_ROW_TOL)
    ap.add_argument("--crop-padding", type=int, default=DEFAULT_CROP_PADDING)
    ap.add_argument("--ocr-batch-size", type=int, default=DEFAULT_OCR_BATCH,
                    help="Batch VietOCR global moi trang (mac dinh: 64; notebook: 32/trang).")
    ap.add_argument("--ocr-workers", type=int, default=DEFAULT_OCR_WORKERS,
                    help="Thread preload anh crop cho VietOCR (mac dinh: 4).")
    ap.add_argument("--ocr-amp", action="store_true",
                    help="Autocast fp16 cho VietOCR (nhanh hon, co the doi text o bien).")
    ap.add_argument("--rule-version", default=DEFAULT_RULE_VERSION)
    ap.add_argument("--no-remove-red", action="store_true",
                    help="Tat B0 tien xu ly xoa dau do (chay pipeline goc nhu 8_merge.py).")
    ap.add_argument("--rr-mode", choices=("pdf", "memory"), default=DEFAULT_RR_MODE,
                    help="B0 remove_red: pdf=rewrite XObject truoc khi render (mac dinh); "
                    "memory=loc tren anh render, khong ghi PDF moi.")
    ap.add_argument("--rr-suffix", default=DEFAULT_RR_SUFFIX,
                    help="Hau to file PDF sach o pdf-mode (mac dinh: %(default)s).")
    ap.add_argument("--rr-encode", choices=RR_ENCODERS, default=DEFAULT_RR_ENCODE,
                    help="pdf-mode: flate=lossless (mac dinh); jpeg=nho hon.")
    ap.add_argument("--rr-jpeg-quality", type=int, default=DEFAULT_RR_JPEG_QUALITY)
    ap.add_argument("--rr-dense-threshold", type=float, default=DEFAULT_RR_DENSE_THRESHOLD)
    ap.add_argument("--rr-dark-red-max", type=int, default=DEFAULT_RR_DARK_RED_MAX)
    ap.add_argument("--rr-min-black-neighbors", type=int, default=DEFAULT_RR_MIN_BLACK_NEIGHBORS)
    ap.add_argument("--rr-workers", type=int, default=DEFAULT_RR_WORKERS,
                    help="Thread loc anh render o memory-mode (mac dinh: %(default)s).")
    ap.add_argument("--overwrite", action="store_true", help="Chay lai ca file da co prediction.json.")
    ap.add_argument("--surya-dir", default=str(REPO_ROOT / "models/surya_layout2"))
    ap.add_argument("--ppocr-dir", default=str(REPO_ROOT / "models/PP-OCRv6_medium_det_onnx"))
    ap.add_argument("--vietocr-root", default=str(REPO_ROOT / "models/vietocr"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pdfs = discover_pdfs(args)
    print(f"Tim thay {len(pdfs)} PDF.", flush=True)
    models = load_models(args)
    models.env = _collect_env(models, args)
    print(f"[env] {models.env['platform']} | CPU x{models.env['cpu_count']} | "
          f"RAM {models.env['ram_total_gb']}GB | GPU {models.env['cuda_devices'] or 'none'} | "
          f"surya {models.layout_device} | det {models.env['det_provider']} | vietocr {models.env['vietocr_device']}", flush=True)
    ok, fail = 0, []
    records: list[dict] = []
    output_root = Path(args.output_root).resolve() if Path(args.output_root).is_absolute() \
        else (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "timing_summary.json"
    batch_started_at = _now_iso()
    t0 = time.perf_counter()
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf}", flush=True)
        try:
            run_dir = run_one_pdf(pdf, args, models)
            ok += 1
            timing_path = run_dir / "timing.json"
            if timing_path.is_file():
                record = json.loads(timing_path.read_text(encoding="utf-8"))
                record["status"] = "ok"
            else:  # file duoc bo qua do da co prediction.json tu lan chay truoc
                record = {"original_pdf": str(pdf), "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                          "status": "skipped", "total_seconds": None, "steps_seconds": {}, "pages": []}
            records.append(record)
        except Exception as exc:  # noqa: BLE001 - batch chay tiep file sau
            logging.exception("Loi file %s", pdf)
            fail.append((str(pdf), f"{type(exc).__name__}: {exc}"))
            records.append({"original_pdf": str(pdf), "status": f"error: {type(exc).__name__}: {exc}"})
        summary_path.write_text(json.dumps(
            {"started_at": batch_started_at, "n_files": len(pdfs), "n_ok": ok,
             "batch_seconds": _round3(time.perf_counter() - t0),
             "env": getattr(models, "env", {}), "config": _config_dict(args), "files": records},
            ensure_ascii=False, indent=2), encoding="utf-8")
    batch_seconds = time.perf_counter() - t0
    summary_path.write_text(json.dumps(
        {"started_at": batch_started_at, "ended_at": _now_iso(),
         "n_files": len(pdfs), "n_ok": ok, "n_fail": len(fail),
         "batch_seconds": _round3(batch_seconds),
         "env": getattr(models, "env", {}), "config": _config_dict(args), "files": records},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nXong {ok}/{len(pdfs)} file trong {batch_seconds:.0f}s.", flush=True)
    print(f"Timing tong hop: {summary_path.relative_to(REPO_ROOT)}", flush=True)
    if fail:
        for pdf, err in fail:
            print(f"FAIL: {pdf}: {err}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
