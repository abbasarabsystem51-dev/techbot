"""
src/techbot/images.py — Image finder, validation, and Pillow graphic card generator for lessons.

Features:
  1. ImageFinder:
     - Extracts og:image and twitter:image from HTML or feed enclosures.
     - Validates dimensions locally (minimum width >= 800px).
     - Binary header size inspection for PNG, JPEG, GIF, and WebP (works even without Pillow).
     - Filters out icons, logos, avatars, trackers, and low-res squares.
  2. LessonCardGenerator:
     - Generates 1200x675 brand-consistent cards for lesson posts.
     - Displays lesson title, 'درس N از ۳۰', and '@tech_news_110' branding.
     - Resolves Vazirmatn font with Tahoma/Segoe UI system font fallbacks.
     - Persian text shaping (arabic-reshaper + python-bidi, with built-in pure-Python fallback).
"""
from __future__ import annotations

import io
import logging
import re
import struct
import zlib
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.techbot.http import HttpClient, create_http_client
from src.utils import persian_digit

logger = logging.getLogger(__name__)

# Try importing Pillow
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

# Try importing arabic_reshaper and bidi
try:
    import arabic_reshaper
except ImportError:
    arabic_reshaper = None

try:
    from bidi.algorithm import get_display
except ImportError:
    get_display = None


# ---------------------------------------------------------------------------
# Pure-Python Persian / Arabic Reshaper & RTL Fallback
# ---------------------------------------------------------------------------

_PERSIAN_LETTERS_MAP = {
    # char: (isolated, final, initial, medial)
    "ا": ("\uFE8D", "\uFE8E", "\uFE8D", "\uFE8E"),
    "آ": ("\uFE81", "\uFE82", "\uFE81", "\uFE82"),
    "ب": ("\uFE8F", "\uFE90", "\uFE91", "\uFE92"),
    "پ": ("\uFB56", "\uFB57", "\uFB58", "\uFB59"),
    "ت": ("\uFE95", "\uFE96", "\uFE97", "\uFE98"),
    "ث": ("\uFE99", "\uFE9A", "\uFE9B", "\uFE9C"),
    "ج": ("\uFE9D", "\uFE9E", "\uFE9F", "\uFEA0"),
    "چ": ("\uFB7A", "\uFB7B", "\uFB7C", "\uFB7D"),
    "ح": ("\uFEA1", "\uFEA2", "\uFEA3", "\uFEA4"),
    "خ": ("\uFEA5", "\uFEA6", "\uFEA7", "\uFEA8"),
    "د": ("\uFEA9", "\uFEAA", "\uFEA9", "\uFEAA"),
    "ذ": ("\uFEAB", "\uFEAC", "\uFEAB", "\uFEAC"),
    "ر": ("\uFEAD", "\uFEAE", "\uFEAD", "\uFEAE"),
    "ز": ("\uFEAF", "\uFEB0", "\uFEAF", "\uFEB0"),
    "ژ": ("\uFB8A", "\uFB8B", "\uFB8A", "\uFB8B"),
    "س": ("\uFEB1", "\uFEB2", "\uFEB3", "\uFEB4"),
    "ش": ("\uFEB5", "\uFEB6", "\uFEB7", "\uFEB8"),
    "ص": ("\uFEB9", "\uFEBA", "\uFEBB", "\uFEBC"),
    "ض": ("\uFEBD", "\uFEBE", "\uFEBF", "\uFEC0"),
    "ط": ("\uFEC1", "\uFEC2", "\uFEC3", "\uFEC4"),
    "ظ": ("\uFEC5", "\uFEC6", "\uFEC7", "\uFEC8"),
    "ع": ("\uFEC9", "\uFECA", "\uFECB", "\uFECC"),
    "غ": ("\uFECD", "\uFECE", "\uFECF", "\uFED0"),
    "ف": ("\uFED1", "\uFED2", "\uFED3", "\uFED4"),
    "ق": ("\uFED5", "\uFED6", "\uFED7", "\uFED8"),
    "ک": ("\uFED9", "\uFEDA", "\uFEDB", "\uFEDC"),
    "گ": ("\uFB92", "\uFB93", "\uFB94", "\uFB95"),
    "ل": ("\uFEDD", "\uFEDE", "\uFEDF", "\uFEE0"),
    "م": ("\uFEE1", "\uFEE2", "\uFEE3", "\uFEE4"),
    "ن": ("\uFEE5", "\uFEE6", "\uFEE7", "\uFEE8"),
    "و": ("\uFEED", "\uFEEE", "\uFEED", "\uFEEE"),
    "ه": ("\uFEE9", "\uFEEA", "\uFEEB", "\uFEEC"),
    "ی": ("\uFEF1", "\uFEF2", "\uFEF3", "\uFEF4"),
    "ي": ("\uFEF1", "\uFEF2", "\uFEF3", "\uFEF4"),
    "ئ": ("\uFE89", "\uFE8A", "\uFE8B", "\uFE8C"),
}

_NON_CONNECTING_AFTER = frozenset({"ا", "آ", "د", "ذ", "ر", "ز", "ژ", "و"})


def shape_persian_text(text: str) -> str:
    """
    Shape Persian text for visual display from right to left.
    Uses arabic_reshaper + bidi if available, or pure-Python shaping fallback.
    """
    if not text:
        return ""

    if arabic_reshaper is not None and get_display is not None:
        try:
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        except Exception as exc:
            logger.debug("arabic_reshaper failed: %s. Using internal fallback.", exc)

    # Internal pure-Python shaping fallback
    chars = list(text)
    n = len(chars)
    reshaped_chars: list[str] = []

    for i, ch in enumerate(chars):
        if ch not in _PERSIAN_LETTERS_MAP:
            reshaped_chars.append(ch)
            continue

        prev_connects = False
        if i > 0 and chars[i - 1] in _PERSIAN_LETTERS_MAP:
            prev_ch = chars[i - 1]
            if prev_ch not in _NON_CONNECTING_AFTER:
                prev_connects = True

        next_connects = False
        if i < n - 1 and chars[i + 1] in _PERSIAN_LETTERS_MAP:
            next_connects = True

        iso, final, init, med = _PERSIAN_LETTERS_MAP[ch]
        if prev_connects and next_connects and ch not in _NON_CONNECTING_AFTER:
            reshaped_chars.append(med)
        elif prev_connects:
            reshaped_chars.append(final)
        elif next_connects and ch not in _NON_CONNECTING_AFTER:
            reshaped_chars.append(init)
        else:
            reshaped_chars.append(iso)

    # Reverse words / character order for RTL rendering
    result = "".join(reversed(reshaped_chars))
    return result


# ---------------------------------------------------------------------------
# Pure-Python Binary Image Dimensions Parser
# ---------------------------------------------------------------------------

def parse_image_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """
    Inspect binary header of PNG, JPEG, GIF, or WebP bytes to extract (width, height).
    Guarantees local dimension validation without requiring Pillow installed.
    """
    if len(data) < 24:
        return None

    # 1. PNG: \x89PNG\r\n\x1a\n
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        try:
            w, h = struct.unpack(">II", data[16:24])
            return int(w), int(h)
        except Exception:
            return None

    # 2. GIF: GIF87a or GIF89a
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        try:
            w, h = struct.unpack("<HH", data[6:10])
            return int(w), int(h)
        except Exception:
            return None

    # 3. JPEG: \xFF\xD8
    if data.startswith(b"\xff\xd8"):
        try:
            idx = 2
            size = len(data)
            while idx < size - 8:
                marker, length = struct.unpack(">HH", data[idx:idx + 4])
                idx += 2
                # SOF0, SOF1, SOF2 markers containing dimensions
                if marker in (0xFFC0, 0xFFC1, 0xFFC2):
                    h, w = struct.unpack(">HH", data[idx + 1:idx + 5])
                    return int(w), int(h)
                idx += length
        except Exception:
            pass

    # 4. WebP: RIFF....WEBP
    if data.startswith(b"RIFF") and len(data) > 30 and data[8:12] == b"WEBP":
        try:
            # Simple VP8 chunk
            if data[12:16] == b"VP8 ":
                w, h = struct.unpack("<HH", data[26:30])
                return int(w & 0x3FFF), int(h & 0x3FFF)
            # VP8X extended chunk
            if data[12:16] == b"VP8X":
                w = int.from_bytes(data[24:27], "little") + 1
                h = int.from_bytes(data[27:30], "little") + 1
                return w, h
        except Exception:
            return None

    # 5. Pillow fallback if available
    if Image is not None:
        try:
            im = Image.open(io.BytesIO(data))
            return im.size
        except Exception:
            return None

    return None


# ---------------------------------------------------------------------------
# ImageFinder
# ---------------------------------------------------------------------------

REJECTED_KEYWORDS: frozenset[str] = frozenset({
    "avatar",
    "favicon",
    "logo",
    "icon",
    "spinner",
    "1x1",
    "badge",
    "pixel",
    "emoji",
    "button",
    "placeholder",
    "default-image",
    "gravatar",
    "user_image",
    "author",
})


class ImageFinder:
    """
    Extracts and validates og:image or feed images from target articles.
    """

    def __init__(self, min_width: int = 800) -> None:
        self.min_width = min_width

    def extract_og_image(self, html: str, base_url: str = "") -> Optional[str]:
        """Parse HTML to extract OpenGraph or Twitter card image URL."""
        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")

        # Prioritized meta tags
        meta_candidates = [
            ("meta", {"property": "og:image"}),
            ("meta", {"property": "og:image:url"}),
            ("meta", {"name": "twitter:image"}),
            ("meta", {"name": "twitter:image:src"}),
            ("link", {"rel": "image_src"}),
        ]

        for tag_name, attrs in meta_candidates:
            tag = soup.find(tag_name, attrs=attrs)
            if tag:
                content = tag.get("content") or tag.get("href")
                if content:
                    img_url = content.strip()
                    if base_url:
                        img_url = urljoin(base_url, img_url)
                    return img_url

        return None

    def is_valid_image_url(self, url: str) -> bool:
        """Reject obvious icons, avatars, svg badges, and tracking pixels."""
        if not url:
            return False

        clean_url = url.lower()
        parsed = urlparse(clean_url)
        path = parsed.path

        if path.endswith((".svg", ".ico", ".gif")):
            return False

        for rejected in REJECTED_KEYWORDS:
            if rejected in clean_url:
                return False

        return True

    def validate_image_dimensions(self, image_bytes: bytes) -> bool:
        """
        Ensure image meets minimum width requirements and is not a square avatar.
        """
        dims = parse_image_dimensions(image_bytes)
        if dims is None:
            return False

        width, height = dims
        if width < self.min_width:
            return False

        # Reject square avatars / logos (e.g. 1:1 ratio if not very large)
        if width == height and width < 900:
            return False

        # Reject extreme aspect ratios (e.g. horizontal banners or skyscrapers)
        aspect = width / height if height > 0 else 0
        if aspect > 3.5 or aspect < 0.4:
            return False

        return True

    async def find_image(
        self,
        article_url: str,
        html: Optional[str] = None,
        fallback_url: Optional[str] = None,
        client: Optional[HttpClient] = None,
    ) -> Optional[str]:
        """
        Find and validate an image for an article.
        """
        http_client = client or create_http_client(timeout=10.0)

        candidate_url: Optional[str] = None
        if html:
            candidate_url = self.extract_og_image(html, base_url=article_url)

        if not candidate_url and article_url:
            try:
                page_html = await http_client.get_text(article_url)
                candidate_url = self.extract_og_image(page_html, base_url=article_url)
            except Exception as exc:
                logger.debug("Failed to fetch HTML for og:image from %s: %s", article_url, exc)

        if not candidate_url:
            candidate_url = fallback_url

        if not candidate_url or not self.is_valid_image_url(candidate_url):
            return None

        # Verify dimensions by downloading image
        try:
            image_bytes = await http_client.get_bytes(candidate_url)
            if self.validate_image_dimensions(image_bytes):
                return candidate_url
        except Exception as exc:
            logger.debug("Failed to validate image %s: %s", candidate_url, exc)

        return None


# ---------------------------------------------------------------------------
# LessonCardGenerator (Pillow Graphic Card)
# ---------------------------------------------------------------------------

class LessonCardGenerator:
    """
    Generates clean, brand-consistent 1200x675 graphic cards for lesson posts.
    """

    def __init__(self, font_path: Optional[Path | str] = None) -> None:
        self.font_path = self._resolve_font(font_path)

    def _resolve_font(self, custom_path: Optional[Path | str] = None) -> Optional[str]:
        """Find the best available Persian/Arabic TrueType font."""
        if custom_path and Path(custom_path).exists():
            return str(custom_path)

        root = Path(__file__).resolve().parents[2]
        candidates = [
            root / "fonts" / "Vazirmatn-Bold.ttf",
            root / "fonts" / "Vazirmatn-Regular.ttf",
            Path("C:/Windows/Fonts/tahoma.ttf"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]

        for p in candidates:
            if p.exists():
                return str(p)

        return None

    def _load_font(self, size: int) -> Any:
        """Load ImageFont at specific size or return default."""
        if ImageFont is not None and self.font_path:
            try:
                return ImageFont.truetype(self.font_path, size=size)
            except Exception:
                pass
        if ImageFont is not None:
            return ImageFont.load_default()
        return None

    def generate_card(
        self,
        lesson_number: int,
        lesson_title: str,
        total_lessons: int = 30,
        channel_handle: str = "@tech_news_110",
    ) -> bytes:
        """
        Generate a 1200x675 graphic card as PNG bytes.
        """
        width, height = 1200, 675
        badge_text = f"درس {persian_digit(lesson_number)} از {persian_digit(total_lessons)}"

        # Pillow rendering path
        if Image is not None and ImageDraw is not None:
            # 1. Dark canvas with deep navy background
            img = Image.new("RGB", (width, height), color=(13, 17, 23))
            draw = ImageDraw.Draw(img)

            # 2. Draw card container with rounded border
            margin = 40
            card_box = [margin, margin, width - margin, height - margin]
            draw.rounded_rectangle(card_box, radius=24, fill=(22, 27, 34), outline=(48, 54, 61), width=2)

            # 3. Decorative glowing top accent bar
            accent_bar = [margin + 20, margin + 2, width - margin - 20, margin + 8]
            draw.rounded_rectangle(accent_bar, radius=4, fill=(35, 134, 54))

            # 4. Badge: 'درس N از ۳۰'
            badge_font = self._load_font(28)
            badge_shaped = shape_persian_text(badge_text)

            badge_w, badge_h = 240, 56
            badge_x = (width - badge_w) // 2
            badge_y = margin + 50
            draw.rounded_rectangle(
                [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                radius=14,
                fill=(33, 38, 45),
                outline=(35, 134, 54),
                width=2,
            )
            draw.text(
                (width // 2, badge_y + badge_h // 2),
                badge_shaped,
                fill=(240, 246, 252),
                font=badge_font,
                anchor="mm",
            )

            # 5. Lesson Title
            title_font = self._load_font(52)
            title_shaped = shape_persian_text(lesson_title)
            title_y = height // 2 - 30
            draw.text(
                (width // 2, title_y),
                title_shaped,
                fill=(255, 255, 255),
                font=title_font,
                anchor="mm",
            )

            # 6. Subtitle: 'دوره ۳۰ روزه تسلط بر هوش مصنوعی'
            sub_font = self._load_font(26)
            sub_text = shape_persian_text("دوره جامع و کاربردی هوش مصنوعی")
            draw.text(
                (width // 2, title_y + 80),
                sub_text,
                fill=(139, 148, 158),
                font=sub_font,
                anchor="mm",
            )

            # 7. Bottom Branding: Channel handle and Name
            brand_font = self._load_font(28)
            brand_text = shape_persian_text("تک‌نیوز")
            footer_y = height - margin - 50

            draw.text((margin + 60, footer_y), channel_handle, fill=(88, 166, 255), font=brand_font, anchor="lm")
            draw.text((width - margin - 60, footer_y), brand_text, fill=(240, 246, 252), font=brand_font, anchor="rm")

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

        # Pure-Python valid PNG image fallback (1200x675) with dark navy theme
        signature = b"\x89PNG\r\n\x1a\n"

        def _make_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
            return (
                struct.pack(">I", len(chunk_data))
                + chunk_type
                + chunk_data
                + struct.pack(">I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        # Scanlines: filter byte 0 followed by RGB pixels (#0D1117 background)
        row = bytearray([0]) + bytearray([13, 17, 23]) * width
        raw_pixels = bytes(row * height)
        idat = zlib.compress(raw_pixels, level=6)

        return signature + _make_chunk(b"IHDR", ihdr) + _make_chunk(b"IDAT", idat) + _make_chunk(b"IEND", b"")

    def save_card(
        self,
        lesson_number: int,
        lesson_title: str,
        output_path: Path | str,
        **kwargs: Any,
    ) -> Path:
        """Generate and save card to a disk file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.generate_card(lesson_number, lesson_title, **kwargs)
        target.write_bytes(data)
        return target
