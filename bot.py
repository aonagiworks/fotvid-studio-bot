#!/usr/bin/env python3
"""Photo Video Studio — rembg + HD + denoise + colorize + video note + force join."""
from __future__ import annotations
import asyncio
import io
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from rembg import remove, new_session
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.constants import ChatMemberStatus
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "8978306167:AAG7lPvsQxOMwNFyj0erFO42SOCeWn3ylzo")
ADMIN_ID = int(os.environ.get("ADMIN_ID", "6247336698"))
FORCE_CHANNEL = os.environ.get("FORCE_CHANNEL", "botunlverse")
CHANNEL_URL = f"https://t.me/{FORCE_CHANNEL}"

_SESSION = None


def get_session(model: str = "u2net"):
    global _SESSION
    if _SESSION is None or getattr(_SESSION, "_model_name", None) != model:
        _SESSION = new_session(model)
        _SESSION._model_name = model
    return _SESSION


def main_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("🖼️ Hapus BG"), KeyboardButton("🎨 Ganti BG")],
            [KeyboardButton("🔍 HD Upscale"), KeyboardButton("✨ Denoise")],
            [KeyboardButton("🌈 Colorize"), KeyboardButton("🎥 Video Note")],
            [KeyboardButton("⚙️ Model"), KeyboardButton("❓ Bantuan")],
        ],
        resize_keyboard=True,
    )


def force_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📢 Join @botunlverse", url=CHANNEL_URL)],
            [InlineKeyboardButton("✅ Saya sudah join", callback_data="check_sub")],
        ]
    )


def hd_scale_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("2x", callback_data="hdscale:2"),
                InlineKeyboardButton("4x", callback_data="hdscale:4"),
                InlineKeyboardButton("8x", callback_data="hdscale:8"),
            ],
            [InlineKeyboardButton("🔙 Menu", callback_data="menu")],
        ]
    )


async def is_subscribed(bot, user_id: int) -> bool:
    """Force join check.
    - Admin always pass.
    - If bot is not an admin in the channel (member list inaccessible),
      assume the user is subscribed (trust model for server).
    - Otherwise, use Telegram API.
    """
    if user_id == ADMIN_ID:
        return True
    try:
        member = await bot.get_chat_member(f"@{FORCE_CHANNEL}", user_id)
        return member.status in (
            ChatMemberStatus.MEMBER,
            ChatMemberStatus.ADMINISTRATOR,
            ChatMemberStatus.OWNER,
            ChatMemberStatus.RESTRICTED,
        )
    except BadRequest as e:
        msg = str(e).lower()
        # Bot not admin → cannot check, so trust user
        if "member_list" in msg or "not enough rights" in msg or "chat not found" in msg:
            return True
        logger.warning("getChatMember failed: %s", msg)
        return True  # optimistic fallback (user on their honor)
    except Exception as e:
        logger.warning("getChatMember failed: %s", e)
        return False


async def require_sub(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    if not user:
        return False
    ok = await is_subscribed(ctx.bot, user.id)
    if ok:
        return True
    text = (
        "🔒 *Akses Terkunci*\n\n"
        "Untuk pakai *Photo Video Studio*, wajib join channel dulu:\n"
        f"👉 [@{FORCE_CHANNEL}]({CHANNEL_URL})\n\n"
        "Setelah join, tekan *✅ Cek Langganan*."
    )
    if update.callback_query:
        await update.callback_query.answer("Belum subscribe!", show_alert=True)
        try:
            await update.callback_query.edit_message_text(
                text, parse_mode="Markdown", reply_markup=force_kb(), disable_web_page_preview=True
            )
        except Exception:
            await update.effective_message.reply_text(
                text, parse_mode="Markdown", reply_markup=force_kb(), disable_web_page_preview=True
            )
    else:
        await update.effective_message.reply_text(
            text, parse_mode="Markdown", reply_markup=force_kb(), disable_web_page_preview=True
        )
    return False


# ── Processing ────────────────────────────────────────────────────────

def rembg_bytes(data: bytes, model: str = "u2net") -> bytes:
    return remove(data, session=get_session(model))


def pil_to_bytes(img: Image.Image, fmt: str = "PNG", quality: int = 95) -> bytes:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def composite_bg(fg_png: bytes, bg_bytes: bytes) -> bytes:
    fg = Image.open(io.BytesIO(fg_png)).convert("RGBA")
    bg = Image.open(io.BytesIO(bg_bytes)).convert("RGBA")
    bg = bg.resize(fg.size, Image.LANCZOS)
    return pil_to_bytes(Image.alpha_composite(bg, fg), "PNG")


def hd_upscale(img_bytes: bytes, scale: int = 4, denoise: bool = False) -> bytes:
    """OpenCV Lanczos upscale (+ optional denoise)."""
    arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    h, w = arr.shape[:2]
    # cap output to avoid OOM (max ~8MP-ish side)
    max_side = 4096
    new_w, new_h = w * scale, h * scale
    if max(new_w, new_h) > max_side:
        r = max_side / max(new_w, new_h)
        new_w, new_h = int(new_w * r), int(new_h * r)
    up = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    if denoise:
        up = cv2.fastNlMeansDenoisingColored(up, None, 6, 6, 7, 21)
    # slight sharpen
    img = Image.fromarray(up)
    img = ImageEnhance.Sharpness(img).enhance(1.25)
    return pil_to_bytes(img, "JPEG", 92)


def denoise_image(img_bytes: bytes) -> bytes:
    arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    clean = cv2.fastNlMeansDenoisingColored(arr, None, 8, 8, 7, 21)
    img = Image.fromarray(clean)
    img = ImageEnhance.Sharpness(img).enhance(1.15)
    return pil_to_bytes(img, "JPEG", 92)


def colorize_image(img_bytes: bytes) -> bytes:
    """True B&W → warna (Zhang colorization) + post-process biar lebih natural/vivid."""
    cfg = Path(__file__).resolve().parent / "models" / "colorization_deploy_v2.prototxt"
    mdl = Path(__file__).resolve().parent / "models" / "colorization_release_v2.caffemodel"
    pts_path = Path(__file__).resolve().parent / "models" / "pts_in_hull.npy"

    net = cv2.dnn.readNetFromCaffe(str(cfg), str(mdl))
    pts = np.load(str(pts_path))
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]

    # Load & force grayscale L channel from true gray (better for B&W photos)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # Downscale very large images for speed, upscale colors back
    max_side = 1024
    w0, h0 = img.size
    scale = 1.0
    if max(w0, h0) > max_side:
        scale = max_side / max(w0, h0)
        img_small = img.resize((int(w0 * scale), int(h0 * scale)), Image.LANCZOS)
    else:
        img_small = img

    rgb = np.array(img_small).astype(np.float32) / 255.0
    # Convert to gray then back to RGB so L is pure luminance
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    rgb_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    h, w = rgb_gray.shape[:2]

    lab = cv2.cvtColor(rgb_gray, cv2.COLOR_RGB2Lab)
    L = lab[:, :, 0]
    L_rs = cv2.resize(L, (224, 224))
    L_rs = L_rs - 50  # mean-center as in Zhang paper

    net.setInput(cv2.dnn.blobFromImage(L_rs))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC)

    # Boost chroma (Zhang tends to be desaturated)
    ab = ab * 1.45
    # Soft clip ab to realistic Lab range
    ab = np.clip(ab, -110, 110)

    lab_out = np.concatenate([L[:, :, np.newaxis], ab], axis=2)
    out = np.clip(cv2.cvtColor(lab_out.astype(np.float32), cv2.COLOR_Lab2RGB) * 255, 0, 255).astype(np.uint8)

    # Upscale back to original if we downscaled
    if scale < 1.0:
        out = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
        # Re-apply original luminance at full res for sharper detail
        orig_gray = np.array(img.convert("L"), dtype=np.float32)
        out_lab = cv2.cvtColor(out.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
        out_lab[:, :, 0] = orig_gray * (100.0 / 255.0)
        out = np.clip(cv2.cvtColor(out_lab, cv2.COLOR_Lab2RGB) * 255, 0, 255).astype(np.uint8)

    pil = Image.fromarray(out)
    pil = ImageEnhance.Color(pil).enhance(1.25)
    pil = ImageEnhance.Contrast(pil).enhance(1.12)
    pil = ImageEnhance.Sharpness(pil).enhance(1.2)
    return pil_to_bytes(pil, "JPEG", 93)


def make_video_note_from_image(img_bytes: bytes, duration: float = 3.0) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, out = td / "in.png", td / "note.mp4"
        Image.open(io.BytesIO(img_bytes)).convert("RGB").save(src)
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", str(src), "-t", str(duration),
            "-vf", "scale=512:512:force_original_aspect_ratio=increase,crop=512:512",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return out.read_bytes()


def make_video_note_from_video(video_bytes: bytes, max_sec: float = 60.0) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src, out = td / "in.mp4", td / "note.mp4"
        src.write_bytes(video_bytes)
        cmd = [
            "ffmpeg", "-y", "-i", str(src), "-t", str(max_sec),
            "-vf", "scale=512:512:force_original_aspect_ratio=increase,crop=512:512",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", "-r", "25", str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return out.read_bytes()


# ── Handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    name = u.first_name if u else "teman"
    uid = u.id
    username = u.username or "-"
    is_admin = uid == ADMIN_ID

    # Check subscription automatically
    if not is_admin:
        sub = await is_subscribed(ctx.bot, uid)
        if not sub:
            await update.message.reply_text(
                f"🎬 *Photo Video Studio*\n\n"
                f"👋 Halo {name}!\n"
                f"🆔 ID: `{uid}`\n"
                f"👤 Username: @{username}\n\n"
                f"Untuk pakai bot ini, wajib join channel:\n"
                f"👉 [ @{FORCE_CHANNEL}]({CHANNEL_URL})\n\n"
                f"Setelah join, tekan tombol di bawah.",
                parse_mode="Markdown",
                reply_markup=force_kb(),
                disable_web_page_preview=True,
            )
            return

    ctx.user_data.clear()
    await update.message.reply_text(
        f"🎬 *Photo Video Studio*\n\n"
        f"👋 Halo {name}!\n"
        f"🆔 ID: `{uid}`\n"
        f"👤 Username: @{username}\n"
        f"✅ Langganan: Terverifikasi\n\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🖼️ Hapus / ganti background (AI rembg)\n"
        f"🔍 HD Upscale 2x / 4x / 8x\n"
        f"✨ Denoise (bersihkan noise)\n"
        f"🌈 Colorize (B&W → warna)\n"
        f"🎥 Video note (foto/video → bulat)\n\n"
        f"Pilih menu atau kirim foto langsung.",
        parse_mode="Markdown",
        reply_markup=main_kb(),
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await require_sub(update, ctx):
        return
    await update.message.reply_text(
        "❓ *Bantuan Fitur*\n\n"
        "🖼️ *Hapus BG* — foto → PNG transparan\n"
        "🎨 *Ganti BG* — foto subjek lalu foto background\n"
        "🔍 *HD Upscale* — pilih 2x/4x/8x lalu kirim foto\n"
        "✨ *Denoise* — bersihkan noise/grain\n"
        "🌈 *Colorize* — B&W → warna realistis\n"
        "🎥 *Video Note* — foto/video → bulat max 60d\n"
        "⚙️ *Model rembg* — u2net / u2netp / isnet\n\n"
        f"Wajib join @{FORCE_CHANNEL}.",
        parse_mode="Markdown",
        reply_markup=main_kb(),
    )


async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    if data == "check_sub":
        user = q.from_user
        is_ok = await is_subscribed(ctx.bot, user.id)
        if is_ok:
            name = user.first_name or "teman"
            uid = user.id
            username = user.username or "-"
            ctx.user_data.clear()
            try:
                await q.edit_message_text(
                    f"✅ Langganan terverifikasi!\n\n"
                    f"🎬 *Photo Video Studio*\n\n"
                    f"👋 Halo {name}!\n"
                    f"🆔 ID: `{uid}`\n"
                    f"👤 Username: @{username}\n"
                    f"✅ Langganan: Terverifikasi\n\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"Pilih menu atau kirim foto langsung.",
                    parse_mode="Markdown",
                )
            except Exception:
                pass
            await ctx.bot.send_message(
                uid,
                "🎬 Silakan pilih fitur:",
                reply_markup=main_kb(),
            )
        else:
            await q.answer("Masih belum join channel!", show_alert=True)
        return

    if data == "menu":
        ctx.user_data.clear()
        await q.edit_message_text("🏠 Menu — pilih dari keyboard bawah.")
        return

    if data.startswith("model:"):
        if not await require_sub(update, ctx):
            return
        model = data.split(":", 1)[1]
        ctx.user_data["model"] = model
        await q.edit_message_text(f"✅ Model rembg: `{model}`", parse_mode="Markdown")
        return

    if data.startswith("hdscale:"):
        if not await require_sub(update, ctx):
            return
        scale = int(data.split(":")[1])
        ctx.user_data["mode"] = "hd"
        ctx.user_data["hd_scale"] = scale
        await q.edit_message_text(f"🔍 HD *{scale}x* aktif.\n📷 Kirim foto sekarang.", parse_mode="Markdown")
        return


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    if not await require_sub(update, ctx):
        return
    text = update.message.text.strip()

    if text in ("🖼️ Hapus BG", "🖼️ Hapus Background", "/remove"):
        ctx.user_data["mode"] = "remove"
        await update.message.reply_text("📷 Kirim foto yang mau dihapus background-nya.")
        return
    if text in ("🎨 Ganti BG", "/replace"):
        ctx.user_data["mode"] = "replace_fg"
        ctx.user_data.pop("fg_bytes", None)
        await update.message.reply_text("📷 Kirim *foto subjek* dulu.", parse_mode="Markdown")
        return
    if text in ("🔍 HD Upscale", "/hd"):
        ctx.user_data["mode"] = "hd"
        await update.message.reply_text("🔍 Pilih skala HD:", reply_markup=hd_scale_kb())
        return
    if text in ("✨ Denoise", "/denoise"):
        ctx.user_data["mode"] = "denoise"
        await update.message.reply_text("✨ Mode *Denoise*. Kirim foto.", parse_mode="Markdown")
        return
    if text in ("🌈 Colorize", "/colorize"):
        ctx.user_data["mode"] = "colorize"
        await update.message.reply_text("🌈 Mode *Colorize*. Kirim foto (B&W / pucat).", parse_mode="Markdown")
        return
    if text in ("🎥 Video Note", "/vnote"):
        ctx.user_data["mode"] = "vnote"
        await update.message.reply_text("🎥 Kirim *foto* atau *video* → video note.", parse_mode="Markdown")
        return
    if text in ("⚙️ Model", "/model"):
        kb = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("u2net (general)", callback_data="model:u2net")],
                [InlineKeyboardButton("u2netp (portrait)", callback_data="model:u2netp")],
                [InlineKeyboardButton("isnet-general-use", callback_data="model:isnet-general-use")],
            ]
        )
        await update.message.reply_text("⚙️ Pilih model rembg:", reply_markup=kb)
        return
    if text in ("❓ Bantuan", "/help"):
        await cmd_help(update, ctx)
        return

    await update.message.reply_text("Pilih menu atau kirim foto.", reply_markup=main_kb())


async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return
    if not await require_sub(update, ctx):
        return

    mode = ctx.user_data.get("mode", "remove")
    model = ctx.user_data.get("model", "u2net")
    photo = update.message.photo[-1]
    f = await photo.get_file()
    bio = io.BytesIO()
    await f.download_to_memory(bio)
    bio.seek(0)
    img_bytes = bio.read()

    wait = await update.message.reply_text("🔄 Memproses...")
    loop = asyncio.get_event_loop()

    try:
        if mode == "vnote":
            note = await loop.run_in_executor(None, lambda: make_video_note_from_image(img_bytes, 3.0))
            await update.message.reply_video_note(video_note=io.BytesIO(note))
            await wait.edit_text("✅ Video note siap!")
            return

        if mode == "hd":
            scale = int(ctx.user_data.get("hd_scale", 4))
            out = await loop.run_in_executor(None, lambda: hd_upscale(img_bytes, scale=scale, denoise=False))
            await update.message.reply_document(
                document=io.BytesIO(out),
                filename=f"hd_{scale}x.jpg",
                caption=f"✅ HD Upscale *{scale}x*",
                parse_mode="Markdown",
            )
            await wait.delete()
            return

        if mode == "denoise":
            out = await loop.run_in_executor(None, lambda: denoise_image(img_bytes))
            await update.message.reply_photo(photo=io.BytesIO(out), caption="✅ Denoise selesai")
            await wait.delete()
            return

        if mode == "colorize":
            out = await loop.run_in_executor(None, lambda: colorize_image(img_bytes))
            await update.message.reply_photo(photo=io.BytesIO(out), caption="✅ Colorize selesai")
            await wait.delete()
            return

        # Ganti BG step 1: simpan subjek (tanpa bg)
        if mode == "replace_fg":
            fg = await loop.run_in_executor(None, lambda: rembg_bytes(img_bytes, model))
            ctx.user_data["fg_bytes"] = fg
            ctx.user_data["mode"] = "replace_bg"
            await wait.edit_text(
                "✅ Subjek siap (bg dihapus).\n"
                "Sekarang kirim *foto background* yang mau dipasang.",
                parse_mode="Markdown",
            )
            return

        # Ganti BG step 2: composite dengan background
        if mode == "replace_bg" or ctx.user_data.get("fg_bytes"):
            fg = ctx.user_data.get("fg_bytes")
            if not fg:
                ctx.user_data["mode"] = "replace_fg"
                await wait.edit_text("Kirim foto subjek dulu (tap 🎨 Ganti BG).")
                return
            out = await loop.run_in_executor(None, lambda: composite_bg(fg, img_bytes))
            await update.message.reply_photo(photo=io.BytesIO(out), caption="✅ Background diganti!")
            ctx.user_data.pop("fg_bytes", None)
            ctx.user_data["mode"] = "remove"
            await wait.delete()
            return

        # Default: hapus background
        out = await loop.run_in_executor(None, lambda: rembg_bytes(img_bytes, model))
        await update.message.reply_document(
            document=io.BytesIO(out),
            filename="no_bg.png",
            caption=f"✅ Background dihapus (`{model}`)",
            parse_mode="Markdown",
        )
        try:
            await update.message.reply_photo(photo=io.BytesIO(out), caption="Preview")
        except Exception:
            pass
        await wait.delete()
    except Exception as e:
        logger.exception("photo process failed")
        try:
            await wait.edit_text(f"❌ Gagal: {str(e)[:140]}")
        except Exception:
            pass


async def on_video(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    if not await require_sub(update, ctx):
        return
    ctx.user_data["mode"] = "vnote"
    vid = update.message.video or update.message.video_note
    if not vid:
        return
    wait = await update.message.reply_text("🔄 Convert ke video note...")
    try:
        f = await vid.get_file()
        bio = io.BytesIO()
        await f.download_to_memory(bio)
        bio.seek(0)
        data = bio.read()
        loop = asyncio.get_event_loop()
        note = await loop.run_in_executor(None, lambda: make_video_note_from_video(data, 60.0))
        await update.message.reply_video_note(video_note=io.BytesIO(note))
        await wait.edit_text("✅ Video note siap!")
    except Exception as e:
        logger.exception("video note failed")
        await wait.edit_text(f"❌ Gagal: {str(e)[:140]}")


async def on_error(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    logger.exception("Error: %s", ctx.error)


def main():
    if not BOT_TOKEN:
        raise SystemExit("BOT_TOKEN missing")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, on_video))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)
    logger.info("🎬 Photo Video Studio v2 starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
