import discord
from discord import app_commands
from discord.ext import commands
import os
from dotenv import load_dotenv
import cv2
import numpy as np
import easyocr
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# --- CONFIGURATION FOR STATIC ROI ---
REF_WIDTH = 2556
REF_HEIGHT = 1179

STATIC_ROIS = {
    "CENTER_SCORE_ROI": {
        "rect":  (980, 875, 300, 60), 
        "anchor_x": "center", 
        "anchor_y": "top",
        "type": "number"
    }
}

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False) 

# Setup Discord Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

executor = ThreadPoolExecutor(max_workers=2)

# --- HELPER FUNCTIONS ---

def clean_text(text: str, is_number: bool = False) -> str:
    if not text: return ""
    if is_number:
        text = text.replace('S', '5').replace('O', '0').replace('o', '0')
        clean = re.sub(r'\D', '', text)
        return int(clean) if clean else 0
    clean = re.sub(r'[^\w\s]', '', text).strip()
    return clean

def get_anchored_roi(image: np.ndarray, ref_rect: tuple, anchor_x: str, anchor_y: str) -> tuple:
    """Calculates static ROI coordinates based on reference resolution."""
    if image is None: return None, (0,0,0,0)
    ref_x, ref_y, ref_w, ref_h = ref_rect
    h, w = image.shape[:2]
    
    scale = h / REF_HEIGHT
    new_w = int(ref_w * scale)
    new_h = int(ref_h * scale)
    
    # X Anchor
    if anchor_x == "center":
        ref_center_x = REF_WIDTH / 2
        offset = ref_x - ref_center_x
        new_x = int((w / 2) + (offset * scale))
    else: # Fallback/Left
        new_x = int(ref_x * scale)

    # Y Anchor
    if anchor_y == "top":
        new_y = int(ref_y * scale)
    else:
        new_y = int(ref_y * scale)

    new_x = max(0, new_x)
    new_y = max(0, new_y)
    
    return image[new_y:new_y+new_h, new_x:new_x+new_w], (new_x, new_y, new_w, new_h)

def ocr_static_roi(roi: np.ndarray, is_number: bool) -> str:
    """Simple OCR for the static box"""
    if roi is None or roi.size == 0: return ""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scale = 2
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(upscaled)
    allowlist = '0123456789' if is_number else None
    try:
        results = reader.readtext(enhanced, detail=0, allowlist=allowlist, paragraph=True)
        return " ".join(results)
    except: return ""

def scan_zone(image: np.ndarray, zone_bbox: tuple) -> list:
    """Scans a dynamic zone and returns all text with coordinates."""
    x, y, w, h = zone_bbox
    if w <= 0 or h <= 0: return []
    
    roi = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scale = 2
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    results = reader.readtext(upscaled, detail=1)
    
    entries = []
    for (bbox, text, prob) in results:
        local_cx = (bbox[0][0] + bbox[1][0]) / 2
        local_cy = (bbox[0][1] + bbox[2][1]) / 2
        global_cx = (local_cx / scale) + x
        global_cy = (local_cy / scale) + y
        
        entries.append({
            "raw_text": text,
            "number": clean_text(text, is_number=True),
            "cx": global_cx,
            "cy": global_cy
        })
    return entries

# --- MAIN LOGIC ---

def analyze_screenshot(image_bytes: bytes, debug: bool = False) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Could not decode image")

    h, w = img.shape[:2]
    results = {}
    debug_img = img.copy() if debug else None

    # ---------------------------------------------------------
    # 1. CENTER HIGH SCORE -> USE STATIC ROI
    # ---------------------------------------------------------
    config = STATIC_ROIS["CENTER_SCORE_ROI"]
    roi, coords = get_anchored_roi(img, config['rect'], config['anchor_x'], config['anchor_y'])
    
    raw_score = ocr_static_roi(roi, is_number=True)
    results['daily_high'] = clean_text(raw_score, is_number=True)
    
    if debug:
        rx, ry, rw, rh = coords
        cv2.rectangle(debug_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)

    # ---------------------------------------------------------
    # 2. TOTAL POWER -> USE SMART ZONE (Bottom Left)
    # ---------------------------------------------------------
    zp_x, zp_y = 0, int(h * 0.85)
    zp_w, zp_h = int(w * 0.40), h - zp_y
    
    power_entries = scan_zone(img, (zp_x, zp_y, zp_w, zp_h))
    valid_power = [e for e in power_entries if e['number'] > 10000]
    
    found_power = 0
    if valid_power:
        # Sort by X position (Leftmost first)
        valid_power.sort(key=lambda k: k['cx'])
        found_power = valid_power[0]['number']
        
        if debug:
            cv2.circle(debug_img, (int(valid_power[0]['cx']), int(valid_power[0]['cy'])), 10, (0, 0, 255), -1)

    results['total_power'] = found_power
    if debug:
        cv2.rectangle(debug_img, (zp_x, zp_y), (zp_x+zp_w, zp_y+zp_h), (0, 0, 255), 2)

    # ---------------------------------------------------------
    # 3. LEADERBOARD -> USE SMART ZONE (Bottom Right)
    # Logic: Rightmost Column -> Bottommost Number -> Name Above
    # ---------------------------------------------------------
    zl_x, zl_y = int(w * 0.65), int(h * 0.60)
    zl_w, zl_h = w - zl_x, h - zl_y
    
    lb_entries = scan_zone(img, (zl_x, zl_y, zl_w, zl_h))
    lb_numbers = [e for e in lb_entries if e['number'] > 10000]
    
    lb_score = 0
    lb_name = "Unknown"
    
    if lb_numbers:
        # Step 1: Identify the Rightmost Column
        # Find the maximum X coordinate (cx) among all numbers found
        max_x = max(e['cx'] for e in lb_numbers)
        
        # Filter: Keep numbers that are within 50px of this rightmost edge
        right_column = [e for e in lb_numbers if e['cx'] >= (max_x - 50)]
        
        # Step 2: Find the Bottommost Number in that column
        # Sort by Y descending (highest Y = bottom)
        right_column.sort(key=lambda k: k['cy'], reverse=True)
        best_entry = right_column[0]
        lb_score = best_entry['number']
        
        # Step 3: Find Name Closest Above
        min_dist = float('inf')
        score_cy = best_entry['cy']
        score_cx = best_entry['cx']
        
        for entry in lb_entries:
            if entry == best_entry: continue
            txt = entry['raw_text'].lower()
            if "start" in txt or "quick" in txt: continue
            
            # Distance calculations
            dy = score_cy - entry['cy'] # Positive if entry is ABOVE score
            dx = abs(score_cx - entry['cx']) # Horizontal alignment difference
            
            # Criteria: 
            # 1. Strictly Above: dy must be positive (0 < dy < 120px)
            # 2. Roughly aligned: dx < 100px
            if 0 < dy < 120 and dx < 100:
                if dy < min_dist:
                    min_dist = dy
                    lb_name = clean_text(entry['raw_text'], is_number=False)
                    
        if debug:
             # Mark the detected score
             cv2.circle(debug_img, (int(best_entry['cx']), int(best_entry['cy'])), 10, (255, 0, 0), -1)

    results['rank_name'] = lb_name
    results['rank_score'] = lb_score
    
    if debug:
        cv2.rectangle(debug_img, (zl_x, zl_y), (zl_x+zl_w, zl_y+zl_h), (255, 0, 0), 2)
        cv2.imwrite("debug_output.png", debug_img)

    return results

@bot.event
async def on_ready():
    print(f'{bot.user} has connected!')
    try:
        await bot.tree.sync()
        print("Slash commands synced.")
    except Exception as e:
        print(f"Sync failed: {e}")

@bot.tree.command(name="submit", description="Submit a screenshot for scoring")
async def submit(interaction: discord.Interaction, image: discord.Attachment):
    await interaction.response.defer()
    
    if not image.content_type.startswith('image/'):
        await interaction.followup.send("❌ Please upload a valid image file.", ephemeral=True)
        return

    try:
        image_data = await image.read()
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(executor, analyze_screenshot, image_data, False)
        
        embed = discord.Embed(title="🎮 Contest Entry Processed", color=discord.Color.gold())
        embed.set_author(name=interaction.user.display_name, icon_url=interaction.user.display_avatar.url)
        
        embed.add_field(
            name="🏆 Daily High Score", 
            value=f"```yaml\n{data.get('daily_high', 0):,}```", 
            inline=True
        )
        embed.add_field(
            name="⚡ Total Power", 
            value=f"```yaml\n{data.get('total_power', 0):,}```", 
            inline=True
        )
        embed.add_field(name="\u200b", value="**Leaderboard Validation**", inline=False)
        embed.add_field(
            name="👤 Name Detected", 
            value=f"{data.get('rank_name', 'Unknown')}", 
            inline=True
        )
        embed.add_field(
            name="📊 Leaderboard Score", 
            value=f"{data.get('rank_score', 0):,}", 
            inline=True
        )
        embed.set_thumbnail(url=image.url)
        
        await interaction.followup.send(embed=embed)

    except Exception as e:
        await interaction.followup.send(f"❌ Processing Error: {str(e)}", ephemeral=True)

@bot.tree.command(name="calibrate", description="Debug view: Green=Static, Red=PowerZone, Blue=LeaderboardZone")
async def calibrate(interaction: discord.Interaction, image: discord.Attachment):
    await interaction.response.defer()
    try:
        image_data = await image.read()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, analyze_screenshot, image_data, True)
        
        if os.path.exists("debug_output.png"):
            f = discord.File("debug_output.png", filename="calibration.png")
            embed = discord.Embed(title="🎯 Calibration Debug", color=discord.Color.blurple())
            embed.set_image(url="attachment://calibration.png")
            await interaction.followup.send(embed=embed, file=f)
            os.remove("debug_output.png")
        else:
            await interaction.followup.send("Failed to generate debug image.")
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {str(e)}", ephemeral=True)

def main():
    if not DISCORD_TOKEN:
        print("Error: DISCORD_BOT_TOKEN not found")
        return
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()