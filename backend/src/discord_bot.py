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
import boto3
from datetime import datetime
from decimal import Decimal
import uuid
import atexit
from typing import Optional, Union, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time
import json
import tempfile

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
DYNAMODB_TABLE_NAME = os.getenv('DYNAMODB_TABLE_NAME')

# Validate required environment variables
required_env_vars = {
    'DISCORD_BOT_TOKEN': DISCORD_TOKEN,
    'AWS_ACCESS_KEY_ID': AWS_ACCESS_KEY_ID,
    'AWS_SECRET_ACCESS_KEY': AWS_SECRET_ACCESS_KEY,
    'AWS_REGION': AWS_REGION,
    'DYNAMODB_TABLE_NAME': DYNAMODB_TABLE_NAME
}

missing_vars = [k for k, v in required_env_vars.items() if not v]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize DynamoDB client with error handling
try:
    dynamodb = boto3.resource(
        'dynamodb',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    table = dynamodb.Table(DYNAMODB_TABLE_NAME)
    # Test connection
    table.load()
    print(f"✅ Successfully connected to DynamoDB table: {DYNAMODB_TABLE_NAME}")
except Exception as e:
    raise ConnectionError(f"Failed to connect to DynamoDB: {str(e)}")

# --- CONFIGURATION CONSTANTS ---
REF_WIDTH = 2556
REF_HEIGHT = 1179
MIN_POWER_THRESHOLD = 10000
MIN_LEADERBOARD_SCORE_THRESHOLD = 10000
LEADERBOARD_RIGHT_COLUMN_TOLERANCE_PX = 50
LEADERBOARD_NAME_VERTICAL_DISTANCE_MAX = 120
LEADERBOARD_NAME_HORIZONTAL_TOLERANCE = 100
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
IMAGE_DOWNLOAD_TIMEOUT = 30.0  # seconds

# Rate limiting: user_id -> list of timestamps
rate_limit_tracker: Dict[int, List[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds
RATE_LIMIT_MAX_REQUESTS = 10  # max submissions per hour

STATIC_ROIS = {
    "CENTER_SCORE_ROI": {
        "rect": (980, 875, 300, 60),
        "anchor_x": "center",
        "anchor_y": "top",
        "type": "number"
    }
}

# --- DATA CLASSES ---
@dataclass
class OCREntry:
    """Structured data for OCR detection results."""
    raw_text: str
    number: int
    cx: float  # Center X coordinate
    cy: float  # Center Y coordinate

@dataclass
class AnalysisResult:
    """Structured result from screenshot analysis."""
    daily_high: int
    total_power: int
    rank_name: str
    rank_score: int

# Initialize EasyOCR Reader with error handling (lazy loading)
reader = None

def get_ocr_reader():
    """Lazy load OCR reader to prevent blocking bot startup."""
    global reader
    if reader is None:
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            print("✅ EasyOCR reader initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR reader: {str(e)}")
    return reader

# Setup Discord Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

# ThreadPoolExecutor with proper cleanup
executor = ThreadPoolExecutor(max_workers=2)

def cleanup_executor():
    """Cleanup executor on shutdown."""
    executor.shutdown(wait=True)
    print("✅ ThreadPoolExecutor shut down cleanly")

atexit.register(cleanup_executor)

# Store activated channels with their associated roles
# Key: (guild_id, channel_id), Value: role_name
activated_channels = {}

# Persistent storage file for activated channels
CHANNELS_STORAGE_FILE = "activated_channels.json"

def load_activated_channels():
    """Load activated channels from persistent storage."""
    global activated_channels
    try:
        if os.path.exists(CHANNELS_STORAGE_FILE):
            with open(CHANNELS_STORAGE_FILE, 'r') as f:
                data = json.load(f)
                # Convert string keys back to tuples
                activated_channels = {
                    tuple(map(int, k.split(','))): v 
                    for k, v in data.items()
                }
                print(f"✅ Loaded {len(activated_channels)} activated channels from storage")
    except Exception as e:
        print(f"⚠️ Could not load activated channels: {str(e)}")
        activated_channels = {}

def save_activated_channels():
    """Save activated channels to persistent storage."""
    try:
        # Convert tuple keys to strings for JSON serialization
        data = {
            f"{guild_id},{channel_id}": role_name
            for (guild_id, channel_id), role_name in activated_channels.items()
        }
        with open(CHANNELS_STORAGE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {len(activated_channels)} activated channels to storage")
    except Exception as e:
        print(f"⚠️ Could not save activated channels: {str(e)}")

# Load channels on startup
load_activated_channels()

# --- HELPER FUNCTIONS ---

def check_rate_limit(user_id: int) -> Tuple[bool, int]:
    """
    Check if user has exceeded rate limit.
    
    Returns:
        Tuple of (is_allowed, seconds_until_reset)
    """
    now = time.time()
    # Clean old timestamps
    rate_limit_tracker[user_id] = [
        ts for ts in rate_limit_tracker[user_id]
        if now - ts < RATE_LIMIT_WINDOW
    ]
    
    if len(rate_limit_tracker[user_id]) >= RATE_LIMIT_MAX_REQUESTS:
        oldest_timestamp = min(rate_limit_tracker[user_id])
        seconds_until_reset = int(RATE_LIMIT_WINDOW - (now - oldest_timestamp))
        return False, seconds_until_reset
    
    rate_limit_tracker[user_id].append(now)
    return True, 0

def createSubmission(datetime_obj: datetime, club_name: str, highest_today: int,
                     weekly_score: int, total_power: int, user_id: int, username: str) -> dict:
    """
    Creates a submission entry in DynamoDB.
    
    Args:
        datetime_obj: The datetime of the submission
        club_name: Name of the club (Partition Key)
        highest_today: Highest score today
        weekly_score: Weekly score
        total_power: Total power
        user_id: Discord user ID
        username: Discord username
        
    Returns:
        dict: Response from DynamoDB put_item operation
    """
    try:
        # Use timestamp as sort key for easy querying, with UUID for uniqueness
        timestamp = int(datetime_obj.timestamp())
        submission_id = str(uuid.uuid4())
        sort_key = f"{timestamp}#{submission_id}"
        
        # Prepare item for DynamoDB
        item = {
            'ClubName': club_name,
            'DateTimestamp': sort_key,  # Sortable by timestamp
            'SubmissionId': submission_id,
            'Timestamp': timestamp,
            'DateTime': datetime_obj.isoformat(),
            'HighestToday': highest_today,
            'WeeklyScore': weekly_score,
            'TotalPower': total_power,
            'UserId': str(user_id),
            'Username': username,
            'EntryType': 'Submission'
        }
        
        # Put item into DynamoDB with error handling
        response = table.put_item(Item=item)
        
        return {
            'success': True,
            'response': response,
            'club_name': club_name,
            'sort_key': sort_key
        }
    except Exception as e:
        print(f"❌ DynamoDB write error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def clean_text(text: str, is_number: bool = False) -> Union[int, str]:
    """
    Clean text extracted from OCR.
    
    Args:
        text: Raw text from OCR
        is_number: Whether to extract numeric value
        
    Returns:
        int if is_number=True, str otherwise
    """
    if not text:
        return 0 if is_number else ""
    
    if is_number:
        text = text.replace('S', '5').replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
        clean = re.sub(r'\D', '', text)
        return int(clean) if clean else 0
    
    clean = re.sub(r'[^\w\s]', '', text).strip()
    return clean

async def get_guild_nick(interaction: discord.Interaction) -> Optional[str]:
    """Return the server-specific nickname for the user, or None if not available."""
    if interaction.guild is None:
        return None
    member = interaction.guild.get_member(interaction.user.id)
    try:
        if member is None:
            member = await interaction.guild.fetch_member(interaction.user.id)
    except Exception:
        return None
    return member.nick if member is not None else None

def calculate_scale_from_image(image: np.ndarray) -> Tuple[float, float]:
    """
    Calculate scale factors for different aspect ratios.
    
    Returns:
        Tuple of (scale_x, scale_y) for proper ROI scaling
    """
    h, w = image.shape[:2]
    scale_x = w / REF_WIDTH
    scale_y = h / REF_HEIGHT
    return scale_x, scale_y

def get_anchored_roi(image: np.ndarray, ref_rect: tuple, anchor_x: str, anchor_y: str) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    """
    Calculates static ROI coordinates based on reference resolution with aspect ratio support.
    
    Returns:
        Tuple of (roi_image, (x, y, w, h))
    """
    if image is None:
        return None, (0, 0, 0, 0)
    
    ref_x, ref_y, ref_w, ref_h = ref_rect
    h, w = image.shape[:2]
    
    scale_x, scale_y = calculate_scale_from_image(image)
    
    new_w = int(ref_w * scale_x)
    new_h = int(ref_h * scale_y)
    
    # X Anchor
    if anchor_x == "center":
        ref_center_x = REF_WIDTH / 2
        offset = ref_x - ref_center_x
        new_x = int((w / 2) + (offset * scale_x))
    else:
        new_x = int(ref_x * scale_x)

    # Y Anchor
    new_y = int(ref_y * scale_y)

    new_x = max(0, min(new_x, w - new_w))
    new_y = max(0, min(new_y, h - new_h))
    
    return image[new_y:new_y+new_h, new_x:new_x+new_w], (new_x, new_y, new_w, new_h)

def ocr_static_roi(roi: np.ndarray, is_number: bool) -> str:
    """Simple OCR for the static box."""
    if roi is None or roi.size == 0:
        return ""
    
    try:
        ocr_reader = get_ocr_reader()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        scale = 2
        upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(upscaled)
        allowlist = '0123456789' if is_number else None
        results = ocr_reader.readtext(enhanced, detail=0, allowlist=allowlist, paragraph=True)
        return " ".join(results)
    except Exception as e:
        print(f"❌ OCR error in static ROI: {str(e)}")
        return ""

def scan_zone(image: np.ndarray, zone_bbox: tuple) -> List[OCREntry]:
    """
    Scans a dynamic zone and returns all text with coordinates.
    
    Returns:
        List of OCREntry objects
    """
    x, y, w, h = zone_bbox
    if w <= 0 or h <= 0:
        return []
    
    try:
        ocr_reader = get_ocr_reader()
        roi = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        scale = 2
        upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        results = ocr_reader.readtext(upscaled, detail=1)
        
        entries = []
        for (bbox, text, prob) in results:
            local_cx = (bbox[0][0] + bbox[1][0]) / 2
            local_cy = (bbox[0][1] + bbox[2][1]) / 2
            global_cx = (local_cx / scale) + x
            global_cy = (local_cy / scale) + y
            
            entries.append(OCREntry(
                raw_text=text,
                number=clean_text(text, is_number=True),
                cx=global_cx,
                cy=global_cy
            ))
        return entries
    except Exception as e:
        print(f"❌ OCR error in zone scan: {str(e)}")
        return []

def find_leaderboard_name_and_score(lb_entries: List[OCREntry], scale_x: float, scale_y: float) -> Tuple[str, int]:
    """
    Find leaderboard name and score using resolution-adaptive thresholds.
    
    Args:
        lb_entries: List of OCR entries from leaderboard zone
        scale_x: Horizontal scale factor
        scale_y: Vertical scale factor
        
    Returns:
        Tuple of (name, score)
    """
    lb_numbers = [e for e in lb_entries if e.number > MIN_LEADERBOARD_SCORE_THRESHOLD]
    
    if not lb_numbers:
        return "Unknown", 0
    
    # Adaptive thresholds based on resolution
    right_col_tolerance = int(LEADERBOARD_RIGHT_COLUMN_TOLERANCE_PX * scale_x)
    name_vertical_max = int(LEADERBOARD_NAME_VERTICAL_DISTANCE_MAX * scale_y)
    name_horizontal_tolerance = int(LEADERBOARD_NAME_HORIZONTAL_TOLERANCE * scale_x)
    
    # Find rightmost column
    max_x = max(e.cx for e in lb_numbers)
    right_column = [e for e in lb_numbers if e.cx >= (max_x - right_col_tolerance)]
    
    # Find bottommost in that column
    right_column.sort(key=lambda k: k.cy, reverse=True)
    best_entry = right_column[0]
    lb_score = best_entry.number
    
    # Find name closest above
    min_dist = float('inf')
    lb_name = "Unknown"
    
    for entry in lb_entries:
        if entry == best_entry:
            continue
        
        txt = entry.raw_text.lower()
        # Filter out UI elements
        if any(keyword in txt for keyword in ["start", "quick", "play", "join", "button"]):
            continue
        
        dy = best_entry.cy - entry.cy  # Positive if entry is above score
        dx = abs(best_entry.cx - entry.cx)
        
        if 0 < dy < name_vertical_max and dx < name_horizontal_tolerance:
            if dy < min_dist:
                min_dist = dy
                lb_name = clean_text(entry.raw_text, is_number=False)
    
    return lb_name, lb_score

def analyze_screenshot(image_bytes: bytes, debug: bool = False) -> AnalysisResult:
    """Analyze screenshot and extract game data."""
    if not image_bytes:
        raise ValueError("No image data provided")
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image - file may be corrupted or invalid format")

    h, w = img.shape[:2]
    scale_x, scale_y = calculate_scale_from_image(img)
    
    # Use temporary file for debug images instead of working directory
    debug_path = None
    if debug:
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as tmp:
            debug_path = tmp.name
    
    debug_img = img.copy() if debug else None

    # 1. CENTER HIGH SCORE -> USE STATIC ROI
    config = STATIC_ROIS["CENTER_SCORE_ROI"]
    roi, coords = get_anchored_roi(img, config['rect'], config['anchor_x'], config['anchor_y'])
    
    raw_score = ocr_static_roi(roi, is_number=True)
    daily_high = clean_text(raw_score, is_number=True)
    
    if debug and coords:
        rx, ry, rw, rh = coords
        cv2.rectangle(debug_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)

    # 2. TOTAL POWER -> USE SMART ZONE (Bottom Left)
    zp_x, zp_y = 0, int(h * 0.85)
    zp_w, zp_h = int(w * 0.40), h - zp_y
    
    power_entries = scan_zone(img, (zp_x, zp_y, zp_w, zp_h))
    valid_power = [e for e in power_entries if e.number > MIN_POWER_THRESHOLD]
    
    found_power = 0
    if valid_power:
        valid_power.sort(key=lambda k: k.cx)
        found_power = valid_power[0].number
        
        if debug:
            cv2.circle(debug_img, (int(valid_power[0].cx), int(valid_power[0].cy)), 10, (0, 0, 255), -1)

    if debug:
        cv2.rectangle(debug_img, (zp_x, zp_y), (zp_x+zp_w, zp_y+zp_h), (0, 0, 255), 2)

    # 3. LEADERBOARD -> USE SMART ZONE (Bottom Right)
    zl_x, zl_y = int(w * 0.65), int(h * 0.60)
    zl_w, zl_h = w - zl_x, h - zl_y
    
    lb_entries = scan_zone(img, (zl_x, zl_y, zl_w, zl_h))
    lb_name, lb_score = find_leaderboard_name_and_score(lb_entries, scale_x, scale_y)
    
    if debug:
        cv2.rectangle(debug_img, (zl_x, zl_y), (zl_x+zl_w, zl_y+zl_h), (255, 0, 0), 2)
        # Mark detected score
        for entry in lb_entries:
            if entry.number == lb_score:
                cv2.circle(debug_img, (int(entry.cx), int(entry.cy)), 10, (255, 0, 0), -1)
                break
        
        try:
            cv2.imwrite(debug_path, debug_img)
        except Exception as e:
            print(f"❌ Failed to save debug image: {str(e)}")

    result = AnalysisResult(
        daily_high=daily_high,
        total_power=found_power,
        rank_name=lb_name,
        rank_score=lb_score
    )
    
    # Store debug path in result if available
    if debug and debug_path:
        result.debug_path = debug_path  # type: ignore
    
    return result

@bot.event
async def on_ready():
    print(f'✅ {bot.user} has connected!')
    try:
        await bot.tree.sync()
        print("✅ Slash commands synced.")
    except Exception as e:
        print(f"❌ Sync failed: {e}")

@bot.tree.command(name="activatecontestsubmissions", description="Activate contest submissions in this channel (Moderator only)")
@app_commands.checks.has_permissions(moderate_members=True)
async def activate_channel(interaction: discord.Interaction, role: discord.Role):
    """Activate contest submissions for this channel with a specific role."""
    if interaction.guild is None:
        await interaction.response.send_message("❌ This command can only be used in a server.", ephemeral=True)
        return
    
    channel_key = (interaction.guild.id, interaction.channel.id)
    
    if channel_key in activated_channels:
        old_role = activated_channels[channel_key]
        activated_channels[channel_key] = role.name
        save_activated_channels()
        await interaction.response.send_message(
            f"✅ Contest submissions in this channel have been updated!\n"
            f"**Previous Role:** {old_role}\n"
            f"**New Role:** {role.mention}\n"
            f"Users can now use `/submit` and `/calibrate` commands here.",
            ephemeral=False
        )
    else:
        activated_channels[channel_key] = role.name
        save_activated_channels()
        await interaction.response.send_message(
            f"✅ Contest submissions are now **activated** in this channel!\n"
            f"**Associated Role:** {role.mention}\n"
            f"Users can now use `/submit` and `/calibrate` commands here.",
            ephemeral=False
        )

@bot.tree.command(name="deactivatecontestsubmissions", description="Deactivate contest submissions in this channel (Moderator only)")
@app_commands.checks.has_permissions(moderate_members=True)
async def deactivate_channel(interaction: discord.Interaction):
    """Deactivate contest submissions for this channel."""
    if interaction.guild is None:
        await interaction.response.send_message("❌ This command can only be used in a server.", ephemeral=True)
        return
    
    channel_key = (interaction.guild.id, interaction.channel.id)
    
    if channel_key not in activated_channels:
        await interaction.response.send_message(
            "❌ Contest submissions are not currently activated in this channel.",
            ephemeral=True
        )
        return
    
    role_name = activated_channels[channel_key]
    del activated_channels[channel_key]
    save_activated_channels()
    
    await interaction.response.send_message(
        f"✅ Contest submissions have been **deactivated** in this channel!\n"
        f"**Previous Role:** {role_name}\n"
        f"Users can no longer use `/submit` and `/calibrate` commands here.",
        ephemeral=False
    )

@activate_channel.error
async def activate_channel_error(interaction: discord.Interaction, error):
    """Handle permission errors for the activate command."""
    if isinstance(error, app_commands.errors.MissingPermissions):
        await interaction.response.send_message(
            "❌ You need moderator permissions to manage contest submissions.",
            ephemeral=True
        )

@deactivate_channel.error
async def deactivate_channel_error(interaction: discord.Interaction, error):
    """Handle permission errors for the deactivate command."""
    if isinstance(error, app_commands.errors.MissingPermissions):
        await interaction.response.send_message(
            "❌ You need moderator permissions to manage contest submissions.",
            ephemeral=True
        )

@bot.tree.command(name="submit", description="Submit a screenshot for scoring")
async def submit(interaction: discord.Interaction, image: discord.Attachment):
    """Submit a screenshot for contest scoring."""
    # Check rate limit
    is_allowed, seconds_until_reset = check_rate_limit(interaction.user.id)
    if not is_allowed:
        minutes = seconds_until_reset // 60
        await interaction.response.send_message(
            f"⏱️ You've reached the submission limit. Please try again in {minutes} minutes.",
            ephemeral=True
        )
        return
    
    # Check if channel is activated
    if interaction.guild is None:
        await interaction.response.send_message("❌ This command can only be used in a server.", ephemeral=True)
        return
    
    channel_key = (interaction.guild.id, interaction.channel.id)
    if channel_key not in activated_channels:
        await interaction.response.send_message(
            "❌ Contest submissions are not activated in this channel.\n"
            "A moderator needs to use `/activatecontestsubmissions @role` first.",
            ephemeral=True
        )
        return
    
    # Validate image
    if not image.content_type or not image.content_type.startswith('image/'):
        await interaction.response.send_message("❌ Please upload a valid image file.", ephemeral=True)
        return
    
    if image.size > MAX_IMAGE_SIZE:
        await interaction.response.send_message(
            f"❌ Image too large. Maximum size is {MAX_IMAGE_SIZE // (1024*1024)}MB.",
            ephemeral=True
        )
        return
    
    channel_role = activated_channels[channel_key]
    
    await interaction.response.defer()
    
    try:
        # Download image with timeout
        image_data = await asyncio.wait_for(image.read(), timeout=IMAGE_DOWNLOAD_TIMEOUT)
        
        if not image_data:
            await interaction.followup.send("❌ Failed to download image data.", ephemeral=True)
            return
        
        loop = asyncio.get_running_loop()
        result: AnalysisResult = await loop.run_in_executor(executor, analyze_screenshot, image_data, False)
        
        embed = discord.Embed(title="🎮 Contest Entry Processed", color=discord.Color.gold())
        embed.set_author(name=interaction.user.display_name, icon_url=interaction.user.display_avatar.url)
        
        embed.add_field(
            name="🏆 Daily High Score",
            value=f"```yaml\n{result.daily_high:,}```",
            inline=True
        )
        embed.add_field(
            name="⚡ Total Power",
            value=f"```yaml\n{result.total_power:,}```",
            inline=True
        )
        embed.add_field(name="\u200b", value="**Leaderboard Validation**", inline=False)
        embed.add_field(
            name="👤 Name Detected",
            value=f"{result.rank_name}",
            inline=True
        )
        embed.add_field(
            name="📊 Leaderboard Score",
            value=f"{result.rank_score:,}",
            inline=True
        )
        embed.set_thumbnail(url=image.url)
        
        # Compare detected leaderboard name with server nickname
        detected_name = clean_text(result.rank_name, is_number=False)

        guild_nick = await get_guild_nick(interaction)
        if guild_nick is None:
            match_text = "⚠️ No server nickname set"
            embed.add_field(
                name="⚠️ Validation Issue",
                value="You must set a server nickname that matches your in-game name to submit scores.",
                inline=False
            )
        else:
            nick_clean = clean_text(guild_nick, is_number=False)
            names_match = nick_clean.lower() == detected_name.lower()
            match_text = "✅ Match" if names_match else "❌ Mismatch"
            
            if not names_match:
                embed.add_field(
                    name="⚠️ Name Mismatch",
                    value=f"Your server nickname `{guild_nick}` doesn't match the detected name `{result.rank_name}`. Please update your nickname to match your in-game name.",
                    inline=False
                )

        embed.add_field(name="👥 Server Nickname", value=guild_nick or "—", inline=True)
        embed.add_field(name="🔎 Detected Name", value=result.rank_name or "—", inline=True)
        embed.add_field(name="✅ Nickname Match", value=match_text, inline=False)
        embed.add_field(name="🏷️ Channel Role", value=channel_role, inline=False)

        # Create submission in DynamoDB if names match AND scores are valid
        should_save = False
        save_reason = ""
        
        if guild_nick is None:
            save_reason = "❌ Not saved - no server nickname set"
        else:
            nick_clean = clean_text(guild_nick, is_number=False)
            if nick_clean.lower() != detected_name.lower():
                save_reason = "❌ Not saved - nickname mismatch"
            elif result.daily_high == 0 and result.rank_score == 0:
                save_reason = "❌ Not saved - no valid scores detected"
            else:
                should_save = True
        
        if should_save:
            submission_result = createSubmission(
                datetime_obj=datetime.now(),
                club_name=channel_role,
                highest_today=result.daily_high,
                weekly_score=result.rank_score,
                total_power=result.total_power,
                user_id=interaction.user.id,
                username=interaction.user.name
            )
            
            if submission_result['success']:
                embed.add_field(
                    name="💾 Database",
                    value="✅ Submission saved successfully!",
                    inline=False
                )
            else:
                embed.add_field(
                    name="💾 Database",
                    value=f"⚠️ Failed to save: {submission_result.get('error', 'Unknown error')}",
                    inline=False
                )
        else:
            embed.add_field(
                name="💾 Database",
                value=save_reason,
                inline=False
            )

        await interaction.followup.send(embed=embed)

    except asyncio.TimeoutError:
        await interaction.followup.send("❌ Image download timed out. Please try again.", ephemeral=True)
    except ValueError as e:
        await interaction.followup.send(f"❌ {str(e)}", ephemeral=True)
    except Exception as e:
        print(f"❌ Unexpected error in submit: {str(e)}")
        await interaction.followup.send(
            "❌ An error occurred while processing your image. Please ensure you've uploaded a clear screenshot and try again.",
            ephemeral=True
        )

@bot.tree.command(name="calibrate", description="Debug view: Green=Static, Red=PowerZone, Blue=LeaderboardZone")
async def calibrate(interaction: discord.Interaction, image: discord.Attachment):
    """Show debug visualization of OCR regions."""
    if interaction.guild is None:
        await interaction.response.send_message("❌ This command can only be used in a server.", ephemeral=True)
        return
    
    channel_key = (interaction.guild.id, interaction.channel.id)
    if channel_key not in activated_channels:
        await interaction.response.send_message(
            "❌ Contest submissions are not activated in this channel.\n"
            "A moderator needs to use `/activatecontestsubmissions @role` first.",
            ephemeral=True
        )
        return
    
    # Validate image
    if not image.content_type or not image.content_type.startswith('image/'):
        await interaction.response.send_message("❌ Please upload a valid image file.", ephemeral=True)
        return
    
    if image.size > MAX_IMAGE_SIZE:
        await interaction.response.send_message(
            f"❌ Image too large. Maximum size is {MAX_IMAGE_SIZE // (1024*1024)}MB.",
            ephemeral=True
        )
        return
    
    await interaction.response.defer()
    
    debug_path = None
    try:
        image_data = await asyncio.wait_for(image.read(), timeout=IMAGE_DOWNLOAD_TIMEOUT)
        
        if not image_data:
            await interaction.followup.send("❌ Failed to download image data.", ephemeral=True)
            return
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, analyze_screenshot, image_data, True)
        
        debug_path = getattr(result, 'debug_path', None)
        
        if debug_path and os.path.exists(debug_path):
            f = discord.File(debug_path, filename="calibration.png")
            embed = discord.Embed(title="🎯 Calibration Debug", color=discord.Color.blurple())
            embed.description = (
                "**Legend:**\n"
                "🟢 Green: Daily High Score ROI\n"
                "🔴 Red: Total Power Zone\n"
                "🔵 Blue: Leaderboard Zone"
            )
            embed.set_image(url="attachment://calibration.png")
            await interaction.followup.send(embed=embed, file=f)
        else:
            await interaction.followup.send("❌ Failed to generate debug image.", ephemeral=True)
    except asyncio.TimeoutError:
        await interaction.followup.send("❌ Image download timed out. Please try again.", ephemeral=True)
    except Exception as e:
        print(f"❌ Error in calibrate: {str(e)}")
        await interaction.followup.send("❌ An error occurred during calibration. Please try again.", ephemeral=True)
    finally:
        # Clean up temporary file
        if debug_path and os.path.exists(debug_path):
            try:
                os.remove(debug_path)
            except Exception as e:
                print(f"⚠️ Could not delete debug file: {str(e)}")

def main():
    """Main entry point for the bot."""
    if not DISCORD_TOKEN:
        print("❌ Error: DISCORD_BOT_TOKEN not found in environment variables")
        return
    
    try:
        bot.run(DISCORD_TOKEN)
    except KeyboardInterrupt:
        print("\n✅ Bot shutdown requested")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
    finally:
        cleanup_executor()

if __name__ == "__main__":
    main()