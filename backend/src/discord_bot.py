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
OCR_SCALE_FACTOR = 1.5  # Reduced from 2.0 for faster processing
MAX_IMAGE_DIMENSION = 3840  # Downscale images larger than 4K

# Pre-compile regex patterns for better performance
NUMBER_PATTERN = re.compile(r'\D')
TEXT_PATTERN = re.compile(r'[^\w\s]')

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

# Initialize EasyOCR Reader (will be loaded during bot startup)
reader = None

def get_ocr_reader():
    """Get the initialized OCR reader."""
    global reader
    if reader is None:
        raise RuntimeError("OCR reader not initialized. This should not happen after bot startup.")
    return reader

async def initialize_ocr_reader():
    """Initialize OCR reader asynchronously during bot startup."""
    global reader
    if reader is None:
        loop = asyncio.get_running_loop()
        try:
            # Run blocking OCR initialization in executor
            reader = await loop.run_in_executor(
                None,
                lambda: easyocr.Reader(['en'], gpu=False)
            )
            print("✅ EasyOCR reader initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR reader: {str(e)}")

# Setup Discord Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

# ThreadPoolExecutor with proper cleanup - increased workers for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

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

def get_game_date(dt: datetime) -> str:
    """
    Calculate the game date based on 8am GMT+2 reset time.
    If current time is before 8am GMT+2, the game date is yesterday.
    
    Args:
        dt: datetime object (assumed to be in UTC or local time)
        
    Returns:
        ISO date string for the game day (YYYY-MM-DD)
    """
    from datetime import timedelta, timezone
    
    # Convert to GMT+2 (SAST - South African Standard Time)
    gmt_plus_2 = timezone(timedelta(hours=2))
    dt_gmt2 = dt.astimezone(gmt_plus_2)
    
    # If before 8am, the game day is yesterday
    if dt_gmt2.hour < 8:
        game_date = (dt_gmt2.date() - timedelta(days=1)).isoformat()
    else:
        game_date = dt_gmt2.date().isoformat()
    
    return game_date

def get_user_daily_history(username: str, days: int = 30) -> List[Dict]:
    """
    Get user's daily HighestToday history for line graphs.
    
    Args:
        username: Discord username
        days: Number of days to retrieve (default 30)
        
    Returns:
        List of dicts with GameDate and BestHighestToday
    """
    try:
        from datetime import timedelta
        
        response = table.query(
            KeyConditionExpression='PK = :pk AND begins_with(SK, :sk)',
            ExpressionAttributeValues={
                ':pk': f'USER#{username}',
                ':sk': 'DAILY#'
            },
            ScanIndexForward=False,  # Sort descending (newest first)
            Limit=days
        )
        
        return [{
            'date': item['GameDate'],
            'score': item['BestHighestToday']
        } for item in response.get('Items', [])]
    except Exception as e:
        print(f"❌ Error fetching user history: {str(e)}")
        return []

def get_club_weekly_history(club_name: str, weeks: int = 12) -> List[Dict]:
    """
    Get club's weekly MaxTotalPower history for line graphs.
    
    Args:
        club_name: Club name
        weeks: Number of weeks to retrieve (default 12)
        
    Returns:
        List of dicts with GameWeek and MaxTotalPower
    """
    try:
        response = table.query(
            KeyConditionExpression='PK = :pk AND begins_with(SK, :sk)',
            ExpressionAttributeValues={
                ':pk': f'CLUB#{club_name}',
                ':sk': 'WEEKLY#'
            },
            ScanIndexForward=False,  # Sort descending (newest first)
            Limit=weeks
        )
        
        return [{
            'week': item['GameWeek'],
            'power': item['MaxTotalPower']
        } for item in response.get('Items', [])]
    except Exception as e:
        print(f"❌ Error fetching club history: {str(e)}")
        return []

def get_game_week(dt: datetime) -> str:
    """
    Calculate the game week identifier based on Monday 8am GMT+2 reset.
    Week format: "YYYY-Wnn" (e.g., "2025-W49")
    
    Args:
        dt: datetime object
        
    Returns:
        ISO week string (YYYY-Wnn)
    """
    from datetime import timedelta, timezone
    
    # Convert to GMT+2
    gmt_plus_2 = timezone(timedelta(hours=2))
    dt_gmt2 = dt.astimezone(gmt_plus_2)
    
    # Adjust for 8am reset
    if dt_gmt2.hour < 8:
        adjusted_date = dt_gmt2.date() - timedelta(days=1)
    else:
        adjusted_date = dt_gmt2.date()
    
    # Adjust for Monday reset
    # ISO weekday: Monday=1, Sunday=7
    weekday = adjusted_date.isoweekday()
    if weekday == 1 and dt_gmt2.hour < 8:
        # If it's Monday before 8am, we're still in last week
        adjusted_date = adjusted_date - timedelta(days=1)
    
    # Get ISO week number
    year, week, _ = adjusted_date.isocalendar()
    return f"{year}-W{week:02d}"

def createSubmission(datetime_obj: datetime, club_name: str, highest_today: int,
                     weekly_score: int, total_power: int, username: str) -> dict:
    """
    Creates a submission entry in DynamoDB with new PK/SK schema.
    Also updates daily and weekly summaries for both user and club in real-time.
    
    Game Time Rules:
    - Daily reset: 8am GMT+2 (SAST)
    - Weekly reset: Monday 8am GMT+2 (SAST)
    
    New Schema:
    - Raw Submission: PK="USER#{username}", SK="SUBMISSION#{iso_datetime}"
    - User Daily: PK="USER#{username}", SK="DAILY#{game_date}"
    - Club Weekly: PK="CLUB#{club_name}", SK="WEEKLY#{week_id}"
    
    Args:
        datetime_obj: The datetime of the submission
        club_name: Name of the club
        highest_today: Highest score today (for user daily tracking)
        weekly_score: Weekly score (not used in summaries currently)
        total_power: Total power (for club weekly tracking)
        username: Discord username
        
    Returns:
        dict: Response with success status and any errors
    """
    try:
        iso_datetime = datetime_obj.isoformat()
        game_date = get_game_date(datetime_obj)
        game_week = get_game_week(datetime_obj)
        
        errors = []
        
        # ==========================================
        # 1. STORE RAW USER SUBMISSION
        # ==========================================
        user_submission = {
            'PK': f'USER#{username}',
            'SK': f'SUBMISSION#{iso_datetime}',
            'ClubName': club_name,
            'HighestToday': highest_today,
            'WeeklyScore': weekly_score,
            'TotalPower': total_power,
            'DateTime': iso_datetime,
            'GameDate': game_date,
            'GameWeek': game_week,
            'EntryType': 'UserSubmission'
        }
        
        try:
            table.put_item(Item=user_submission)
            print(f"✅ Stored user submission: {username} at {iso_datetime} (game date: {game_date})")
        except Exception as e:
            errors.append(f"User submission failed: {str(e)}")
            print(f"❌ Failed to store user submission: {str(e)}")
        
        # ==========================================
        # 2. UPDATE USER DAILY SUMMARY (Optimized with conditional update)
        # ==========================================
        try:
            # Use update_item with condition to avoid GET request
            table.update_item(
                Key={
                    'PK': f'USER#{username}',
                    'SK': f'DAILY#{game_date}'
                },
                UpdateExpression='SET BestHighestToday = if_not_exists(BestHighestToday, :zero), ClubName = :club, GameDate = :date, LastUpdated = :updated, EntryType = :type SET BestHighestToday = if_not_exists(BestHighestToday, :zero) SET BestHighestToday = :new_score',
                ConditionExpression='attribute_not_exists(BestHighestToday) OR BestHighestToday < :new_score',
                ExpressionAttributeValues={
                    ':new_score': highest_today,
                    ':zero': 0,
                    ':club': club_name,
                    ':date': game_date,
                    ':updated': iso_datetime,
                    ':type': 'UserDailySummary'
                }
            )
            print(f"✅ Updated user daily summary: {username} on {game_date} - new best {highest_today}")
        except table.meta.client.exceptions.ConditionalCheckFailedException:
            print(f"ℹ️ User daily not updated: score {highest_today} not higher than current best")
        except Exception as e:
            errors.append(f"User daily update failed: {str(e)}")
            print(f"❌ Failed to update user daily: {str(e)}")
        
        # ==========================================
        # 3. UPDATE CLUB WEEKLY SUMMARY (Optimized with conditional update)
        # ==========================================
        try:
            # Use update_item with condition to avoid GET request
            table.update_item(
                Key={
                    'PK': f'CLUB#{club_name}',
                    'SK': f'WEEKLY#{game_week}'
                },
                UpdateExpression='SET MaxTotalPower = if_not_exists(MaxTotalPower, :zero), GameWeek = :week, LastUpdated = :updated, EntryType = :type SET MaxTotalPower = :new_power',
                ConditionExpression='attribute_not_exists(MaxTotalPower) OR MaxTotalPower < :new_power',
                ExpressionAttributeValues={
                    ':new_power': total_power,
                    ':zero': 0,
                    ':week': game_week,
                    ':updated': iso_datetime,
                    ':type': 'ClubWeeklySummary'
                }
            )
            print(f"✅ Updated club weekly summary: {club_name} week {game_week} - new max {total_power}")
        except table.meta.client.exceptions.ConditionalCheckFailedException:
            print(f"ℹ️ Club weekly not updated: power {total_power} not higher than current max")
        except Exception as e:
            errors.append(f"Club weekly update failed: {str(e)}")
            print(f"❌ Failed to update club weekly: {str(e)}")
        
        # ==========================================
        # RETURN RESULTS
        # ==========================================
        if errors:
            return {
                'success': False,
                'partial': True,
                'errors': errors,
                'username': username,
                'club_name': club_name,
                'datetime': iso_datetime
            }
        
        return {
            'success': True,
            'username': username,
            'club_name': club_name,
            'datetime': iso_datetime,
            'game_date': game_date,
            'game_week': game_week
        }
        
    except Exception as e:
        print(f"❌ DynamoDB write error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def clean_text(text: str, is_number: bool = False) -> Union[int, str]:
    """
    Clean text extracted from OCR (optimized with pre-compiled regex).
    
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
        clean = NUMBER_PATTERN.sub('', text)
        return int(clean) if clean else 0
    
    clean = TEXT_PATTERN.sub('', text).strip()
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

def preprocess_roi(roi: np.ndarray, scale: float = OCR_SCALE_FACTOR) -> np.ndarray:
    """
    Preprocess ROI for OCR (optimized to avoid redundant operations).
    
    Args:
        roi: Input image ROI
        scale: Upscaling factor
        
    Returns:
        Preprocessed image ready for OCR
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(upscaled)
    return enhanced

def ocr_static_roi(roi: np.ndarray, is_number: bool) -> str:
    """Simple OCR for the static box (optimized)."""
    if roi is None or roi.size == 0:
        return ""
    
    try:
        ocr_reader = get_ocr_reader()
        enhanced = preprocess_roi(roi)
        allowlist = '0123456789' if is_number else None
        results = ocr_reader.readtext(enhanced, detail=0, allowlist=allowlist, paragraph=True)
        return " ".join(results)
    except Exception as e:
        print(f"❌ OCR error in static ROI: {str(e)}")
        return ""

def scan_zone(image: np.ndarray, zone_bbox: tuple) -> List[OCREntry]:
    """
    Scans a dynamic zone and returns all text with coordinates (optimized).
    
    Returns:
        List of OCREntry objects
    """
    x, y, w, h = zone_bbox
    if w <= 0 or h <= 0:
        return []
    
    try:
        ocr_reader = get_ocr_reader()
        roi = image[y:y+h, x:x+w]
        enhanced = preprocess_roi(roi)
        
        results = ocr_reader.readtext(enhanced, detail=1)
        
        entries = []
        for (bbox, text, prob) in results:
            local_cx = (bbox[0][0] + bbox[1][0]) / 2
            local_cy = (bbox[0][1] + bbox[2][1]) / 2
            global_cx = (local_cx / OCR_SCALE_FACTOR) + x
            global_cy = (local_cy / OCR_SCALE_FACTOR) + y
            
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

def downscale_if_needed(image: np.ndarray) -> np.ndarray:
    """
    Downscale image if it's too large for efficient processing.
    
    Args:
        image: Input image
        
    Returns:
        Downscaled image if needed, otherwise original
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    if max_dim > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image

def process_static_score(img: np.ndarray) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
    """Process static center score ROI."""
    config = STATIC_ROIS["CENTER_SCORE_ROI"]
    roi, coords = get_anchored_roi(img, config['rect'], config['anchor_x'], config['anchor_y'])
    raw_score = ocr_static_roi(roi, is_number=True)
    daily_high = clean_text(raw_score, is_number=True)
    return daily_high, coords

def process_power_zone(img: np.ndarray) -> Tuple[int, Tuple[int, int, int, int]]:
    """Process power zone (bottom left)."""
    h, w = img.shape[:2]
    scale_x, scale_y = calculate_scale_from_image(img)
    
    # Use temporary file for debug images instead of working directory
    debug_path = None
    if debug:
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as tmp:
            debug_path = tmp.name
    
    debug_img = img.copy() if debug else None

    # PARALLEL PROCESSING: Process all three zones concurrently
    loop = asyncio.new_event_loop()
    
    # Submit all tasks to executor for parallel processing
    future_score = loop.run_in_executor(executor, process_static_score, img)
    future_power = loop.run_in_executor(executor, process_power_zone, img)
    future_leaderboard = loop.run_in_executor(executor, process_leaderboard_zone, img, scale_x, scale_y)
    
    # Wait for all results
    daily_high, score_coords = loop.run_until_complete(future_score)
    found_power, power_zone = loop.run_until_complete(future_power)
    lb_name, lb_score, lb_zone = loop.run_until_complete(future_leaderboard)
    
    loop.close()

    # Debug visualization if requested
    if debug and debug_img is not None:
        # Draw static ROI
        if score_coords:
            rx, ry, rw, rh = score_coords
            cv2.rectangle(debug_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
        
        # Draw power zone
        zp_x, zp_y, zp_w, zp_h = power_zone
        cv2.rectangle(debug_img, (zp_x, zp_y), (zp_x+zp_w, zp_y+zp_h), (0, 0, 255), 2)
        
        # Draw leaderboard zone
        zl_x, zl_y, zl_w, zl_h = lb_zone
        cv2.rectangle(debug_img, (zl_x, zl_y), (zl_x+zl_w, zl_y+zl_h), (255, 0, 0), 2)
        
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
    zp_x, zp_y = 0, int(h * 0.85)
    zp_w, zp_h = int(w * 0.40), h - zp_y
    
    power_entries = scan_zone(img, (zp_x, zp_y, zp_w, zp_h))
    valid_power = [e for e in power_entries if e.number > MIN_POWER_THRESHOLD]
    
    found_power = 0
    if valid_power:
        valid_power.sort(key=lambda k: k.cx)
        found_power = valid_power[0].number
    
    return found_power, (zp_x, zp_y, zp_w, zp_h)

def process_leaderboard_zone(img: np.ndarray, scale_x: float, scale_y: float) -> Tuple[str, int, Tuple[int, int, int, int]]:
    """Process leaderboard zone (bottom right)."""
    h, w = img.shape[:2]
    zl_x, zl_y = int(w * 0.65), int(h * 0.60)
    zl_w, zl_h = w - zl_x, h - zl_y
    
    lb_entries = scan_zone(img, (zl_x, zl_y, zl_w, zl_h))
    lb_name, lb_score = find_leaderboard_name_and_score(lb_entries, scale_x, scale_y)
    
    return lb_name, lb_score, (zl_x, zl_y, zl_w, zl_h)

def analyze_screenshot(image_bytes: bytes, debug: bool = False) -> AnalysisResult:
    """Analyze screenshot and extract game data (optimized with parallel processing)."""
    if not image_bytes:
        raise ValueError("No image data provided")
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image - file may be corrupted or invalid format")

    # Downscale if needed for faster processing
    img = downscale_if_needed(img)
    
    h, w = img.shape[:2]