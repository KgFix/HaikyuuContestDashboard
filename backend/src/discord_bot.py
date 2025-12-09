import discord
from discord import app_commands
from discord.ext import commands, tasks
import os
from dotenv import load_dotenv
import cv2
import numpy as np
import easyocr
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import boto3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import atexit
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time
import json
import tempfile

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
load_dotenv()

required_env_vars = {
    'DISCORD_BOT_TOKEN': os.getenv('DISCORD_BOT_TOKEN'),
    'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
    'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
    'AWS_REGION': os.getenv('AWS_REGION'),
    'DYNAMODB_TABLE_NAME': os.getenv('DYNAMODB_TABLE_NAME')
}

missing_vars = [k for k, v in required_env_vars.items() if not v]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

DISCORD_TOKEN = required_env_vars['DISCORD_BOT_TOKEN']
AWS_ACCESS_KEY_ID = required_env_vars['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = required_env_vars['AWS_SECRET_ACCESS_KEY']
AWS_REGION = required_env_vars['AWS_REGION']
DYNAMODB_TABLE_NAME = required_env_vars['DYNAMODB_TABLE_NAME']

# ============================================================================
# DYNAMODB CONNECTION
# ============================================================================
try:
    dynamodb = boto3.resource(
        'dynamodb',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    table = dynamodb.Table(DYNAMODB_TABLE_NAME)
    table.load()
    logger.info(f"Connected to DynamoDB table: {DYNAMODB_TABLE_NAME}")
except Exception as e:
    raise ConnectionError(f"Failed to connect to DynamoDB: {str(e)}")

# ============================================================================
# CONSTANTS - Image Processing
# ============================================================================
REF_WIDTH = 2556
REF_HEIGHT = 1179
MIN_POWER_THRESHOLD = 10000
MIN_LEADERBOARD_SCORE_THRESHOLD = 10000
LEADERBOARD_RIGHT_COLUMN_TOLERANCE_PX = 50
LEADERBOARD_NAME_VERTICAL_DISTANCE_MAX = 120
LEADERBOARD_NAME_HORIZONTAL_TOLERANCE = 100
MAX_IMAGE_SIZE = 10 * 1024 * 1024
IMAGE_DOWNLOAD_TIMEOUT = 30.0
OCR_SCALE_FACTOR = 1.5
MAX_IMAGE_DIMENSION = 3840

# Image zone ratios
POWER_ZONE_Y_RATIO = 0.85
POWER_ZONE_WIDTH_RATIO = 0.40
LEADERBOARD_ZONE_X_RATIO = 0.65
LEADERBOARD_ZONE_Y_RATIO = 0.60

# ============================================================================
# CONSTANTS - DynamoDB Keys
# ============================================================================
DDB_USER_PREFIX = 'USER#'
DDB_CLUB_PREFIX = 'CLUB#'
DDB_SUBMISSION_PREFIX = 'SUBMISSION#'
DDB_DAILY_PREFIX = 'DAILY#'
DDB_ACTIVITY_PREFIX = 'ACTIVITY#'

# ============================================================================
# CONSTANTS - Rate Limiting
# ============================================================================
RATE_LIMIT_WINDOW = 3600  # 1 hour
RATE_LIMIT_MAX_REQUESTS = 10

# ============================================================================
# CONSTANTS - Reminders
# ============================================================================
REMINDER_DAYS = [0, 2, 4, 5, 6]  # Mon, Wed, Fri, Sat, Sun
REMINDER_HOUR = 20  # 8 PM GMT+2

# ============================================================================
# REGEX PATTERNS
# ============================================================================
NUMBER_PATTERN = re.compile(r'\D')
TEXT_PATTERN = re.compile(r'[^\w\s]')

# ============================================================================
# STATIC ROI DEFINITIONS
# ============================================================================
STATIC_ROIS = {
    "CENTER_SCORE_ROI": {
        "rect": (980, 875, 300, 60),
        "anchor_x": "center",
        "anchor_y": "top",
        "type": "number"
    }
}

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class OCREntry:
    """Represents a single OCR detection result"""
    raw_text: str
    number: int
    cx: float  # center x coordinate
    cy: float  # center y coordinate

@dataclass
class AnalysisResult:
    """Results from screenshot analysis"""
    daily_high: int
    total_power: int
    rank_name: str
    rank_score: int

# ============================================================================
# GLOBAL STATE
# ============================================================================
reader = None  # EasyOCR reader instance
executor = ThreadPoolExecutor(max_workers=4)
rate_limit_tracker: Dict[int, List[float]] = defaultdict(list)
last_reminder_date = None

# Persistent storage
CHANNELS_STORAGE_FILE = os.getenv('CHANNELS_STORAGE_FILE', 'activated_channels.json')
REMINDERS_STORAGE_FILE = os.getenv('REMINDERS_STORAGE_FILE', 'reminder_channels.json')
REMINDER_ROLES_FILE = os.getenv('REMINDER_ROLES_FILE', 'reminder_roles.json')

activated_channels: Dict[Tuple[int, int], str] = {}  # (guild_id, channel_id) -> club_name
reminder_channels: Dict[Tuple[int, int], str] = {}  # Legacy, kept for compatibility
reminder_roles: Dict[Tuple[int, int], Dict] = {}  # (guild_id, channel_id) -> {club_name, role_id}

# ============================================================================
# DISCORD BOT SETUP
# ============================================================================
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='/', intents=intents)

# ============================================================================
# UTILITY FUNCTIONS - Persistence
# ============================================================================
def load_channels(filename: str) -> Dict[Tuple[int, int], str]:
    """Load channel data from JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                channels = {
                    tuple(map(int, k.split(','))): v 
                    for k, v in data.items()
                }
                logger.info(f"Loaded {len(channels)} channels from {filename}")
                return channels
    except Exception as e:
        logger.warning(f"Could not load channels from {filename}: {str(e)}")
    return {}

def save_channels(channels: Dict[Tuple[int, int], str], filename: str) -> None:
    """Save channel data to JSON file"""
    try:
        data = {
            f"{guild_id},{channel_id}": value
            for (guild_id, channel_id), value in channels.items()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(channels)} channels to {filename}")
    except Exception as e:
        logger.error(f"Could not save channels to {filename}: {str(e)}")

def load_reminder_roles(filename: str) -> Dict[Tuple[int, int], Dict]:
    """Load reminder role data from JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                roles = {
                    tuple(map(int, k.split(','))): v 
                    for k, v in data.items()
                }
                logger.info(f"Loaded {len(roles)} reminder roles from {filename}")
                return roles
    except Exception as e:
        logger.warning(f"Could not load reminder roles from {filename}: {str(e)}")
    return {}

def save_reminder_roles(roles: Dict[Tuple[int, int], Dict], filename: str) -> None:
    """Save reminder role data to JSON file"""
    try:
        data = {
            f"{guild_id},{channel_id}": value
            for (guild_id, channel_id), value in roles.items()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(roles)} reminder roles to {filename}")
    except Exception as e:
        logger.error(f"Could not save reminder roles to {filename}: {str(e)}")

# Load persistent data on startup
activated_channels = load_channels(CHANNELS_STORAGE_FILE)
reminder_channels = load_channels(REMINDERS_STORAGE_FILE)
reminder_roles = load_reminder_roles(REMINDER_ROLES_FILE)

# ============================================================================
# UTILITY FUNCTIONS - Rate Limiting
# ============================================================================
def check_rate_limit(user_id: int) -> Tuple[bool, int]:
    """
    Check if user has exceeded rate limit
    Returns: (is_allowed, seconds_until_reset)
    """
    now = time.time()
    
    # Clean old timestamps
    rate_limit_tracker[user_id] = [
        ts for ts in rate_limit_tracker[user_id]
        if now - ts < RATE_LIMIT_WINDOW
    ]
    
    # Check if over limit
    if len(rate_limit_tracker[user_id]) >= RATE_LIMIT_MAX_REQUESTS:
        oldest_timestamp = min(rate_limit_tracker[user_id])
        seconds_until_reset = int(RATE_LIMIT_WINDOW - (now - oldest_timestamp))
        return False, seconds_until_reset
    
    # Add current request
    rate_limit_tracker[user_id].append(now)
    return True, 0

# ============================================================================
# UTILITY FUNCTIONS - Game Time Management
# ============================================================================
def get_game_date(dt: datetime) -> str:
    """
    Convert datetime to game date (GMT+2, resets at 8 AM)
    Returns: ISO date string (YYYY-MM-DD)
    """
    gmt_plus_2 = timezone(timedelta(hours=2))
    dt_gmt2 = dt.astimezone(gmt_plus_2)
    
    # If before 8 AM, it's still the previous day's game date
    if dt_gmt2.hour < 8:
        game_date = (dt_gmt2.date() - timedelta(days=1)).isoformat()
    else:
        game_date = dt_gmt2.date().isoformat()
    
    return game_date

def get_game_week(dt: datetime) -> str:
    """
    Convert datetime to game week (GMT+2, week starts Monday at 8 AM)
    Returns: ISO week string (YYYY-Www)
    """
    gmt_plus_2 = timezone(timedelta(hours=2))
    dt_gmt2 = dt.astimezone(gmt_plus_2)
    
    # Adjust for 8 AM reset
    if dt_gmt2.hour < 8:
        adjusted_date = dt_gmt2.date() - timedelta(days=1)
    else:
        adjusted_date = dt_gmt2.date()
    
    # Handle week boundary
    weekday = adjusted_date.isoweekday()
    if weekday == 1 and dt_gmt2.hour < 8:
        adjusted_date = adjusted_date - timedelta(days=1)
    
    year, week, _ = adjusted_date.isocalendar()
    return f"{year}-W{week:02d}"

# ============================================================================
# UTILITY FUNCTIONS - Discord
# ============================================================================
async def get_guild_nick(interaction: discord.Interaction) -> Optional[str]:
    """Get user's guild nickname, falling back to display name"""
    if interaction.guild is None:
        return None
    
    member = interaction.guild.get_member(interaction.user.id)
    
    if member is None:
        try:
            member = await interaction.guild.fetch_member(interaction.user.id)
        except Exception:
            return None
    
    return member.nick if member and member.nick else None

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================
def create_submission(
    datetime_obj: datetime,
    club_name: str,
    highest_today: int,
    weekly_score: int,
    total_power: int,
    username: str
) -> dict:
    """
    Create submission entries in DynamoDB for user and club tracking
    Creates 4 types of records:
    1. User Submission (full record)
    2. User Daily Summary (best score for the day)
    3. Club Daily Summary (highest power for the day)
    4. Club Daily Activity (tracks which users submitted)
    """
    try:
        iso_datetime = datetime_obj.isoformat()
        game_date = get_game_date(datetime_obj)
        game_week = get_game_week(datetime_obj)
        
        errors = []
        
        # 1. User Submission - Full submission record
        user_submission = {
            'PK': f'{DDB_USER_PREFIX}{username}',
            'SK': f'{DDB_SUBMISSION_PREFIX}{iso_datetime}',
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
            logger.info(f"Stored user submission: {username} - {highest_today:,}")
        except Exception as e:
            logger.error(f"User submission failed for {username}: {str(e)}")
            errors.append(f"User submission failed: {str(e)}")
        
        # 2. User Daily Summary - Best score for the day
        try:
            table.update_item(
                Key={
                    'PK': f'{DDB_USER_PREFIX}{username}',
                    'SK': f'{DDB_DAILY_PREFIX}{game_date}'
                },
                UpdateExpression=(
                    'SET BestHighestToday = if_not_exists(BestHighestToday, :score), '
                    'ClubName = :club, GameDate = :date, GameWeek = :week, '
                    'LastUpdated = :updated, EntryType = :type'
                ),
                ConditionExpression=(
                    'attribute_not_exists(BestHighestToday) OR BestHighestToday < :score'
                ),
                ExpressionAttributeValues={
                    ':score': highest_today,
                    ':club': club_name,
                    ':date': game_date,
                    ':week': game_week,
                    ':updated': iso_datetime,
                    ':type': 'UserDailySummary'
                }
            )
            logger.info(f"Updated user daily summary: {username}")
        except table.meta.client.exceptions.ConditionalCheckFailedException:
            logger.debug(f"User daily not updated (no improvement): {username}")
        except Exception as e:
            logger.error(f"User daily summary failed for {username}: {str(e)}")
            errors.append(f"User daily summary failed: {str(e)}")
        
        # 3. Club Daily Summary - Highest power for the day
        try:
            table.update_item(
                Key={
                    'PK': f'{DDB_CLUB_PREFIX}{club_name}',
                    'SK': f'{DDB_DAILY_PREFIX}{game_date}'
                },
                UpdateExpression=(
                    'SET MaxTotalPower = if_not_exists(MaxTotalPower, :power), '
                    'GameDate = :date, GameWeek = :week, '
                    'LastUpdated = :updated, EntryType = :type'
                ),
                ConditionExpression=(
                    'attribute_not_exists(MaxTotalPower) OR MaxTotalPower < :power'
                ),
                ExpressionAttributeValues={
                    ':power': total_power,
                    ':date': game_date,
                    ':week': game_week,
                    ':updated': iso_datetime,
                    ':type': 'ClubDailySummary'
                }
            )
            logger.info(f"Updated club daily summary: {club_name}")
        except table.meta.client.exceptions.ConditionalCheckFailedException:
            logger.debug(f"Club daily not updated (no improvement): {club_name}")
        except Exception as e:
            logger.error(f"Club daily summary failed for {club_name}: {str(e)}")
            errors.append(f"Club daily summary failed: {str(e)}")
        
        # 4. Club Daily Activity - Track which users submitted
        try:
            response = table.get_item(
                Key={
                    'PK': f'{DDB_CLUB_PREFIX}{club_name}',
                    'SK': f'{DDB_ACTIVITY_PREFIX}{game_date}'
                }
            )
            
            existing_users = {}
            if 'Item' in response and 'Users' in response['Item']:
                existing_users = response['Item']['Users']
            
            # Update user's best score for the day
            current_score = existing_users.get(username, 0)
            if highest_today > current_score:
                existing_users[username] = highest_today
            
            table.put_item(
                Item={
                    'PK': f'{DDB_CLUB_PREFIX}{club_name}',
                    'SK': f'{DDB_ACTIVITY_PREFIX}{game_date}',
                    'Users': existing_users,
                    'GameDate': game_date,
                    'GameWeek': game_week,
                    'LastUpdated': iso_datetime,
                    'EntryType': 'ClubDailyActivity'
                }
            )
            logger.info(f"Updated club activity: {club_name} ({len(existing_users)} users)")
        except Exception as e:
            logger.error(f"Club activity tracking failed for {club_name}: {str(e)}")
            errors.append(f"Club activity tracking failed: {str(e)}")
        
        # Return result
        if errors:
            return {'success': False, 'partial': True, 'errors': errors}
        
        return {
            'success': True,
            'username': username,
            'club_name': club_name,
            'game_date': game_date,
            'game_week': game_week
        }
        
    except Exception as e:
        logger.error(f"Submission creation failed: {str(e)}")
        return {'success': False, 'error': str(e)}

async def get_members_who_havent_submitted_by_role(
    guild: discord.Guild,
    role: discord.Role,
    club_name: str,
    game_date: str
) -> List[discord.Member]:
    """Get list of members with specific role who haven't submitted for the day"""
    try:
        # Get club activity for the day
        response = table.get_item(
            Key={
                'PK': f'{DDB_CLUB_PREFIX}{club_name}',
                'SK': f'{DDB_ACTIVITY_PREFIX}{game_date}'
            }
        )
        
        # Extract usernames who have submitted
        submitted_users = set()
        if 'Item' in response and 'Users' in response['Item']:
            submitted_users = set(response['Item']['Users'].keys())
        
        # Find members with role who haven't submitted
        members_to_ping = []
        for member in role.members:
            if member.bot:
                continue
            
            member_name = member.nick if member.nick else member.display_name
            
            if member_name not in submitted_users:
                members_to_ping.append(member)
        
        return members_to_ping
        
    except Exception as e:
        logger.error(f"Error getting non-submitters by role: {str(e)}")
        return []

# ============================================================================
# IMAGE PROCESSING FUNCTIONS - Text Cleaning
# ============================================================================
def clean_text(text: str, is_number: bool = False) -> int | str:
    """
    Clean OCR text output
    - For numbers: removes non-digits, handles common OCR errors (S->5, O->0, etc)
    - For text: removes special characters
    """
    if not text:
        return 0 if is_number else ""
    
    if is_number:
        # Fix common OCR mistakes
        text = text.replace('S', '5').replace('O', '0').replace('o', '0')
        text = text.replace('I', '1').replace('l', '1')
        clean = NUMBER_PATTERN.sub('', text)
        return int(clean) if clean else 0
    
    # Clean text (keep alphanumeric and spaces)
    clean = TEXT_PATTERN.sub('', text).strip()
    return clean

# ============================================================================
# IMAGE PROCESSING FUNCTIONS - OCR
# ============================================================================
def get_ocr_reader():
    """Get the global OCR reader instance"""
    global reader
    if reader is None:
        raise RuntimeError("OCR reader not initialized. Call initialize_ocr_reader() first.")
    return reader

async def initialize_ocr_reader():
    """Initialize the EasyOCR reader asynchronously"""
    global reader
    if reader is None:
        try:
            logger.info("Initializing EasyOCR...")
            loop = asyncio.get_running_loop()
            reader = await loop.run_in_executor(
                executor,
                lambda: easyocr.Reader(['en'], gpu=False)
            )
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"EasyOCR initialization failed: {str(e)}")
            raise

# ============================================================================
# IMAGE PROCESSING FUNCTIONS - ROI & Preprocessing
# ============================================================================
def calculate_scale_from_image(image: np.ndarray) -> Tuple[float, float]:
    """Calculate scale factors relative to reference dimensions"""
    h, w = image.shape[:2]
    scale_x = w / REF_WIDTH
    scale_y = h / REF_HEIGHT
    return scale_x, scale_y

def get_anchored_roi(
    image: np.ndarray,
    ref_rect: tuple,
    anchor_x: str,
    anchor_y: str
) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    """
    Extract ROI from image using anchored positioning
    Anchors: 'center' or standard (0,0 = top-left)
    Returns: (roi_image, (x, y, w, h))
    """
    if image is None:
        return None, (0, 0, 0, 0)
    
    ref_x, ref_y, ref_w, ref_h = ref_rect
    h, w = image.shape[:2]
    
    scale_x, scale_y = calculate_scale_from_image(image)
    
    # Calculate scaled dimensions
    new_w = int(ref_w * scale_x)
    new_h = int(ref_h * scale_y)
    
    # Calculate x position based on anchor
    if anchor_x == "center":
        ref_center_x = REF_WIDTH / 2
        offset = ref_x - ref_center_x
        new_x = int((w / 2) + (offset * scale_x))
    else:
        new_x = int(ref_x * scale_x)
    
    new_y = int(ref_y * scale_y)
    
    # Clamp to image bounds
    new_x = max(0, min(new_x, w - new_w))
    new_y = max(0, min(new_y, h - new_h))
    
    return image[new_y:new_y+new_h, new_x:new_x+new_w], (new_x, new_y, new_w, new_h)

def preprocess_roi(roi: np.ndarray, scale: float = OCR_SCALE_FACTOR) -> np.ndarray:
    """
    Preprocess ROI for better OCR accuracy
    - Convert to grayscale
    - Upscale for better recognition
    - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(upscaled)
    return enhanced

def downscale_if_needed(image: np.ndarray) -> np.ndarray:
    """Downscale image if it exceeds maximum dimension"""
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    if max_dim > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        logger.info(f"Downscaling image from {w}x{h} to {new_w}x{new_h}")
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image

# ============================================================================
# IMAGE PROCESSING FUNCTIONS - OCR Operations
# ============================================================================
def ocr_static_roi(roi: np.ndarray, is_number: bool) -> str:
    """Perform OCR on a static ROI"""
    if roi is None or roi.size == 0:
        return ""
    
    try:
        ocr_reader = get_ocr_reader()
        enhanced = preprocess_roi(roi)
        allowlist = '0123456789' if is_number else None
        results = ocr_reader.readtext(
            enhanced,
            detail=0,
            allowlist=allowlist,
            paragraph=True
        )
        return " ".join(results)
    except Exception as e:
        logger.error(f"OCR error on static ROI: {str(e)}")
        return ""

def scan_zone(image: np.ndarray, zone_bbox: tuple) -> List[OCREntry]:
    """
    Scan a zone of the image and return all OCR detections
    Returns list of OCREntry objects with global coordinates
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
            # Calculate center position in ROI coordinates
            local_cx = (bbox[0][0] + bbox[1][0]) / 2
            local_cy = (bbox[0][1] + bbox[2][1]) / 2
            
            # Convert to global image coordinates
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
        logger.error(f"Zone scan error: {str(e)}")
        return []

# ============================================================================
# IMAGE PROCESSING FUNCTIONS - Specialized Analysis
# ============================================================================
def find_leaderboard_name_and_score(
    lb_entries: List[OCREntry],
    scale_x: float,
    scale_y: float
) -> Tuple[str, int]:
    """
    Find player's leaderboard name and score from OCR entries
    Algorithm:
    1. Find all large numbers (potential scores)
    2. Identify rightmost column (where scores are)
    3. Take bottom-most score (player's score)
    4. Find closest text above it (player's name)
    """
    # Filter for valid leaderboard numbers
    lb_numbers = [e for e in lb_entries if e.number > MIN_LEADERBOARD_SCORE_THRESHOLD]
    
    if not lb_numbers:
        return "Unknown", 0
    
    # Calculate scaled tolerances
    right_col_tolerance = int(LEADERBOARD_RIGHT_COLUMN_TOLERANCE_PX * scale_x)
    name_vertical_max = int(LEADERBOARD_NAME_VERTICAL_DISTANCE_MAX * scale_y)
    name_horizontal_tolerance = int(LEADERBOARD_NAME_HORIZONTAL_TOLERANCE * scale_x)
    
    # Find rightmost column of numbers
    max_x = max(e.cx for e in lb_numbers)
    right_column = [e for e in lb_numbers if e.cx >= (max_x - right_col_tolerance)]
    
    # Get bottom-most entry (player's score)
    right_column.sort(key=lambda k: k.cy, reverse=True)
    best_entry = right_column[0]
    lb_score = best_entry.number
    
    # Find closest name above the score
    min_dist = float('inf')
    lb_name = "Unknown"
    
    for entry in lb_entries:
        if entry == best_entry:
            continue
        
        # Filter out UI elements
        txt = entry.raw_text.lower()
        if any(keyword in txt for keyword in ["start", "quick", "play", "join", "button"]):
            continue
        
        # Check if text is above score and within horizontal tolerance
        dy = best_entry.cy - entry.cy
        dx = abs(best_entry.cx - entry.cx)
        
        if 0 < dy < name_vertical_max and dx < name_horizontal_tolerance:
            if dy < min_dist:
                min_dist = dy
                lb_name = clean_text(entry.raw_text, is_number=False)
    
    return lb_name, lb_score

def process_static_score(img: np.ndarray) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
    """Process the static center score ROI"""
    config = STATIC_ROIS["CENTER_SCORE_ROI"]
    roi, coords = get_anchored_roi(img, config['rect'], config['anchor_x'], config['anchor_y'])
    raw_score = ocr_static_roi(roi, is_number=True)
    daily_high = clean_text(raw_score, is_number=True)
    return daily_high, coords

def process_power_zone(img: np.ndarray) -> Tuple[int, Tuple[int, int, int, int]]:
    """Process the power zone area (bottom-left of screenshot)"""
    h, w = img.shape[:2]
    
    zp_x = 0
    zp_y = int(h * POWER_ZONE_Y_RATIO)
    zp_w = int(w * POWER_ZONE_WIDTH_RATIO)
    zp_h = h - zp_y
    
    power_entries = scan_zone(img, (zp_x, zp_y, zp_w, zp_h))
    valid_power = [e for e in power_entries if e.number > MIN_POWER_THRESHOLD]
    
    found_power = 0
    if valid_power:
        # Take leftmost valid power value
        valid_power.sort(key=lambda k: k.cx)
        found_power = valid_power[0].number
    
    return found_power, (zp_x, zp_y, zp_w, zp_h)

def process_leaderboard_zone(
    img: np.ndarray,
    scale_x: float,
    scale_y: float
) -> Tuple[str, int, Tuple[int, int, int, int]]:
    """Process the leaderboard zone (bottom-right of screenshot)"""
    h, w = img.shape[:2]
    
    zl_x = int(w * LEADERBOARD_ZONE_X_RATIO)
    zl_y = int(h * LEADERBOARD_ZONE_Y_RATIO)
    zl_w = w - zl_x
    zl_h = h - zl_y
    
    lb_entries = scan_zone(img, (zl_x, zl_y, zl_w, zl_h))
    lb_name, lb_score = find_leaderboard_name_and_score(lb_entries, scale_x, scale_y)
    
    return lb_name, lb_score, (zl_x, zl_y, zl_w, zl_h)

# ============================================================================
# IMAGE PROCESSING FUNCTIONS - Main Analysis
# ============================================================================
async def analyze_screenshot_async(
    image_bytes: bytes,
    debug: bool = False
) -> AnalysisResult:
    """
    Analyze game screenshot and extract all relevant data
    Processes three areas in parallel:
    1. Center score (daily high)
    2. Power zone (total power)
    3. Leaderboard zone (rank name and score)
    """
    if not image_bytes:
        raise ValueError("No image data provided")
    
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    
    # Downscale if needed
    img = downscale_if_needed(img)
    
    # Calculate scale factors
    scale_x, scale_y = calculate_scale_from_image(img)
    
    # Process all zones in parallel
    loop = asyncio.get_event_loop()
    
    score_task = loop.run_in_executor(executor, process_static_score, img)
    power_task = loop.run_in_executor(executor, process_power_zone, img)
    leaderboard_task = loop.run_in_executor(
        executor,
        process_leaderboard_zone,
        img,
        scale_x,
        scale_y
    )
    
    # Await all results
    daily_high, score_coords = await score_task
    found_power, power_zone = await power_task
    lb_name, lb_score, lb_zone = await leaderboard_task
    
    # Debug visualization if requested
    if debug:
        debug_img = img.copy()
        
        # Draw score ROI
        if score_coords:
            rx, ry, rw, rh = score_coords
            cv2.rectangle(debug_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
        
        # Draw power zone
        zp_x, zp_y, zp_w, zp_h = power_zone
        cv2.rectangle(debug_img, (zp_x, zp_y), (zp_x+zp_w, zp_y+zp_h), (0, 0, 255), 2)
        
        # Draw leaderboard zone
        zl_x, zl_y, zl_w, zl_h = lb_zone
        cv2.rectangle(debug_img, (zl_x, zl_y), (zl_x+zl_w, zl_y+zl_h), (255, 0, 0), 2)
        
        # Save debug image
        try:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as tmp:
                cv2.imwrite(tmp.name, debug_img)
                logger.info(f"Debug image saved: {tmp.name}")
        except Exception as e:
            logger.error(f"Failed to save debug image: {str(e)}")
    
    return AnalysisResult(
        daily_high=daily_high,
        total_power=found_power,
        rank_name=lb_name,
        rank_score=lb_score
    )

# ============================================================================
# CLEANUP FUNCTIONS
# ============================================================================
def cleanup_executor():
    """Cleanup thread pool executor on shutdown"""
    if executor and not executor._shutdown:
        executor.shutdown(wait=True)
        logger.info("ThreadPoolExecutor shut down")

atexit.register(cleanup_executor)

# ============================================================================
# BOT EVENTS
# ============================================================================
@bot.event
async def on_ready():
    """Initialize bot when ready"""
    logger.info(f'Logged in as {bot.user}')
    
    try:
        # Initialize OCR
        await initialize_ocr_reader()
        
        # Sync commands
        logger.info('Syncing slash commands...')
        await bot.tree.sync()
        
        # Start reminder task
        if not check_reminder_schedule.is_running():
            check_reminder_schedule.start()
            logger.info('Reminder task started')
        
        logger.info('Bot is ready!')
        
    except Exception as e:
        logger.error(f'Bot initialization failed: {str(e)}')
        await bot.close()

@bot.event
async def on_error(event, *args, **kwargs):
    """Handle general bot errors"""
    import traceback
    logger.error(f'Error in {event}:')
    logger.error(traceback.format_exc())

@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError
):
    """Global error handler for slash commands"""
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(
            f"⏳ Command on cooldown. Try again in {error.retry_after:.1f}s",
            ephemeral=True
        )
    elif isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message(
            "❌ You don't have permission to use this command.",
            ephemeral=True
        )
    else:
        logger.error(f"Command error: {str(error)}")
        import traceback
        logger.error(traceback.format_exc())
        
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    f"❌ An error occurred: {str(error)}",
                    ephemeral=True
                )
            else:
                await interaction.followup.send(
                    f"❌ An error occurred: {str(error)}",
                    ephemeral=True
                )
        except Exception as e:
            logger.error(f"Failed to send error message: {str(e)}")

# ============================================================================
# SCHEDULED TASKS
# ============================================================================
@tasks.loop(hours=1)
async def check_reminder_schedule():
    """
    Check if it's time to send reminders
    Schedule: Mon, Wed, Fri, Sat, Sun at 8 PM GMT+2
    """
    global last_reminder_date
    
    try:
        gmt_plus_2 = timezone(timedelta(hours=2))
        now = datetime.now(gmt_plus_2)
        
        # Only run between 8:00 PM and 9:00 PM
        if now.hour != REMINDER_HOUR:
            return
        
        # Only run on specified days
        if now.weekday() not in REMINDER_DAYS:
            return
        
        # Prevent duplicate runs on same day
        today_date = now.date().isoformat()
        if last_reminder_date == today_date:
            return
        
        last_reminder_date = today_date
        game_date = get_game_date(now)
        
        logger.info(f"Reminder time! Checking {len(reminder_roles)} channels for {game_date}")
        
        # Process each configured channel
        for (guild_id, channel_id), config in reminder_roles.items():
            try:
                guild = bot.get_guild(guild_id)
                if not guild:
                    logger.warning(f"Guild {guild_id} not found")
                    continue
                
                channel = guild.get_channel(channel_id)
                if not channel:
                    logger.warning(f"Channel {channel_id} not found in guild {guild_id}")
                    continue
                
                role = guild.get_role(config['role_id'])
                if not role:
                    logger.warning(f"Role {config['role_id']} not found in guild {guild_id}")
                    continue
                
                club_name = config['club_name']
                
                # Get members who haven't submitted
                members_to_ping = await get_members_who_havent_submitted_by_role(
                    guild, role, club_name, game_date
                )
                
                if not members_to_ping:
                    logger.info(
                        f"All members with role {role.name} in "
                        f"{guild.name}#{channel.name} have submitted!"
                    )
                    continue
                
                # Send reminder
                mentions = " ".join([member.mention for member in members_to_ping])
                message = (
                    f"**Attention:** {mentions}\n\n"
                    f"🔔 This is a friendly reminder to submit a screenshot "
                    f"of your contest scores for the day!"
                )
                
                await channel.send(message)
                logger.info(
                    f"Sent reminder to {len(members_to_ping)} members in "
                    f"{guild.name}#{channel.name}"
                )
                
            except Exception as e:
                logger.error(
                    f"Error processing reminder for guild {guild_id}, "
                    f"channel {channel_id}: {str(e)}"
                )
                
    except Exception as e:
        logger.error(f"Error in reminder task: {str(e)}")

# ============================================================================
# SLASH COMMANDS - Channel Management
# ============================================================================
@bot.tree.command(
    name="activatecontestsubmissions",
    description="[MOD] Activate contest submissions in this channel"
)
@app_commands.describe(club_name="The club name for this channel")
@app_commands.default_permissions(moderate_members=True)
async def activate_channel(interaction: discord.Interaction, club_name: str):
    """Activate contest submissions for this channel"""
    try:
        if interaction.guild is None:
            await interaction.response.send_message(
                "❌ This command can only be used in a server.",
                ephemeral=True
            )
            return
        
        if not interaction.user.guild_permissions.moderate_members:
            await interaction.response.send_message(
                "❌ You need moderator permissions to manage contest submissions.",
                ephemeral=True
            )
            return
        
        channel_key = (interaction.guild.id, interaction.channel.id)
        
        # Check if already activated
        if channel_key in activated_channels:
            old_club = activated_channels[channel_key]
            activated_channels[channel_key] = club_name
            save_channels(activated_channels, CHANNELS_STORAGE_FILE)
            
            await interaction.response.send_message(
                f"✅ **Contest submissions updated!**\n\n"
                f"**Previous Club:** {old_club}\n"
                f"**New Club:** {club_name}\n\n"
                f"Users can now use `/submit` commands here.",
                ephemeral=False
            )
            logger.info(
                f"Updated channel {interaction.guild.name}#{interaction.channel.name} "
                f"from {old_club} to {club_name}"
            )
        else:
            activated_channels[channel_key] = club_name
            save_channels(activated_channels, CHANNELS_STORAGE_FILE)
            
            await interaction.response.send_message(
                f"✅ **Contest submissions activated!**\n\n"
                f"**Club Name:** {club_name}\n\n"
                f"Users can now use `/submit` commands here.",
                ephemeral=False
            )
            logger.info(
                f"Activated channel {interaction.guild.name}#{interaction.channel.name} "
                f"for club {club_name}"
            )
    
    except Exception as e:
        logger.error(f"Activate channel error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        await interaction.response.send_message(
            f"❌ Error: {str(e)}",
            ephemeral=True
        )

@bot.tree.command(
    name="deactivatecontestsubmissions",
    description="[MOD] Deactivate contest submissions in this channel"
)
@app_commands.default_permissions(moderate_members=True)
async def deactivate_channel(interaction: discord.Interaction):
    """Deactivate contest submissions for this channel"""
    try:
        if interaction.guild is None:
            await interaction.response.send_message(
                "❌ This command can only be used in a server.",
                ephemeral=True
            )
            return
        
        if not interaction.user.guild_permissions.moderate_members:
            await interaction.response.send_message(
                "❌ You need moderator permissions to manage contest submissions.",
                ephemeral=True
            )
            return
        
        channel_key = (interaction.guild.id, interaction.channel.id)
        
        # Check if activated
        if channel_key not in activated_channels:
            await interaction.response.send_message(
                "❌ Contest submissions are not currently activated in this channel.",
                ephemeral=True
            )
            return
        
        club_name = activated_channels[channel_key]
        del activated_channels[channel_key]
        save_channels(activated_channels, CHANNELS_STORAGE_FILE)
        
        # Also remove reminder configuration if exists
        if channel_key in reminder_roles:
            del reminder_roles[channel_key]
            save_reminder_roles(reminder_roles, REMINDER_ROLES_FILE)
            logger.info(
                f"Also removed reminder configuration for "
                f"{interaction.guild.name}#{interaction.channel.name}"
            )
        
        await interaction.response.send_message(
            f"✅ **Contest submissions deactivated!**\n\n"
            f"**Previous Club:** {club_name}\n\n"
            f"Users can no longer use `/submit` commands here.",
            ephemeral=False
        )
        logger.info(
            f"Deactivated channel {interaction.guild.name}#{interaction.channel.name}"
        )
    
    except Exception as e:
        logger.error(f"Deactivate channel error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        await interaction.response.send_message(
            f"❌ Error: {str(e)}",
            ephemeral=True
        )

# ============================================================================
# SLASH COMMANDS - Submission
# ============================================================================
@bot.tree.command(name="submit", description="Submit a screenshot of your contest scores")
@app_commands.describe(
    screenshot="Screenshot of your game results",
    club_name="Override club name (optional)"
)
async def submit_screenshot(
    interaction: discord.Interaction,
    screenshot: discord.Attachment,
    club_name: Optional[str] = None
):
    """Submit a screenshot for processing"""
    # Don't defer yet - we'll handle it based on what happens
    
    try:
        # Rate limiting
        is_allowed, seconds = check_rate_limit(interaction.user.id)
        if not is_allowed:
            await interaction.response.send_message(
                f"⏳ **Rate limit exceeded**\n\n"
                f"You can submit again in {seconds // 60} minutes.",
                ephemeral=True
            )
            return
        
        # Validate image
        if not screenshot.content_type or not screenshot.content_type.startswith('image/'):
            await interaction.response.send_message(
                "❌ Invalid file type. Please upload an image.",
                ephemeral=True
            )
            return
        
        if screenshot.size > MAX_IMAGE_SIZE:
            await interaction.response.send_message(
                f"❌ Image too large. Maximum size is "
                f"{MAX_IMAGE_SIZE // (1024*1024)}MB.",
                ephemeral=True
            )
            return
        
        # Now defer as ephemeral for status updates
        await interaction.response.defer(ephemeral=True)
        
        # Status updates (ephemeral)
        status_msg = await interaction.followup.send(
            "⏳ **Processing screenshot...**\n📥 Downloading image...",
            wait=True
        )
        
        # Download image
        try:
            image_bytes = await asyncio.wait_for(
                screenshot.read(),
                timeout=IMAGE_DOWNLOAD_TIMEOUT
            )
        except asyncio.TimeoutError:
            await status_msg.edit(
                content="❌ Download timeout. Please try again with a smaller image."
            )
            return
        
        # Analyze image
        await status_msg.edit(
            content="⏳ **Processing screenshot...**\n🔍 Analyzing with OCR..."
        )
        
        result = await analyze_screenshot_async(image_bytes, debug=False)
        
        # Save to database
        await status_msg.edit(
            content="⏳ **Processing screenshot...**\n💾 Saving to database..."
        )
        
        # Get username
        username = await get_guild_nick(interaction) or interaction.user.display_name
        
        # Get club name
        if not club_name:
            channel_key = (interaction.guild_id, interaction.channel_id)
            club_name = activated_channels.get(channel_key, "Unknown")
        
        # Create submission
        submission = create_submission(
            datetime_obj=datetime.now(),
            club_name=club_name,
            highest_today=result.daily_high,
            weekly_score=result.rank_score,
            total_power=result.total_power,
            username=username
        )
        
        # Send result
        if submission['success']:
            # Delete ephemeral status message
            await status_msg.delete()
            # Send public success message
            await interaction.followup.send(
                content=(
                    f"✅ **Submission Recorded!**\n"
                    f"🏆 **Club:** {club_name}\n"
                    f"👤 **Player:** {username}\n"
                    f"📈 **Daily Submission:** {result.daily_high:,}\n"
                    f"🎯 **Weekly Submissions:** {result.rank_score:,}"
                ),
                ephemeral=False
            )
        else:
            error_msg = submission.get('error', 'Unknown error')
            if submission.get('partial'):
                errors = '\n'.join(f"• {e}" for e in submission.get('errors', []))
                # Delete ephemeral status message
                await status_msg.delete()
                # Send public partial success message
                await interaction.followup.send(
                    content=(
                        f"⚠️ **Partial Success**\n\n"
                        f"Some data was saved, but errors occurred:\n{errors}"
                    ),
                    ephemeral=False
                )
            else:
                # Keep as ephemeral for full failure
                await status_msg.edit(
                    content=f"❌ **Submission Failed**\n\n{error_msg}"
                )
            
    except Exception as e:
        logger.error(f"Submit error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        try:
            # Try to send ephemeral error
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    f"❌ **Error:** {str(e)}",
                    ephemeral=True
                )
            else:
                await interaction.followup.send(
                    f"❌ **Error:** {str(e)}",
                    ephemeral=True
                )
        except Exception:
            pass
# ============================================================================
# SLASH COMMANDS - Reminder Management
# ============================================================================
@bot.tree.command(
    name="setsubmissionreminder",
    description="[MOD] Enable automated daily submission reminders"
)
@app_commands.describe(role="The role whose members should be reminded")
@app_commands.default_permissions(administrator=True)
async def set_submission_reminder(
    interaction: discord.Interaction,
    role: discord.Role
):
    """Set up automated submission reminders for this channel"""
    try:
        if not interaction.user.guild_permissions.manage_guild:
            await interaction.response.send_message(
                "❌ You need 'Manage Server' permission to use this command.",
                ephemeral=True
            )
            return
        
        channel_key = (interaction.guild_id, interaction.channel_id)
        
        # Check if channel is activated
        if channel_key not in activated_channels:
            await interaction.response.send_message(
                "⚠️ **Channel not activated**\n\n"
                "This channel is not activated for contest submissions yet.\n"
                "Please use `/activatecontestsubmissions` first to set the club name.",
                ephemeral=True
            )
            return
        
        club_name = activated_channels[channel_key]
        
        # Check if already configured
        if channel_key in reminder_roles:
            existing_config = reminder_roles[channel_key]
            existing_role = interaction.guild.get_role(existing_config['role_id'])
            role_name = existing_role.name if existing_role else "Unknown"
            
            await interaction.response.send_message(
                f"⚠️ **Reminders already active**\n\n"
                f"🏆 **Club:** {existing_config['club_name']}\n"
                f"👥 **Role:** {role_name}\n\n"
                f"Use `/deactivatesubmissionreminder` to disable first.",
                ephemeral=True
            )
            return
        
        # Activate reminders
        reminder_roles[channel_key] = {
            'club_name': club_name,
            'role_id': role.id
        }
        save_reminder_roles(reminder_roles, REMINDER_ROLES_FILE)
        
        await interaction.response.send_message(
            f"✅ **Submission Reminders Activated!**\n\n"
            f"🏆 **Club:** {club_name}\n"
            f"👥 **Role:** {role.mention}\n"
            f"📅 **Schedule:** Monday, Wednesday, Friday, Saturday, Sunday\n"
            f"⏰ **Time:** 8:00-9:00 PM GMT+2\n\n"
            f"Members with the {role.mention} role who haven't submitted "
            f"will be pinged automatically.\n\n"
            f"Use `/deactivatesubmissionreminder` to stop reminders.",
            ephemeral=False
        )
        logger.info(
            f"Reminders activated for {interaction.guild.name}#{interaction.channel.name} "
            f"(Club: {club_name}, Role: {role.name})"
        )
        
    except Exception as e:
        logger.error(f"Set reminder error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        await interaction.response.send_message(
            f"❌ Error: {str(e)}",
            ephemeral=True
        )

@bot.tree.command(
    name="deactivatesubmissionreminder",
    description="[MOD] Disable automated daily submission reminders"
)
@app_commands.default_permissions(administrator=True)
async def deactivate_submission_reminder(interaction: discord.Interaction):
    """Disable automated submission reminders for this channel"""
    try:
        if not interaction.user.guild_permissions.manage_guild:
            await interaction.response.send_message(
                "❌ You need 'Manage Server' permission to use this command.",
                ephemeral=True
            )
            return
        
        channel_key = (interaction.guild_id, interaction.channel_id)
        
        # Check if reminders are active
        if channel_key not in reminder_roles:
            await interaction.response.send_message(
                "⚠️ No active reminders found in this channel.",
                ephemeral=True
            )
            return
        
        # Get config before removing
        config = reminder_roles[channel_key]
        club_name = config['club_name']
        role = interaction.guild.get_role(config['role_id'])
        role_name = role.name if role else "Unknown"
        
        # Deactivate reminders
        del reminder_roles[channel_key]
        save_reminder_roles(reminder_roles, REMINDER_ROLES_FILE)
        
        await interaction.response.send_message(
            f"✅ **Submission Reminders Deactivated**\n\n"
            f"🏆 **Club:** {club_name}\n"
            f"👥 **Role:** {role_name}\n\n"
            f"No more automated reminders will be sent in this channel.",
            ephemeral=False
        )
        logger.info(
            f"Reminders deactivated for "
            f"{interaction.guild.name}#{interaction.channel.name}"
        )
        
    except Exception as e:
        logger.error(f"Deactivate reminder error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        await interaction.response.send_message(
            f"❌ Error: {str(e)}",
            ephemeral=True
        )

@bot.tree.command(
    name="submissionreminder",
    description="[MOD] Manually send submission reminder now"
)
@app_commands.default_permissions(administrator=True)
async def submission_reminder(interaction: discord.Interaction):
    """Manually trigger a submission reminder check and message"""
    try:
        if not interaction.user.guild_permissions.manage_guild:
            await interaction.response.send_message(
                "❌ You need 'Manage Server' permission to use this command.",
                ephemeral=True
            )
            return
        
        channel_key = (interaction.guild_id, interaction.channel_id)
        
        # Check if reminders are configured
        if channel_key not in reminder_roles:
            await interaction.response.send_message(
                "⚠️ **No reminder configuration found**\n\n"
                "Use `/setsubmissionreminder` first to configure automated reminders.",
                ephemeral=True
            )
            return
        
        await interaction.response.defer(ephemeral=True)
        
        config = reminder_roles[channel_key]
        club_name = config['club_name']
        role = interaction.guild.get_role(config['role_id'])
        
        if not role:
            await interaction.followup.send(
                f"❌ Configured role (ID: {config['role_id']}) not found in this server.",
                ephemeral=True
            )
            return
        
        # Get current game date
        gmt_plus_2 = timezone(timedelta(hours=2))
        now = datetime.now(gmt_plus_2)
        game_date = get_game_date(now)
        
        # Check who hasn't submitted
        members_to_ping = await get_members_who_havent_submitted_by_role(
            interaction.guild, role, club_name, game_date
        )
        
        if not members_to_ping:
            await interaction.followup.send(
                f"✅ **All members have submitted!**\n\n"
                f"🏆 **Club:** {club_name}\n"
                f"👥 **Role:** {role.name}\n"
                f"📅 **Date:** {game_date}\n\n"
                f"Everyone with the {role.mention} role has submitted their scores for today.",
                ephemeral=False
            )
            logger.info(
                f"Manual reminder check: all members submitted in "
                f"{interaction.guild.name}#{interaction.channel.name}"
            )
            return
        
        # Send reminder message
        mentions = " ".join([member.mention for member in members_to_ping])
        message = (
            f"**Attention:** {mentions}\n\n"
            f"🔔 This is a friendly reminder to submit a screenshot "
            f"of your contest scores for the day!"
        )
        
        await interaction.channel.send(message)
        
        # Send confirmation to admin
        await interaction.followup.send(
            f"✅ **Reminder sent!**\n\n"
            f"🏆 **Club:** {club_name}\n"
            f"👥 **Role:** {role.name}\n"
            f"📅 **Date:** {game_date}\n"
            f"👤 **Members pinged:** {len(members_to_ping)}",
            ephemeral=False
        )
        logger.info(
            f"Manual reminder sent to {len(members_to_ping)} members in "
            f"{interaction.guild.name}#{interaction.channel.name}"
        )
        
    except Exception as e:
        logger.error(f"Manual reminder error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        if not interaction.response.is_done():
            await interaction.response.send_message(
                f"❌ Error: {str(e)}",
                ephemeral=True
            )
        else:
            await interaction.followup.send(
                f"❌ Error: {str(e)}",
                ephemeral=True
            )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    try:
        logger.info("Starting bot...")
        bot.run(DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        cleanup_executor()