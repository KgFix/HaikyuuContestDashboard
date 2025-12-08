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
from datetime import datetime, timedelta, timezone, time as dt_time
from decimal import Decimal
import uuid
import atexit
import sys
import logging
from typing import Optional, Union, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time
import json
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
DYNAMODB_TABLE_NAME = os.getenv('DYNAMODB_TABLE_NAME')

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

# Image processing constants
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

# Image zone constants
POWER_ZONE_Y_RATIO = 0.85
POWER_ZONE_WIDTH_RATIO = 0.40
LEADERBOARD_ZONE_X_RATIO = 0.65
LEADERBOARD_ZONE_Y_RATIO = 0.60

# DynamoDB key prefixes
DDB_USER_PREFIX = 'USER#'
DDB_CLUB_PREFIX = 'CLUB#'
DDB_SUBMISSION_PREFIX = 'SUBMISSION#'
DDB_DAILY_PREFIX = 'DAILY#'
DDB_ACTIVITY_PREFIX = 'ACTIVITY#'

NUMBER_PATTERN = re.compile(r'\D')
TEXT_PATTERN = re.compile(r'[^\w\s]')

rate_limit_tracker: Dict[int, List[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 3600
RATE_LIMIT_MAX_REQUESTS = 10

STATIC_ROIS = {
    "CENTER_SCORE_ROI": {
        "rect": (980, 875, 300, 60),
        "anchor_x": "center",
        "anchor_y": "top",
        "type": "number"
    }
}

@dataclass
class OCREntry:
    raw_text: str
    number: int
    cx: float
    cy: float

@dataclass
class AnalysisResult:
    daily_high: int
    total_power: int
    rank_name: str
    rank_score: int

reader = None

def get_ocr_reader():
    global reader
    if reader is None:
        raise RuntimeError("OCR reader not initialized.")
    return reader

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='/', intents=intents)

executor = ThreadPoolExecutor(max_workers=4)

def cleanup_executor():
    if executor and not executor._shutdown:
        executor.shutdown(wait=True)
        logger.info("ThreadPoolExecutor shut down")

atexit.register(cleanup_executor)

activated_channels = {}
CHANNELS_STORAGE_FILE = os.getenv('CHANNELS_STORAGE_FILE', 'activated_channels.json')

reminder_channels = {}
REMINDERS_STORAGE_FILE = os.getenv('REMINDERS_STORAGE_FILE', 'reminder_channels.json')
last_reminder_date = None

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
        logger.warning(f"Could not save channels to {filename}: {str(e)}")

activated_channels = load_channels(CHANNELS_STORAGE_FILE)
reminder_channels = load_channels(REMINDERS_STORAGE_FILE)

def check_rate_limit(user_id: int) -> Tuple[bool, int]:
    now = time.time()
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
    gmt_plus_2 = timezone(timedelta(hours=2))
    dt_gmt2 = dt.astimezone(gmt_plus_2)
    
    if dt_gmt2.hour < 8:
        game_date = (dt_gmt2.date() - timedelta(days=1)).isoformat()
    else:
        game_date = dt_gmt2.date().isoformat()
    
    return game_date

def get_game_week(dt: datetime) -> str:
    gmt_plus_2 = timezone(timedelta(hours=2))
    dt_gmt2 = dt.astimezone(gmt_plus_2)
    
    if dt_gmt2.hour < 8:
        adjusted_date = dt_gmt2.date() - timedelta(days=1)
    else:
        adjusted_date = dt_gmt2.date()
    
    weekday = adjusted_date.isoweekday()
    if weekday == 1 and dt_gmt2.hour < 8:
        adjusted_date = adjusted_date - timedelta(days=1)
    
    year, week, _ = adjusted_date.isocalendar()
    return f"{year}-W{week:02d}"

def create_submission(datetime_obj: datetime, club_name: str, highest_today: int,
                      weekly_score: int, total_power: int, username: str) -> dict:
    """Create submission entries in DynamoDB for user and club tracking"""
    try:
        iso_datetime = datetime_obj.isoformat()
        game_date = get_game_date(datetime_obj)
        game_week = get_game_week(datetime_obj)
        
        errors = []
        
        # 1. User Submission
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
            logger.info(f"Stored user submission: {username}")
        except Exception as e:
            logger.error(f"User submission failed: {str(e)}")
            errors.append(f"User submission failed: {str(e)}")
        
        # 2. User Daily Summary
        try:
            table.update_item(
                Key={
                    'PK': f'{DDB_USER_PREFIX}{username}',
                    'SK': f'{DDB_DAILY_PREFIX}{game_date}'
                },
                UpdateExpression='SET BestHighestToday = if_not_exists(BestHighestToday, :score), ClubName = :club, GameDate = :date, GameWeek = :week, LastUpdated = :updated, EntryType = :type',
                ConditionExpression='attribute_not_exists(BestHighestToday) OR BestHighestToday < :score',
                ExpressionAttributeValues={
                    ':score': highest_today,
                    ':club': club_name,
                    ':date': game_date,
                    ':week': game_week,
                    ':updated': iso_datetime,
                    ':type': 'UserDailySummary'
                }
            )
            logger.info(f"Updated user daily: {username}")
        except table.meta.client.exceptions.ConditionalCheckFailedException:
            logger.debug(f"User daily not updated (no improvement): {username}")
        except Exception as e:
            logger.error(f"User daily failed: {str(e)}")
            errors.append(f"User daily failed: {str(e)}")
        
        # 3. Club Daily Summary
        try:
            table.update_item(
                Key={
                    'PK': f'{DDB_CLUB_PREFIX}{club_name}',
                    'SK': f'{DDB_DAILY_PREFIX}{game_date}'
                },
                UpdateExpression='SET MaxTotalPower = if_not_exists(MaxTotalPower, :power), GameDate = :date, GameWeek = :week, LastUpdated = :updated, EntryType = :type',
                ConditionExpression='attribute_not_exists(MaxTotalPower) OR MaxTotalPower < :power',
                ExpressionAttributeValues={
                    ':power': total_power,
                    ':date': game_date,
                    ':week': game_week,
                    ':updated': iso_datetime,
                    ':type': 'ClubDailySummary'
                }
            )
            logger.info(f"Updated club daily: {club_name}")
        except table.meta.client.exceptions.ConditionalCheckFailedException:
            logger.debug(f"Club daily not updated (no improvement): {club_name}")
        except Exception as e:
            logger.error(f"Club daily failed: {str(e)}")
            errors.append(f"Club daily failed: {str(e)}")
        
        # 4. Club Daily User Activity Tracker
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
            logger.error(f"Club activity failed: {str(e)}")
            errors.append(f"Club activity failed: {str(e)}")
        
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
        return {'success': False, 'error': str(e)}

def clean_text(text: str, is_number: bool = False) -> Union[int, str]:
    if not text:
        return 0 if is_number else ""
    
    if is_number:
        text = text.replace('S', '5').replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
        clean = NUMBER_PATTERN.sub('', text)
        return int(clean) if clean else 0
    
    clean = TEXT_PATTERN.sub('', text).strip()
    return clean

async def get_members_who_havent_submitted(guild: discord.Guild, channel: discord.TextChannel, club_name: str, game_date: str) -> List[discord.Member]:
    """Get list of members in channel who haven't submitted for the day"""
    try:
        response = table.get_item(
            Key={
                'PK': f'{DDB_CLUB_PREFIX}{club_name}',
                'SK': f'{DDB_ACTIVITY_PREFIX}{game_date}'
            }
        )
        
        submitted_users = set()
        if 'Item' in response and 'Users' in response['Item']:
            submitted_users = set(response['Item']['Users'].keys())
        
        members_to_ping = []
        async for member in guild.fetch_members(limit=None):
            if member.bot:
                continue
            
            permissions = channel.permissions_for(member)
            if not permissions.view_channel:
                continue
            
            member_name = member.nick if member.nick else member.display_name
            
            if member_name not in submitted_users:
                members_to_ping.append(member)
        
        return members_to_ping
        
    except Exception as e:
        logger.error(f"Error getting non-submitters: {str(e)}")
        return []

@tasks.loop(minutes=30)
async def check_reminder_schedule():
    """Check if it's time to send reminders (Mon, Wed, Fri, Sat, Sun at 8 PM GMT+2)"""
    global last_reminder_date
    
    try:
        gmt_plus_2 = timezone(timedelta(hours=2))
        now = datetime.now(gmt_plus_2)
        
        # Only run between 8:00 PM and 8:30 PM
        if now.hour != 20 or now.minute >= 30:
            return
        
        weekday = now.weekday()
        if weekday not in [0, 2, 4, 5, 6]:  # Mon, Wed, Fri, Sat, Sun
            return
        
        today_date = now.date().isoformat()
        if last_reminder_date == today_date:
            return
        
        last_reminder_date = today_date
        game_date = get_game_date(now)
        
        logger.info(f"Reminder time! Checking {len(reminder_channels)} channels for {game_date}")
        
        for (guild_id, channel_id), club_name in reminder_channels.items():
            try:
                guild = bot.get_guild(guild_id)
                if not guild:
                    logger.warning(f"Guild {guild_id} not found")
                    continue
                
                channel = guild.get_channel(channel_id)
                if not channel:
                    logger.warning(f"Channel {channel_id} not found")
                    continue
                
                members_to_ping = await get_members_who_havent_submitted(
                    guild, channel, club_name, game_date
                )
                
                if not members_to_ping:
                    logger.info(f"All members in {guild.name}#{channel.name} have submitted!")
                    continue
                
                mentions = " ".join([member.mention for member in members_to_ping])
                message = (
                    f"**Attention:** {mentions}\n\n"
                    f"🔔 This is a friendly reminder to submit a screenshot of your contest scores for the day!"
                )
                
                await channel.send(message)
                logger.info(f"Sent reminder to {len(members_to_ping)} members in {guild.name}#{channel.name}")
                
            except Exception as e:
                logger.error(f"Error processing reminder for guild {guild_id}, channel {channel_id}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error in reminder task: {str(e)}")

async def get_guild_nick(interaction: discord.Interaction) -> Optional[str]:
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
    h, w = image.shape[:2]
    scale_x = w / REF_WIDTH
    scale_y = h / REF_HEIGHT
    return scale_x, scale_y

def get_anchored_roi(image: np.ndarray, ref_rect: tuple, anchor_x: str, anchor_y: str) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    if image is None:
        return None, (0, 0, 0, 0)
    
    ref_x, ref_y, ref_w, ref_h = ref_rect
    h, w = image.shape[:2]
    
    scale_x, scale_y = calculate_scale_from_image(image)
    
    new_w = int(ref_w * scale_x)
    new_h = int(ref_h * scale_y)
    
    if anchor_x == "center":
        ref_center_x = REF_WIDTH / 2
        offset = ref_x - ref_center_x
        new_x = int((w / 2) + (offset * scale_x))
    else:
        new_x = int(ref_x * scale_x)

    new_y = int(ref_y * scale_y)

    new_x = max(0, min(new_x, w - new_w))
    new_y = max(0, min(new_y, h - new_h))
    
    return image[new_y:new_y+new_h, new_x:new_x+new_w], (new_x, new_y, new_w, new_h)

def preprocess_roi(roi: np.ndarray, scale: float = OCR_SCALE_FACTOR) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(upscaled)
    return enhanced

def ocr_static_roi(roi: np.ndarray, is_number: bool) -> str:
    if roi is None or roi.size == 0:
        return ""
    
    try:
        ocr_reader = get_ocr_reader()
        enhanced = preprocess_roi(roi)
        allowlist = '0123456789' if is_number else None
        results = ocr_reader.readtext(enhanced, detail=0, allowlist=allowlist, paragraph=True)
        return " ".join(results)
    except Exception as e:
        print(f"❌ OCR error: {str(e)}")
        return ""

def scan_zone(image: np.ndarray, zone_bbox: tuple) -> List[OCREntry]:
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
        print(f"❌ Zone scan error: {str(e)}")
        return []

def find_leaderboard_name_and_score(lb_entries: List[OCREntry], scale_x: float, scale_y: float) -> Tuple[str, int]:
    lb_numbers = [e for e in lb_entries if e.number > MIN_LEADERBOARD_SCORE_THRESHOLD]
    
    if not lb_numbers:
        return "Unknown", 0
    
    right_col_tolerance = int(LEADERBOARD_RIGHT_COLUMN_TOLERANCE_PX * scale_x)
    name_vertical_max = int(LEADERBOARD_NAME_VERTICAL_DISTANCE_MAX * scale_y)
    name_horizontal_tolerance = int(LEADERBOARD_NAME_HORIZONTAL_TOLERANCE * scale_x)
    
    max_x = max(e.cx for e in lb_numbers)
    right_column = [e for e in lb_numbers if e.cx >= (max_x - right_col_tolerance)]
    
    right_column.sort(key=lambda k: k.cy, reverse=True)
    best_entry = right_column[0]
    lb_score = best_entry.number
    
    min_dist = float('inf')
    lb_name = "Unknown"
    
    for entry in lb_entries:
        if entry == best_entry:
            continue
        
        txt = entry.raw_text.lower()
        if any(keyword in txt for keyword in ["start", "quick", "play", "join", "button"]):
            continue
        
        dy = best_entry.cy - entry.cy
        dx = abs(best_entry.cx - entry.cx)
        
        if 0 < dy < name_vertical_max and dx < name_horizontal_tolerance:
            if dy < min_dist:
                min_dist = dy
                lb_name = clean_text(entry.raw_text, is_number=False)
    
    return lb_name, lb_score

def downscale_if_needed(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    if max_dim > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image

def process_static_score(img: np.ndarray) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
    config = STATIC_ROIS["CENTER_SCORE_ROI"]
    roi, coords = get_anchored_roi(img, config['rect'], config['anchor_x'], config['anchor_y'])
    raw_score = ocr_static_roi(roi, is_number=True)
    daily_high = clean_text(raw_score, is_number=True)
    return daily_high, coords

def process_power_zone(img: np.ndarray) -> Tuple[int, Tuple[int, int, int, int]]:
    """Process the power zone area of the screenshot"""
    h, w = img.shape[:2]
    
    zp_x, zp_y = 0, int(h * POWER_ZONE_Y_RATIO)
    zp_w, zp_h = int(w * POWER_ZONE_WIDTH_RATIO), h - zp_y
    
    power_entries = scan_zone(img, (zp_x, zp_y, zp_w, zp_h))
    valid_power = [e for e in power_entries if e.number > MIN_POWER_THRESHOLD]
    
    found_power = 0
    if valid_power:
        valid_power.sort(key=lambda k: k.cx)
        found_power = valid_power[0].number
    
    return found_power, (zp_x, zp_y, zp_w, zp_h)

def process_leaderboard_zone(img: np.ndarray, scale_x: float, scale_y: float) -> Tuple[str, int, Tuple[int, int, int, int]]:
    """Process the leaderboard zone area of the screenshot"""
    h, w = img.shape[:2]
    zl_x, zl_y = int(w * LEADERBOARD_ZONE_X_RATIO), int(h * LEADERBOARD_ZONE_Y_RATIO)
    zl_w, zl_h = w - zl_x, h - zl_y
    
    lb_entries = scan_zone(img, (zl_x, zl_y, zl_w, zl_h))
    lb_name, lb_score = find_leaderboard_name_and_score(lb_entries, scale_x, scale_y)
    
    return lb_name, lb_score, (zl_x, zl_y, zl_w, zl_h)

async def analyze_screenshot_async(image_bytes: bytes, debug: bool = False) -> AnalysisResult:
    if not image_bytes:
        raise ValueError("No image data provided")
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    img = downscale_if_needed(img)
    
    h, w = img.shape[:2]
    scale_x, scale_y = calculate_scale_from_image(img)
    
    loop = asyncio.get_event_loop()
    
    score_task = loop.run_in_executor(executor, process_static_score, img)
    power_task = loop.run_in_executor(executor, process_power_zone, img)
    leaderboard_task = loop.run_in_executor(executor, process_leaderboard_zone, img, scale_x, scale_y)
    
    daily_high, score_coords = await score_task
    found_power, power_zone = await power_task
    lb_name, lb_score, lb_zone = await leaderboard_task

    if debug:
        debug_img = img.copy()
        if score_coords:
            rx, ry, rw, rh = score_coords
            cv2.rectangle(debug_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
        
        zp_x, zp_y, zp_w, zp_h = power_zone
        cv2.rectangle(debug_img, (zp_x, zp_y), (zp_x+zp_w, zp_y+zp_h), (0, 0, 255), 2)
        
        zl_x, zl_y, zl_w, zl_h = lb_zone
        cv2.rectangle(debug_img, (zl_x, zl_y), (zl_x+zl_w, zl_y+zl_h), (255, 0, 0), 2)
        
        try:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as tmp:
                cv2.imwrite(tmp.name, debug_img)
                print(f"📷 Debug image: {tmp.name}")
        except Exception as e:
            print(f"❌ Debug save failed: {str(e)}")

    return AnalysisResult(
        daily_high=daily_high,
        total_power=found_power,
        rank_name=lb_name,
        rank_score=lb_score
    )

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
            logger.error(f"EasyOCR init failed: {str(e)}")
            raise

@bot.event
async def on_error(event, *args, **kwargs):
    """Handle general bot errors"""
    import traceback
    logger.error(f'Error in {event}:')
    logger.error(traceback.format_exc())

@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
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

@bot.event
async def on_ready():
    """Initialize bot when ready"""
    logger.info(f'Logged in as {bot.user}')
    
    try:
        await initialize_ocr_reader()
        logger.info('Syncing commands...')
        await bot.tree.sync()
        
        if not check_reminder_schedule.is_running():
            check_reminder_schedule.start()
            logger.info('Reminder task started')
        
        logger.info('Bot ready!')
        
    except Exception as e:
        logger.error(f'Init failed: {str(e)}')
        await bot.close()

@bot.tree.command(name="submit", description="Submit a screenshot")
@app_commands.describe(
    screenshot="Screenshot of game results",
    club_name="Your club name"
)
async def submit_screenshot(
    interaction: discord.Interaction,
    screenshot: discord.Attachment,
    club_name: Optional[str] = None
):
    await interaction.response.defer(ephemeral=True)
    
    try:
        is_allowed, seconds = check_rate_limit(interaction.user.id)
        if not is_allowed:
            await interaction.followup.send(
                f"⏳ Rate limit exceeded. Try again in {seconds // 60} minutes.",
                ephemeral=True
            )
            return
        
        if not screenshot.content_type or not screenshot.content_type.startswith('image/'):
            await interaction.followup.send("❌ Invalid image file", ephemeral=True)
            return
        
        if screenshot.size > MAX_IMAGE_SIZE:
            await interaction.followup.send(
                f"❌ Image too large (max {MAX_IMAGE_SIZE // (1024*1024)}MB)",
                ephemeral=True
            )
            return
        
        status_msg = await interaction.followup.send(
            "⏳ **Processing screenshot...**\n📥 Downloading...",
            ephemeral=True,
            wait=True
        )
        
        try:
            image_bytes = await asyncio.wait_for(
                screenshot.read(),
                timeout=IMAGE_DOWNLOAD_TIMEOUT
            )
        except asyncio.TimeoutError:
            await status_msg.edit(content="❌ Download timeout")
            return
        
        await status_msg.edit(content="⏳ **Processing screenshot...**\n🔍 Analyzing with OCR...")
        
        result = await analyze_screenshot_async(image_bytes, False)
        
        await status_msg.edit(content="⏳ **Processing screenshot...**\n💾 Saving to database...")
        
        username = await get_guild_nick(interaction) or interaction.user.display_name
        
        if not club_name:
            channel_key = (interaction.guild_id, interaction.channel_id)
            club_name = activated_channels.get(channel_key, "Unknown")
        
        submission = create_submission(
            datetime_obj=datetime.now(),
            club_name=club_name,
            highest_today=result.daily_high,
            weekly_score=result.rank_score,
            total_power=result.total_power,
            username=username
        )
        
        if submission['success']:
            await status_msg.edit(
                content=f"✅ **Submission Recorded!**\n"
                f"👤 Player: **{username}**\n"
                f"🏆 Club: **{club_name}**\n"
                f"📈 Daily High: **{result.daily_high:,}**\n"
                f"⚡ Total Power: **{result.total_power:,}**\n"
                f"🎯 Rank: **{result.rank_name}** ({result.rank_score:,})"
            )
        else:
            error_msg = submission.get('error', 'Unknown error')
            await status_msg.edit(content=f"⚠️ **Failed**\n{error_msg}")
            
    except Exception as e:
        logger.error(f"Submit error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        await interaction.followup.send(f"❌ Error: {str(e)}", ephemeral=True)

@bot.tree.command(name="setsubmissionreminder", description="[MOD] Enable daily submission reminders")
@app_commands.describe(
    club_name="The club name to track submissions for"
)
@app_commands.default_permissions(administrator=True)
async def set_submission_reminder(
    interaction: discord.Interaction,
    club_name: str
):
    """Set up automated submission reminders for this channel"""
    try:
        # Check if user has manage_guild permission
        if not interaction.user.guild_permissions.manage_guild:
            await interaction.response.send_message(
                "❌ You need 'Manage Server' permission to use this command.",
                ephemeral=True
            )
            return
        
        channel_key = (interaction.guild_id, interaction.channel_id)
        
        # Check if already activated
        if channel_key in reminder_channels:
            await interaction.response.send_message(
                f"⚠️ Reminders are already active in this channel for club: **{reminder_channels[channel_key]}**\n"
                f"Use `/deactivatesubmissionreminder` to disable first.",
                ephemeral=True
            )
            return
        
        # Activate reminders
        reminder_channels[channel_key] = club_name
        save_channels(reminder_channels, REMINDERS_STORAGE_FILE)
        
        await interaction.response.send_message(
            f"✅ **Submission Reminders Activated!**\n\n"
            f"🏆 Club: **{club_name}**\n"
            f"📅 Schedule: Monday, Wednesday, Friday, Saturday, Sunday\n"
            f"⏰ Time: 8:00-8:30 PM GMT+2\n\n"
            f"Members who haven't submitted will be pinged automatically.\n"
            f"Use `/deactivatesubmissionreminder` to stop reminders.",
            ephemeral=True
        )
        logger.info(f"Reminders activated for {interaction.guild.name}#{interaction.channel.name} (Club: {club_name})")
        
    except Exception as e:
        logger.error(f"Set reminder error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        await interaction.response.send_message(f"❌ Error: {str(e)}", ephemeral=True)

@bot.tree.command(name="deactivatesubmissionreminder", description="[MOD] Disable daily submission reminders")
@app_commands.default_permissions(administrator=True)
async def deactivate_submission_reminder(interaction: discord.Interaction):
    """Disable automated submission reminders for this channel"""
    try:
        # Check if user has manage_guild permission
        if not interaction.user.guild_permissions.manage_guild:
            await interaction.response.send_message(
                "❌ You need 'Manage Server' permission to use this command.",
                ephemeral=True
            )
            return
        
        channel_key = (interaction.guild_id, interaction.channel_id)
        
        # Check if reminders are active
        if channel_key not in reminder_channels:
            await interaction.response.send_message(
                "⚠️ No active reminders found in this channel.",
                ephemeral=True
            )
            return
        
        # Get club name before removing
        club_name = reminder_channels[channel_key]
        
        # Deactivate reminders
        del reminder_channels[channel_key]
        save_channels(reminder_channels, REMINDERS_STORAGE_FILE)
        
        await interaction.response.send_message(
            f"✅ **Submission Reminders Deactivated**\n\n"
            f"🏆 Club: **{club_name}**\n"
            f"No more automated reminders will be sent in this channel.",
            ephemeral=True
        )
        logger.info(f"Reminders deactivated for {interaction.guild.name}#{interaction.channel.name}")
        
    except Exception as e:
        logger.error(f"Deactivate reminder error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        await interaction.response.send_message(f"❌ Error: {str(e)}", ephemeral=True)

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