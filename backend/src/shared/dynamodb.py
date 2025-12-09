"""
Shared DynamoDB client for use across Discord bot and API
"""
import os
import boto3
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

# DynamoDB configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
DYNAMODB_TABLE_NAME = os.getenv('DYNAMODB_TABLE_NAME')

# DynamoDB key prefixes (must match discord_bot.py)
DDB_USER_PREFIX = 'USER#'
DDB_CLUB_PREFIX = 'CLUB#'
DDB_SUBMISSION_PREFIX = 'SUBMISSION#'
DDB_DAILY_PREFIX = 'DAILY#'
DDB_ACTIVITY_PREFIX = 'ACTIVITY#'


def get_dynamodb_client():
    """Get boto3 DynamoDB client"""
    return boto3.client(
        'dynamodb',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )


def get_dynamodb_resource():
    """Get boto3 DynamoDB resource"""
    return boto3.resource(
        'dynamodb',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )


def get_table():
    """Get DynamoDB table resource"""
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(DYNAMODB_TABLE_NAME)
    return table


# Initialize and verify connection on module import
try:
    table = get_table()
    table.load()
    logger.info(f"Shared DynamoDB connection initialized: {DYNAMODB_TABLE_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize DynamoDB connection: {str(e)}")
    raise
