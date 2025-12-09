"""
DynamoDB query functions for API endpoints
"""
import logging
from typing import List, Optional, Dict
from boto3.dynamodb.conditions import Key
from src.shared.dynamodb import (
    get_table,
    DDB_USER_PREFIX,
    DDB_CLUB_PREFIX,
    DDB_DAILY_PREFIX,
    DDB_ACTIVITY_PREFIX,
)

logger = logging.getLogger(__name__)


# ============================================================================
# User Queries
# ============================================================================

async def get_user_daily_history(username: str) -> List[Dict]:
    """
    Get user's daily performance history
    Returns list of UserDailySummary records
    """
    try:
        table = get_table()
        response = table.query(
            KeyConditionExpression=Key('PK').eq(f'{DDB_USER_PREFIX}{username}') &
                                   Key('SK').begins_with(DDB_DAILY_PREFIX),
            FilterExpression='EntryType = :type',
            ExpressionAttributeValues={
                ':type': 'UserDailySummary'
            }
        )
        
        items = response.get('Items', [])
        
        # Transform to match frontend expectations
        result = []
        for item in items:
            result.append({
                'date': item.get('GameDate', ''),
                'bestHighestToday': int(item.get('BestHighestToday', 0)),
                'clubName': item.get('ClubName', ''),
                'gameWeek': item.get('GameWeek', ''),
                'lastUpdated': item.get('LastUpdated', '')
            })
        
        # Sort by date descending
        result.sort(key=lambda x: x['date'], reverse=True)
        
        logger.info(f"Retrieved {len(result)} daily records for user {username}")
        return result
        
    except Exception as e:
        logger.error(f"Error getting user history for {username}: {str(e)}")
        raise


async def get_user_club(username: str) -> Optional[str]:
    """
    Get the club name for a specific user by checking their most recent submission
    """
    try:
        table = get_table()
        response = table.query(
            KeyConditionExpression=Key('PK').eq(f'{DDB_USER_PREFIX}{username}') &
                                   Key('SK').begins_with(DDB_DAILY_PREFIX),
            Limit=1,
            ScanIndexForward=False  # Get most recent
        )
        
        items = response.get('Items', [])
        if items:
            return items[0].get('ClubName')
        return None
        
    except Exception as e:
        logger.error(f"Error getting club for user {username}: {str(e)}")
        return None


async def get_all_users() -> List[str]:
    """
    Get list of all unique usernames from DynamoDB
    This scans for all USER# partition keys
    """
    try:
        table = get_table()
        
        # Scan with filter for USER# prefix
        response = table.scan(
            FilterExpression='begins_with(PK, :prefix) AND begins_with(SK, :daily)',
            ExpressionAttributeValues={
                ':prefix': DDB_USER_PREFIX,
                ':daily': DDB_DAILY_PREFIX
            },
            ProjectionExpression='PK'
        )
        
        # Extract unique usernames
        users = set()
        for item in response.get('Items', []):
            pk = item.get('PK', '')
            if pk.startswith(DDB_USER_PREFIX):
                username = pk[len(DDB_USER_PREFIX):]
                users.add(username)
        
        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='begins_with(PK, :prefix) AND begins_with(SK, :daily)',
                ExpressionAttributeValues={
                    ':prefix': DDB_USER_PREFIX,
                    ':daily': DDB_DAILY_PREFIX
                },
                ProjectionExpression='PK',
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            for item in response.get('Items', []):
                pk = item.get('PK', '')
                if pk.startswith(DDB_USER_PREFIX):
                    username = pk[len(DDB_USER_PREFIX):]
                    users.add(username)
        
        result = sorted(list(users))
        logger.info(f"Retrieved {len(result)} unique users")
        return result
        
    except Exception as e:
        logger.error(f"Error getting all users: {str(e)}")
        raise


# ============================================================================
# Club Queries
# ============================================================================

async def get_club_daily_history(club_name: str) -> List[Dict]:
    """
    Get club's daily performance history
    Returns list of ClubDailySummary records
    """
    try:
        table = get_table()
        response = table.query(
            KeyConditionExpression=Key('PK').eq(f'{DDB_CLUB_PREFIX}{club_name}') &
                                   Key('SK').begins_with(DDB_DAILY_PREFIX),
            FilterExpression='EntryType = :type',
            ExpressionAttributeValues={
                ':type': 'ClubDailySummary'
            }
        )
        
        items = response.get('Items', [])
        
        # Transform to match frontend expectations
        result = []
        for item in items:
            result.append({
                'date': item.get('GameDate', ''),
                'maxTotalPower': int(item.get('MaxTotalPower', 0)),
                'gameWeek': item.get('GameWeek', ''),
                'lastUpdated': item.get('LastUpdated', '')
            })
        
        # Sort by date descending
        result.sort(key=lambda x: x['date'], reverse=True)
        
        logger.info(f"Retrieved {len(result)} daily records for club {club_name}")
        return result
        
    except Exception as e:
        logger.error(f"Error getting club history for {club_name}: {str(e)}")
        raise


async def get_club_daily_activity(club_name: str) -> List[Dict]:
    """
    Get club's daily activity history
    Returns list of ClubDailyActivity records
    """
    try:
        table = get_table()
        response = table.query(
            KeyConditionExpression=Key('PK').eq(f'{DDB_CLUB_PREFIX}{club_name}') &
                                   Key('SK').begins_with(DDB_ACTIVITY_PREFIX),
            FilterExpression='EntryType = :type',
            ExpressionAttributeValues={
                ':type': 'ClubDailyActivity'
            }
        )
        
        items = response.get('Items', [])
        
        # Transform to match frontend expectations
        result = []
        for item in items:
            users_dict = item.get('Users', {})
            result.append({
                'date': item.get('GameDate', ''),
                'users': users_dict,
                'totalUsers': len(users_dict),
                'gameWeek': item.get('GameWeek', ''),
                'lastUpdated': item.get('LastUpdated', '')
            })
        
        # Sort by date descending
        result.sort(key=lambda x: x['date'], reverse=True)
        
        logger.info(f"Retrieved {len(result)} activity records for club {club_name}")
        return result
        
    except Exception as e:
        logger.error(f"Error getting club activity for {club_name}: {str(e)}")
        raise


async def get_all_clubs() -> List[str]:
    """
    Get list of all unique club names from DynamoDB
    This scans for all CLUB# partition keys
    """
    try:
        table = get_table()
        
        # Scan with filter for CLUB# prefix
        response = table.scan(
            FilterExpression='begins_with(PK, :prefix) AND begins_with(SK, :daily)',
            ExpressionAttributeValues={
                ':prefix': DDB_CLUB_PREFIX,
                ':daily': DDB_DAILY_PREFIX
            },
            ProjectionExpression='PK'
        )
        
        # Extract unique club names
        clubs = set()
        for item in response.get('Items', []):
            pk = item.get('PK', '')
            if pk.startswith(DDB_CLUB_PREFIX):
                club_name = pk[len(DDB_CLUB_PREFIX):]
                clubs.add(club_name)
        
        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='begins_with(PK, :prefix) AND begins_with(SK, :daily)',
                ExpressionAttributeValues={
                    ':prefix': DDB_CLUB_PREFIX,
                    ':daily': DDB_DAILY_PREFIX
                },
                ProjectionExpression='PK',
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            for item in response.get('Items', []):
                pk = item.get('PK', '')
                if pk.startswith(DDB_CLUB_PREFIX):
                    club_name = pk[len(DDB_CLUB_PREFIX):]
                    clubs.add(club_name)
        
        result = sorted(list(clubs))
        logger.info(f"Retrieved {len(result)} unique clubs")
        return result
        
    except Exception as e:
        logger.error(f"Error getting all clubs: {str(e)}")
        raise
