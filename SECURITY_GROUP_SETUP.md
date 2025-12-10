# AWS EC2 Security Group Configuration

## Required Inbound Rules

Your EC2 instance needs the following inbound rules configured:

### 1. SSH Access (Already configured)
- **Type:** SSH
- **Protocol:** TCP
- **Port Range:** 22
- **Source:** Your IP or 0.0.0.0/0
- **Description:** SSH access

### 2. API Server Access (NEW - Required)
- **Type:** Custom TCP
- **Protocol:** TCP
- **Port Range:** 8000
- **Source:** 0.0.0.0/0 (or restrict to specific IPs)
- **Description:** API Server

## How to Update Security Group

### Via AWS Console:

1. Navigate to **EC2 Dashboard** → **Instances**
2. Select your EC2 instance (running the Discord bot)
3. Click the **Security** tab
4. Click on the **Security Group** link (e.g., sg-xxxxxxxxx)
5. Click **Edit inbound rules** button
6. Click **Add rule**
7. Configure the new rule:
   - **Type:** Custom TCP
   - **Port range:** 8000
   - **Source:** 0.0.0.0/0
   - **Description:** API Server
8. Click **Save rules**

### Via AWS CLI (Alternative):

```bash
# Get your security group ID
SECURITY_GROUP_ID=$(aws ec2 describe-instances \
  --instance-ids YOUR_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' \
  --output text)

# Add port 8000 rule
aws ec2 authorize-security-group-ingress \
  --group-id $SECURITY_GROUP_ID \
  --protocol tcp \
  --port 8000 \
  --cidr 0.0.0.0/0 \
  --description "API Server"
```

## Security Considerations

### Production Recommendations:

1. **Restrict API Access (Recommended):**
   - Instead of `0.0.0.0/0`, use Netlify's IP ranges
   - Or restrict to your office/home IP if static

2. **Use HTTPS (Highly Recommended):**
   - Set up a domain name (e.g., api.yourdomain.com)
   - Use AWS Certificate Manager + Application Load Balancer
   - Or use nginx as reverse proxy with Let's Encrypt SSL

3. **VPC Configuration:**
   - Consider placing EC2 in private subnet
   - Use Application Load Balancer in public subnet
   - More complex but more secure

### Testing Access:

After adding the rule, test from your local machine:

```bash
# Replace YOUR_EC2_IP with your actual EC2 public IP
curl http://YOUR_EC2_IP:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "message": "API is running",
  "timestamp": "2025-12-10T..."
}
```

## Current Configuration Summary

| Service | Port | Protocol | Access |
|---------|------|----------|--------|
| SSH | 22 | TCP | Your IP |
| API Server | 8000 | TCP | Public (0.0.0.0/0) |
| Discord Bot | N/A | N/A | Outbound only |

## Notes

- The Discord bot doesn't need an inbound port (it connects outbound to Discord's servers)
- The API needs port 8000 open for frontend OAuth callbacks
- Both services run on the same EC2 instance
- Both services share the same `.env` file and data files
