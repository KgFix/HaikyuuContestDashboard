# Discord Bot Stuck - Troubleshooting Guide

## Symptoms
Bot gets stuck at "🔍 Analyzing with OCR..." message and never completes.

## Root Causes (t2g.small EC2 Instance)

### 1. **CPU Credit Exhaustion (Most Likely)**
- t2g.small uses burstable CPU credits
- EasyOCR is CPU-intensive and exhausts credits quickly
- Once credits are depleted, CPU is throttled to baseline (5-10%)
- OCR operations become extremely slow or timeout

**Check CPU Credits:**
```bash
# On EC2 instance
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUCreditBalance \
  --dimensions Name=InstanceId,Value=<your-instance-id> \
  --start-time <24-hours-ago> \
  --end-time <now> \
  --period 3600 \
  --statistics Average
```

### 2. **Memory Exhaustion (2GB RAM is insufficient)**
- EasyOCR loads large neural network models (~1.5GB)
- opencv-python + numpy arrays consume additional memory
- Bot likely gets OOM-killed by Linux kernel

**Check Memory Usage:**
```bash
# On EC2 instance
free -h
top -o %MEM
journalctl -u api.service | grep -i "killed\|oom"
```

### 3. **Missing Timeouts on Individual OCR Operations**
- Only the overall analysis has a 60s timeout
- Individual `ocr_reader.readtext()` calls have NO timeouts
- Can hang indefinitely on slow/stuck operations

### 4. **ThreadPoolExecutor Deadlock**
- 4 workers + 3 parallel tasks can cause resource contention
- On memory-constrained systems, threads can deadlock

## Solutions

### Immediate Fix (No Code Changes)

#### Option A: Upgrade EC2 Instance
Upgrade to **t3.medium** or **t3.small** (unlimited mode):
- t3.small: 2 vCPUs, 2GB RAM, better baseline CPU
- t3.medium: 2 vCPUs, 4GB RAM (recommended)
- Enable **Unlimited Mode** to prevent CPU throttling

#### Option B: Add Swap Space (Temporary)
```bash
# On EC2 instance - add 2GB swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Code Fixes (Apply These)

#### Fix 1: Add Individual OCR Timeouts
Wrap all `ocr_reader.readtext()` calls with timeouts:

```python
# In ocr_static_roi() function (around line 643)
def ocr_static_roi(roi: np.ndarray, is_number: bool) -> str:
    if roi is None or roi.size == 0:
        return ""
    
    try:
        ocr_reader = get_ocr_reader()
        enhanced = preprocess_roi(roi)
        allowlist = '0123456789' if is_number else None
        
        # ADD TIMEOUT HERE
        loop = asyncio.new_event_loop()
        future = loop.run_in_executor(
            None,
            lambda: ocr_reader.readtext(
                enhanced,
                detail=0,
                allowlist=allowlist,
                paragraph=True
            )
        )
        results = loop.run_until_complete(
            asyncio.wait_for(future, timeout=15.0)
        )
        loop.close()
        
        return " ".join(results)
    except asyncio.TimeoutError:
        logger.warning("OCR timeout on static ROI")
        return ""
    except Exception as e:
        logger.error(f"OCR error on static ROI: {str(e)}")
        return ""
```

#### Fix 2: Reduce Image Preprocessing Overhead
```python
# Change OCR_SCALE_FACTOR from 1.5 to 1.0
OCR_SCALE_FACTOR = 1.0  # Reduce memory/CPU usage
```

#### Fix 3: Increase Timeout to 120 seconds
```python
# Line 1289 - increase from 60 to 120 seconds
result = await asyncio.wait_for(
    analyze_screenshot_async(image_bytes, debug=False),
    timeout=120.0  # Increased for slow EC2
)
```

#### Fix 4: Add Memory Cleanup
```python
# After analyze_screenshot_async completes
import gc
result = await asyncio.wait_for(
    analyze_screenshot_async(image_bytes, debug=False),
    timeout=120.0
)
gc.collect()  # Force garbage collection
```

### Monitoring

**Add Health Check Endpoint:**
```python
# Add to discord_bot.py
@bot.tree.command(name="botstatus", description="Check bot health")
async def bot_status(interaction: discord.Interaction):
    import psutil
    process = psutil.Process()
    
    await interaction.response.send_message(
        f"✅ **Bot Status**\n"
        f"💾 Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB\n"
        f"🔄 CPU: {process.cpu_percent()}%\n"
        f"⏱️ Uptime: {time.time() - process.create_time():.0f}s",
        ephemeral=True
    )
```

**Check Logs:**
```bash
# On EC2
journalctl -u api.service -f
tail -f /var/log/syslog | grep -i oom
```

## Quick Diagnosis Checklist

1. ✅ Check EC2 instance CPU credits in CloudWatch
2. ✅ Check memory usage with `free -h`
3. ✅ Check if process is being OOM-killed: `dmesg | grep -i kill`
4. ✅ Check bot logs for timeout errors
5. ✅ Verify EasyOCR model files are downloaded: `ls ~/.EasyOCR/`

## Recommended Action Plan

**Short-term (Immediate):**
1. Restart bot service: `sudo systemctl restart api.service`
2. Check CPU credits - if low, wait for them to replenish
3. Add swap space as temporary relief

**Long-term (Permanent Fix):**
1. Upgrade to t3.medium with Unlimited Mode
2. Apply code fixes for timeouts and memory management
3. Consider using AWS Lambda with pre-warmed containers for OCR
4. Add monitoring/alerting for resource usage

## Alternative: Lighter OCR Solution

If resource constraints persist, consider replacing EasyOCR with:
- **Tesseract OCR** (much lighter, ~50MB vs 1.5GB)
- **Google Cloud Vision API** (offload processing)
- **AWS Textract** (serverless, pay-per-use)
