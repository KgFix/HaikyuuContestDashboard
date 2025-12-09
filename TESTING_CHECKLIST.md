# Testing Checklist

Use this checklist to verify everything works correctly.

## ✅ Backend Setup

- [ ] Updated `backend/.env` with all required variables
- [ ] Generated JWT secret key with `openssl rand -hex 32`
- [ ] Installed new Python dependencies (`pip install -r requirements.txt`)
- [ ] Verified DynamoDB connection still works
- [ ] `activated_channels.json` exists and has correct data

## ✅ Discord OAuth Setup

- [ ] Created Discord OAuth application at https://discord.com/developers/applications
- [ ] Copied Client ID to `.env`
- [ ] Copied Client Secret to `.env` (kept secure)
- [ ] Added redirect URL: `http://localhost:8000/api/auth/discord/callback`
- [ ] Verified scopes are correct: `identify`, `guilds`, `guilds.members.read`

## ✅ Backend API Testing

### Start API Server
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.api.main:app --reload
```

### Test Endpoints

- [ ] **Health Check:**
  ```powershell
  curl http://localhost:8000/api/health
  ```
  Expected: `{"status":"healthy","message":"API is running",...}`

- [ ] **API Documentation:**
  - Open: http://localhost:8000/docs
  - Should see Swagger UI with all endpoints

- [ ] **OAuth Login Redirect:**
  - Open: http://localhost:8000/api/auth/discord/login
  - Should redirect to Discord

- [ ] **No Errors in Logs:**
  - Check terminal for any errors
  - Should see "Application startup complete"

## ✅ Frontend Setup

- [ ] Created `frontend/.env` with `VITE_API_BASE_URL=http://localhost:8000`
- [ ] Installed dependencies: `npm install`
- [ ] Installed react-router-dom: `npm install react-router-dom`
- [ ] No TypeScript errors: `npm run build` (optional)

## ✅ Frontend Testing

### Start Frontend Server
```powershell
cd frontend
npm run dev
```

### Manual Tests

- [ ] **Home Page Loads:**
  - Open: http://localhost:5173
  - Should see login page (if not logged in)
  - OR dashboard (if previously logged in)

- [ ] **Login Button Exists:**
  - Should see "Login with Discord" button
  - Button has Discord logo

- [ ] **Click Login:**
  - Redirects to Discord OAuth
  - Shows correct application name
  - Shows requested permissions
  - Has "Authorize" button

## ✅ Complete Authentication Flow

### Step-by-Step Test

1. **Start Fresh:**
   - [ ] Clear browser localStorage (F12 → Application → Local Storage → Clear)
   - [ ] Close all Discord authorization prompts

2. **Click "Login with Discord":**
   - [ ] Redirects to Discord
   - [ ] URL starts with `https://discord.com/api/oauth2/authorize`
   - [ ] Shows your application name

3. **Click "Authorize":**
   - [ ] Redirects back to your app
   - [ ] URL briefly shows `/auth/callback?token=...`
   - [ ] Then redirects to `/` (dashboard)

4. **Verify Logged In:**
   - [ ] Dashboard loads
   - [ ] See user info in header (username, avatar)
   - [ ] See number of clubs user has access to
   - [ ] See "Logout" button

5. **Check localStorage:**
   - [ ] F12 → Application → Local Storage
   - [ ] Should have `haikyuu_auth_token`
   - [ ] Should have `haikyuu_user_info`

6. **Test API Calls:**
   - [ ] F12 → Network tab
   - [ ] Select a club from dropdown
   - [ ] See API calls with `Authorization: Bearer ...` header
   - [ ] Calls return 200 status
   - [ ] Charts display data

## ✅ Authorization Testing

### Test Club Access

- [ ] **User with Access:**
  1. Login with Discord account that has activated role
  2. Should see club in "Your Clubs" list
  3. Can view that club's data
  4. Charts load successfully

- [ ] **User without Access:**
  1. Login with different Discord account (without role)
  2. Should NOT see club in "Your Clubs" list
  3. Cannot access club data
  4. Attempting direct API call returns 403

### Test Token Expiration

- [ ] **Manual Token Deletion:**
  1. F12 → Application → Local Storage
  2. Delete `haikyuu_auth_token`
  3. Refresh page
  4. Should redirect to login page

## ✅ Error Handling

### Test Error Scenarios

- [ ] **Invalid Token:**
  1. F12 → Application → Local Storage
  2. Edit `haikyuu_auth_token` to invalid value
  3. Refresh page
  4. Should redirect to login (not crash)

- [ ] **API Offline:**
  1. Stop API server
  2. Try to login
  3. Should show error message (not crash)

- [ ] **Discord OAuth Failure:**
  1. Click "Authorize" then immediately click "Cancel"
  2. Should handle gracefully (error message or redirect)

## ✅ UI/UX Testing

- [ ] **Login Page:**
  - [ ] Looks good on desktop
  - [ ] Looks good on mobile (resize browser)
  - [ ] Smooth animations
  - [ ] Clear instructions

- [ ] **Dashboard:**
  - [ ] User info displays correctly
  - [ ] Logout button works
  - [ ] Club selector shows correct clubs
  - [ ] Charts render properly
  - [ ] Loading states work
  - [ ] Error messages are clear

- [ ] **Logout Flow:**
  1. Click logout
  2. localStorage cleared
  3. Redirects to login page
  4. Can login again successfully

## ✅ Security Checks

- [ ] **JWT Token:**
  - [ ] Copy token from localStorage
  - [ ] Paste into https://jwt.io
  - [ ] Verify payload has: `sub`, `username`, `clubs`, `exp`
  - [ ] Verify expiration is in future

- [ ] **CORS:**
  - [ ] Check Network tab in browser
  - [ ] API responses have CORS headers
  - [ ] No CORS errors in console

- [ ] **Secrets:**
  - [ ] `.env` is in `.gitignore`
  - [ ] No secrets in git history
  - [ ] Discord Client Secret never exposed to frontend

## ✅ Backend Logs Review

Check logs for:
- [ ] No ERROR level messages
- [ ] Successful OAuth token exchange
- [ ] JWT token creation
- [ ] DynamoDB queries executing
- [ ] User club access determination

Common log entries to verify:
```
INFO: Application startup complete
INFO: Loaded 1 activated channels
INFO: User has access to 1 clubs: ['@BotTestRole']
INFO: Retrieved 5 daily records for club @BotTestRole
```

## ✅ Data Verification

### Check DynamoDB Data

Using AWS Console or CLI:
- [ ] USER# records exist
- [ ] CLUB# records exist
- [ ] Recent submissions have correct ClubName
- [ ] ClubName matches activated_channels.json

### Test with Real Data

- [ ] Submit screenshot via Discord bot
- [ ] Verify appears in DynamoDB
- [ ] Login to dashboard
- [ ] Verify data shows in charts
- [ ] Data updates in real-time (refresh)

## ✅ Cross-Browser Testing

Test in multiple browsers:
- [ ] Chrome/Edge
- [ ] Firefox
- [ ] Safari (if available)

All should work identically.

## ✅ Performance

- [ ] Login flow completes in < 5 seconds
- [ ] Dashboard loads in < 2 seconds
- [ ] API responses < 1 second
- [ ] No memory leaks (refresh 10+ times)

## 🐛 Common Issues & Solutions

### "Missing required environment variables"
→ Check `backend/.env` has all variables from `.env.example`

### "Discord OAuth not configured"
→ Verify `DISCORD_CLIENT_ID` and `DISCORD_CLIENT_SECRET` are set

### "Invalid redirect_uri"
→ Check Discord Developer Portal has exact URL: `http://localhost:8000/api/auth/discord/callback`

### "You do not have access to club"
→ Verify user has role mentioned in `activated_channels.json`

### "CORS error"
→ Check `FRONTEND_URL` in `backend/.env` matches frontend URL

### Charts don't load
→ Check Network tab for API errors, verify token is valid

## ✅ Ready for Deployment?

Before deploying to AWS, ensure:
- [ ] All local tests pass
- [ ] No console errors
- [ ] No TypeScript errors
- [ ] Documentation reviewed
- [ ] `.env.example` updated
- [ ] README updated
- [ ] Git committed (excluding secrets)

## 📝 Test Results

Document your test results:

**Date:** ___________
**Tester:** ___________
**Environment:** Local / Staging / Production

**Overall Status:** ✅ Pass / ❌ Fail / ⚠️ Issues

**Issues Found:**
1. 
2. 
3. 

**Notes:**


---

**Good luck with testing! 🚀**

If all checks pass, you're ready to deploy to AWS EC2!
