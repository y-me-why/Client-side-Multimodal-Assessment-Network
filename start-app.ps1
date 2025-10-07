# AI Interview Prep - Complete Application Startup Script
# This script starts all services: Frontend, Backend, and AI Services

Write-Host "🚀 Starting AI Interview Prep Application..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if we're in the right directory
if (!(Test-Path "package.json")) {
    Write-Host "❌ Error: Please run this script from the ai-interview-prep root directory" -ForegroundColor Red
    exit 1
}

Write-Host "📝 Checking dependencies..." -ForegroundColor Yellow

# Function to check if a port is in use
function Test-Port {
    param([int]$Port)
    try {
        $connection = Test-NetConnection -ComputerName localhost -Port $Port -InformationLevel Quiet
        return $connection
    } catch {
        return $false
    }
}

# Kill any existing services on our ports
Write-Host "🧹 Cleaning up existing services..." -ForegroundColor Yellow
Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object {$_.ProcessName -eq "node"} | Stop-Process -Force
Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*main.py*"} | Stop-Process -Force

Start-Sleep -Seconds 2

# Start Backend (Express.js on port 5000)
Write-Host "🗄️  Starting Backend Server..." -ForegroundColor Green
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "cd 'backend'; Write-Host '🗄️ Backend Server Running on http://localhost:5000' -ForegroundColor Green; npm start" -WindowStyle Normal

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start AI Services (FastAPI on port 8001) 
Write-Host "🧠 Starting AI Services..." -ForegroundColor Magenta
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "cd 'ai-services'; Write-Host '🧠 AI Services Running on http://localhost:8001' -ForegroundColor Magenta; python main.py" -WindowStyle Normal

# Wait for AI services to start
Start-Sleep -Seconds 5

# Start Frontend (React on port 3000)
Write-Host "🌐 Starting Frontend..." -ForegroundColor Blue
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "cd 'frontend'; Write-Host '🌐 Frontend Running on http://localhost:3000' -ForegroundColor Blue; npm start" -WindowStyle Normal

Write-Host ""
Write-Host "✅ All services are starting up!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "📋 Service URLs:" -ForegroundColor White
Write-Host "   🌐 Frontend:     http://localhost:3000" -ForegroundColor Blue
Write-Host "   🗄️  Backend API:  http://localhost:5000" -ForegroundColor Green
Write-Host "   🧠 AI Services:  http://localhost:8001" -ForegroundColor Magenta
Write-Host "   📊 Health Check: http://localhost:5000/health" -ForegroundColor Yellow
Write-Host ""
Write-Host "🔧 Features Available:" -ForegroundColor White
Write-Host "   ✅ Real-time facial analysis using MediaPipe" -ForegroundColor Green
Write-Host "   ✅ Voice analysis with librosa" -ForegroundColor Green  
Write-Host "   ✅ Gemini AI question generation (set GOOGLE_API_KEY)" -ForegroundColor Yellow
Write-Host "   ✅ WebSocket live analysis on ws://localhost:8001/ws/live-analysis/{session_id}" -ForegroundColor Green
Write-Host "   ✅ Resume upload and analysis" -ForegroundColor Green
Write-Host ""
Write-Host "⚠️  Note: Set GOOGLE_API_KEY in ai-services/.env for full AI features" -ForegroundColor Yellow
Write-Host ""
Write-Host "🛑 To stop all services, close the PowerShell windows or run:" -ForegroundColor Red
Write-Host "   Get-Process -Name 'node','python' | Stop-Process -Force" -ForegroundColor Red
Write-Host ""
Write-Host "🎉 Application is ready! Open http://localhost:3000 to start" -ForegroundColor Cyan

# Wait for user input to keep the main window open
Write-Host "Press any key to exit this window (services will continue running)..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")