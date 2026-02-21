@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "EXITCODE=0"

cd /d "%~dp0"

echo ============================================================
echo LLamaSharp Unity CUDA Setup (Windows, CUDA12, LLamaSharp 0.10.0)
echo ============================================================
echo.

if not exist "Assets\packages.config" (
  echo [ERROR] Could not find Assets\packages.config.
  echo Run this .bat from the Unity project root.
  set "EXITCODE=1"
  goto :cleanup
)

where powershell >nul 2>nul
if errorlevel 1 (
  echo [ERROR] PowerShell is required but not found.
  set "EXITCODE=1"
  goto :cleanup
)

set "CUDA_BACKEND_VERSION=0.10.0"
set "CUDA_BACKEND_ID=llamasharp.backend.cuda12"
set "NUGET_URL=https://api.nuget.org/v3-flatcontainer/%CUDA_BACKEND_ID%/%CUDA_BACKEND_VERSION%/%CUDA_BACKEND_ID%.%CUDA_BACKEND_VERSION%.nupkg"

for /f %%I in ('powershell -NoProfile -Command "[Guid]::NewGuid().ToString('N')"') do set "RUN_ID=%%I"
set "TMP_DIR=%TEMP%\llamasharp_cuda_setup_%RUN_ID%"
set "PKG_FILE=%TMP_DIR%\backend.nupkg"
set "PKG_ZIP=%TMP_DIR%\backend.zip"
set "PKG_EXTRACT=%TMP_DIR%\pkg"

mkdir "%TMP_DIR%" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Could not create temp folder: %TMP_DIR%
  set "EXITCODE=1"
  goto :cleanup
)

echo [1/5] Ensuring LLamaSharp.Backend.Cuda12 package entry exists...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; $path='Assets/packages.config'; [xml]$xml=Get-Content -LiteralPath $path; $exists=$false; foreach($p in $xml.packages.package){ if($p.id -eq 'LLamaSharp.Backend.Cuda12'){ $exists=$true; break } }; if(-not $exists){ $node=$xml.CreateElement('package'); $node.SetAttribute('id','LLamaSharp.Backend.Cuda12'); $node.SetAttribute('version','0.10.0'); [void]$xml.packages.AppendChild($node); $settings=New-Object System.Xml.XmlWriterSettings; $settings.Indent=$true; $settings.OmitXmlDeclaration=$false; $settings.Encoding=New-Object System.Text.UTF8Encoding($false); $writer=[System.Xml.XmlWriter]::Create((Resolve-Path $path),$settings); $xml.Save($writer); $writer.Close(); Write-Host '  Added package entry.' } else { Write-Host '  Package entry already present.' }"
if errorlevel 1 goto :fail

echo [2/5] Downloading LLamaSharp CUDA backend package...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%NUGET_URL%' -OutFile '%PKG_FILE%'"
if errorlevel 1 goto :fail

echo [3/5] Extracting CUDA backend package...
copy /y "%PKG_FILE%" "%PKG_ZIP%" >nul
if errorlevel 1 goto :fail
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Expand-Archive -LiteralPath '%PKG_ZIP%' -DestinationPath '%PKG_EXTRACT%' -Force"
if errorlevel 1 goto :fail

if not exist "%PKG_EXTRACT%\runtimes\win-x64\native\cuda12\llama.dll" (
  echo [ERROR] Could not find llama.dll in downloaded backend package.
  goto :fail
)

echo [4/5] Installing backend DLLs into Unity project...
mkdir "Assets\Plugins\x86_64" >nul 2>nul
copy /y "%PKG_EXTRACT%\runtimes\win-x64\native\cuda12\llama.dll" "Assets\Plugins\x86_64\llama.dll" >nul 2>nul
if errorlevel 1 (
  if exist "Assets\Plugins\x86_64\llama.dll" (
    echo   [WARN] Could not overwrite Assets\Plugins\x86_64\llama.dll ^(likely locked by Unity^). Using existing file.
  ) else (
    echo [ERROR] Could not install Assets\Plugins\x86_64\llama.dll.
    goto :fail
  )
) else (
  echo   Installed Assets\Plugins\x86_64\llama.dll
)

copy /y "%PKG_EXTRACT%\runtimes\win-x64\native\cuda12\llama.dll" "Assets\libllama.dll" >nul 2>nul
if errorlevel 1 (
  if exist "Assets\libllama.dll" (
    echo   [WARN] Could not overwrite Assets\libllama.dll ^(likely locked by Unity^). Using existing file.
  ) else (
    echo   [WARN] Could not write Assets\libllama.dll.
  )
) else (
  echo   Replaced Assets\libllama.dll with CUDA backend.
)

echo [5/5] Locating and copying CUDA runtime dependencies...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; $root=(Resolve-Path '.').Path; $dests=@((Join-Path $root 'Assets'), (Join-Path $root 'Assets\\Plugins\\x86_64')); $dlls=@('cudart64_12.dll','cublas64_12.dll','cublasLt64_12.dll'); foreach($dll in $dlls){ $candidates=@(); $whereOut=& where.exe $dll 2>$null; if($LASTEXITCODE -eq 0 -and $whereOut){ $candidates += ($whereOut -split '\\r?\\n' | Where-Object { $_ -and (Test-Path $_) }) }; $cudaDirs=Get-ChildItem -Path 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA' -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending; foreach($dir in $cudaDirs){ $p=Join-Path $dir.FullName ('bin\\' + $dll); if(Test-Path $p){ $candidates += $p } }; $src=$candidates | Select-Object -First 1; if(-not $src){ Write-Host ('  MISSING: ' + $dll); continue }; foreach($dest in $dests){ try { Copy-Item -LiteralPath $src -Destination (Join-Path $dest $dll) -Force } catch { Write-Host ('  WARN: could not overwrite ' + (Join-Path $dest $dll) + ' (' + $_.Exception.Message + ')') } }; Write-Host ('  COPIED: ' + $dll + ' <- ' + $src) }"
if errorlevel 1 goto :fail

set "MISSING=0"
for %%F in (cudart64_12.dll cublas64_12.dll cublasLt64_12.dll) do (
  if not exist "Assets\Plugins\x86_64\%%F" (
    echo [WARN] Missing %%F in Assets\Plugins\x86_64
    set "MISSING=1"
  )
)

echo.
echo Done.
if "%MISSING%"=="1" (
  echo Some CUDA runtime DLLs are missing.
  echo Install NVIDIA CUDA Toolkit 12.x, then rerun this script.
  echo https://developer.nvidia.com/cuda-downloads
) else (
  echo CUDA DLLs are in place.
)

echo.
echo Next:
echo 1) Close and reopen Unity (if it was running).
echo 2) Let assets reimport.
echo 3) In NuGetForUnity, Restore packages if prompted.
echo 4) Enter Play mode and check Console for LLama backend logs.
goto :cleanup

:fail
echo.
echo [ERROR] CUDA setup failed.
echo See messages above for details.
set "EXITCODE=1"

:cleanup
if defined TMP_DIR if exist "%TMP_DIR%" rmdir /s /q "%TMP_DIR%" >nul 2>nul
exit /b %EXITCODE%

