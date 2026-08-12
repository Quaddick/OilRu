<#
    Автоматическая установка LAMPA и TorrServe на Android TV по локальной сети (Windows).

    Скрипт сам находит/скачивает adb, ищет телевизор в сети, подключается,
    качает свежие APK с GitHub-релизов и ставит их.

    Запуск в PowerShell (компьютер должен быть в той же сети, что и ТВ):
        powershell -ExecutionPolicy Bypass -File install-tv-apps.ps1
        powershell -ExecutionPolicy Bypass -File install-tv-apps.ps1 -TvIp 192.168.1.50
        powershell -ExecutionPolicy Bypass -File install-tv-apps.ps1 -PairAddr 192.168.1.50:41234 -PairCode 123456
#>
[CmdletBinding()]
param(
    [string]$TvIp     = "",
    [string]$PairAddr = "",
    [string]$PairCode = "",
    [int]   $AdbPort  = 5555
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$LampaRepo     = 'lampa-app/LAMPA'
$LampaPkg      = 'top.rootu.lampa'
$TorrServeRepo = 'YouROK/TorrServe'
$TorrServePkg  = 'ru.yourok.torrserve'

$WorkDir = Join-Path $env:TEMP 'atv-install'
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

function Info($m) { Write-Host "==> $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[!] $m"  -ForegroundColor Yellow }
function Dim($m)  { Write-Host "    $m"  -ForegroundColor DarkGray }
function Die($m)  { Write-Host "[x] $m"  -ForegroundColor Red; exit 1 }

# ------------------------------------------------------------------- adb ----

function Get-Adb {
    $cmd = Get-Command adb -ErrorAction SilentlyContinue
    if ($cmd) { Info "adb найден: $($cmd.Source)"; return $cmd.Source }

    $local = Join-Path $WorkDir 'platform-tools\adb.exe'
    if (Test-Path $local) { Info "adb найден: $local"; return $local }

    Info 'adb не найден — скачиваю Android platform-tools…'
    $zip = Join-Path $WorkDir 'platform-tools.zip'
    Invoke-WebRequest -Uri 'https://dl.google.com/android/repository/platform-tools-latest-windows.zip' `
                      -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $WorkDir -Force
    if (-not (Test-Path $local)) { Die 'adb не распаковался.' }
    Info "adb установлен: $local"
    return $local
}

# ----------------------------------------------------------------- поиск ----

function Get-LocalSubnet {
    $ip = Get-NetIPConfiguration -ErrorAction SilentlyContinue |
          Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' } |
          Select-Object -First 1 -ExpandProperty IPv4Address |
          Select-Object -First 1 -ExpandProperty IPAddress
    if (-not $ip) {
        $ip = (Get-WmiObject Win32_NetworkAdapterConfiguration |
               Where-Object { $_.IPEnabled -and $_.DefaultIPGateway } |
               Select-Object -First 1).IPAddress[0]
    }
    if (-not $ip) { return $null }
    return ($ip -split '\.')[0..2] -join '.'
}

function Find-Tv {
    $subnet = Get-LocalSubnet
    if (-not $subnet) { Die 'Не удалось определить подсеть. Укажи адрес вручную: -TvIp 192.168.x.x' }

    Info "Сканирую сеть $subnet.0/24 на открытый ADB-порт $AdbPort…"
    Dim '(это занимает несколько секунд)'

    # асинхронные TCP-пробы: 254 адреса разом, ждём общий таймаут
    $probes = foreach ($i in 1..254) {
        $client = New-Object Net.Sockets.TcpClient
        [pscustomobject]@{
            Host   = "$subnet.$i"
            Client = $client
            Async  = $client.BeginConnect("$subnet.$i", $AdbPort, $null, $null)
        }
    }
    Start-Sleep -Milliseconds 1500

    $hits = foreach ($p in $probes) {
        $ok = $false
        try { if ($p.Async.IsCompleted) { $p.Client.EndConnect($p.Async); $ok = $p.Client.Connected } } catch { }
        try { $p.Client.Close() } catch { }
        if ($ok) { $p.Host }
    }

    if (-not $hits) {
        Die @"
Устройство с открытым ADB не найдено.
  Проверь на телевизоре: Настройки → Для разработчиков → «Отладка по USB» / «Отладка по сети» включена,
  телевизор в той же сети Wi-Fi, и запусти скрипт с явным адресом: -TvIp <IP телевизора>
"@
    }
    if ($hits.Count -gt 1) {
        Warn 'Найдено несколько устройств с открытым ADB:'
        $hits | ForEach-Object { Dim $_ }
        Warn 'Беру первое. Если это не телевизор — перезапусти с -TvIp <IP>.'
    }
    return @($hits)[0]
}

# ------------------------------------------------------------ соединение ----

function Connect-Tv([string]$Adb, [string]$Target) {
    if ($PairAddr) {
        Info "Сопряжение (pairing) с $PairAddr…"
        & $Adb pair $PairAddr $PairCode
        if ($LASTEXITCODE -ne 0) {
            Die 'Сопряжение не удалось. Код одноразовый — открой на ТВ «Отладка по Wi-Fi → Подключить с кодом» и возьми свежий.'
        }
    }

    Info "Подключаюсь к $Target…"
    & $Adb disconnect $Target 2>&1 | Out-Null
    & $Adb connect    $Target 2>&1 | Out-Null

    $state = ''
    for ($i = 1; $i -le 45; $i++) {
        $state = (& $Adb -s $Target get-state 2>&1 | Out-String).Trim()
        if ($state -eq 'device') { Info 'Подключено, отладка разрешена.'; return }
        if ($state -eq 'unauthorized' -and $i -eq 1) {
            Warn 'На экране телевизора появился запрос — нажми «Разрешить отладку по сети» (галочка «Всегда»). Жду…'
        }
        if ($state -ne 'unauthorized') { & $Adb connect $Target 2>&1 | Out-Null }
        Start-Sleep -Seconds 2
    }

    Die @"
Не удалось выйти в состояние 'device' (последнее состояние: $state).
  Частые причины: не нажали «Разрешить» на телевизоре; ТВ в другой сети; на Android 11+/Google TV
  нужен режим «Отладка по Wi-Fi» с сопряжением — тогда запусти с -PairAddr <IP>:<порт> -PairCode <код>
"@
}

# --------------------------------------------------------------- релизы ----

function Get-LatestApkUrl([string]$Repo) {
    try {
        $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" `
                                 -Headers @{ 'User-Agent' = 'atv-installer' } -UseBasicParsing
    } catch { return $null }

    $apks = @($rel.assets | Where-Object { $_.name -like '*.apk' } | ForEach-Object { $_.browser_download_url })
    if (-not $apks) { return $null }

    $pick = $apks | Where-Object { $_ -match 'universal' } | Select-Object -First 1
    if (-not $pick) { $pick = $apks | Where-Object { $_ -match 'arm64|aarch64' } | Select-Object -First 1 }
    if (-not $pick) { $pick = $apks[0] }
    return $pick
}

# ------------------------------------------------------------- установка ----

function Install-App([string]$Adb, [string]$Title, [string]$Repo, [string]$Pkg, [string]$Target) {
    Info "${Title}: определяю последний релиз ($Repo)…"
    $url = Get-LatestApkUrl $Repo
    if (-not $url) { Warn "${Title}: не удалось получить ссылку на APK (GitHub недоступен?). Пропускаю."; return $false }

    $file = Join-Path $WorkDir (Split-Path $url -Leaf)
    Dim (Split-Path $url -Leaf)
    if (Test-Path $file) {
        Dim 'уже скачан, использую локальную копию'
    } else {
        Invoke-WebRequest -Uri $url -OutFile $file -UseBasicParsing
    }

    Info "${Title}: устанавливаю на телевизор…"
    & $Adb -s $Target install -r -g $file
    if ($LASTEXITCODE -ne 0) {
        & $Adb -s $Target install -r $file
        if ($LASTEXITCODE -ne 0) {
            Warn "${Title}: установка не удалась. Если ошибка INSTALL_FAILED_UPDATE_INCOMPATIBLE — удали старую версию:"
            Dim "$Adb -s $Target uninstall $Pkg"
            return $false
        }
    }

    $installed = (& $Adb -s $Target shell pm list packages 2>&1 | Out-String)
    if ($installed -match [Regex]::Escape("package:$Pkg")) {
        Info "${Title}: установлен OK ($Pkg)"
    } else {
        Warn "${Title}: APK поставился, но пакет $Pkg в списке не найден — проверь приложение на ТВ вручную."
    }
    return $true
}

# ------------------------------------------------------------------ main ----

$Adb = Get-Adb
& $Adb start-server 2>&1 | Out-Null

if (-not $TvIp) { $TvIp = Find-Tv }
if ($TvIp -notmatch ':') { $TvIp = "${TvIp}:$AdbPort" }
Info "Телевизор: $TvIp"

Connect-Tv $Adb $TvIp

$model = (& $Adb -s $TvIp shell getprop ro.product.model 2>&1 | Out-String).Trim()
$andv  = (& $Adb -s $TvIp shell getprop ro.build.version.release 2>&1 | Out-String).Trim()
if ($model) { Dim "модель: $model, Android $andv" }

$ok = $true
if (-not (Install-App $Adb 'LAMPA'     $LampaRepo     $LampaPkg     $TvIp)) { $ok = $false }
if (-not (Install-App $Adb 'TorrServe' $TorrServeRepo $TorrServePkg $TvIp)) { $ok = $false }

Write-Host ''
if ($ok) {
    Info 'Готово. Оба приложения на телевизоре — ищи их в списке приложений.'
    Dim 'Дальше: запусти TorrServe (поднимет сервер на 127.0.0.1:8090), затем в LAMPA →'
    Dim 'Настройки → Торренты → TorrServe, адрес http://127.0.0.1:8090'
} else {
    Warn 'Завершено с ошибками — смотри сообщения выше.'
}

& $Adb disconnect $TvIp 2>&1 | Out-Null
if ($ok) { exit 0 } else { exit 1 }
