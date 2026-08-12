#!/usr/bin/env bash
#
# Автоматическая установка LAMPA и TorrServe на Android TV по локальной сети.
#
# Скрипт сам:
#   1. находит или скачивает adb (Android platform-tools);
#   2. находит телевизор в локальной сети (или берёт IP из --tv);
#   3. подключается к нему по ADB;
#   4. скачивает свежие APK с GitHub-релизов;
#   5. устанавливает их и проверяет результат.
#
# Запуск (Linux / macOS), с компьютера в той же сети, что и телевизор:
#   bash install-tv-apps.sh
#   bash install-tv-apps.sh --tv 192.168.1.50
#   bash install-tv-apps.sh --pair 192.168.1.50:41234 123456   # Android 11+ / Google TV
#
set -euo pipefail

# ---------------------------------------------------------------- параметры --

LAMPA_REPO="lampa-app/LAMPA"
LAMPA_PKG="top.rootu.lampa"
TORRSERVE_REPO="YouROK/TorrServe"
TORRSERVE_PKG="ru.yourok.torrserve"

ADB_PORT=5555
WORKDIR="${ATV_WORKDIR:-${TMPDIR:-/tmp}/atv-install}"

TV_IP=""
PAIR_ADDR=""
PAIR_CODE=""

# ------------------------------------------------------------------ вывод ----

c_ok=$'\033[32m'; c_warn=$'\033[33m'; c_err=$'\033[31m'; c_dim=$'\033[2m'; c_off=$'\033[0m'
[[ -t 1 ]] || { c_ok=""; c_warn=""; c_err=""; c_dim=""; c_off=""; }

info() { printf '%s==>%s %s\n' "$c_ok"   "$c_off" "$*"; }
warn() { printf '%s[!]%s %s\n' "$c_warn" "$c_off" "$*"; }
dim()  { printf '%s    %s%s\n' "$c_dim"  "$*"     "$c_off"; }
die()  { printf '%s[x]%s %s\n' "$c_err"  "$c_off" "$*" >&2; exit 1; }

# ------------------------------------------------------------------ разбор ---

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tv)   TV_IP="${2:-}";      shift 2 ;;
    --pair) PAIR_ADDR="${2:-}"; PAIR_CODE="${3:-}"; shift 3 ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) die "Неизвестный аргумент: $1 (см. --help)" ;;
  esac
done

mkdir -p "$WORKDIR"

# --------------------------------------------------------------------- adb ---

ensure_adb() {
  if command -v adb >/dev/null 2>&1; then
    ADB="$(command -v adb)"
    info "adb найден: $ADB"
    return
  fi
  if [[ -x "$WORKDIR/platform-tools/adb" ]]; then
    ADB="$WORKDIR/platform-tools/adb"
    info "adb найден: $ADB"
    return
  fi

  local os_tag
  case "$(uname -s)" in
    Linux)  os_tag="linux"  ;;
    Darwin) os_tag="darwin" ;;
    *) die "Неподдерживаемая ОС для автоскачивания adb. Установи Android platform-tools вручную." ;;
  esac

  info "adb не найден — скачиваю Android platform-tools…"
  command -v unzip >/dev/null 2>&1 || die "Нужна утилита unzip (apt install unzip / brew install unzip)."
  curl -fL --progress-bar \
    "https://dl.google.com/android/repository/platform-tools-latest-${os_tag}.zip" \
    -o "$WORKDIR/platform-tools.zip" \
    || die "Не удалось скачать platform-tools."
  unzip -q -o "$WORKDIR/platform-tools.zip" -d "$WORKDIR"
  ADB="$WORKDIR/platform-tools/adb"
  [[ -x "$ADB" ]] || die "adb не распаковался в $WORKDIR."
  info "adb установлен: $ADB"
}

# ------------------------------------------------------------------ поиск ----

local_subnet() {
  local ip=""
  case "$(uname -s)" in
    Linux)
      ip="$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oE 'src [0-9.]+' | awk '{print $2}')"
      [[ -n "$ip" ]] || ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
      ;;
    Darwin)
      local dev
      dev="$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')"
      [[ -n "$dev" ]] && ip="$(ipconfig getifaddr "$dev" 2>/dev/null || true)"
      [[ -n "$ip" ]] || ip="$(ipconfig getifaddr en0 2>/dev/null || true)"
      ;;
  esac
  [[ -n "$ip" ]] || return 1
  echo "${ip%.*}"
}

# Проверка одного адреса: открыт ли TCP-порт ADB.
probe_one() {
  local host="$1"
  if command -v nc >/dev/null 2>&1; then
    nc -z -w 1 "$host" "$ADB_PORT" >/dev/null 2>&1 && echo "$host"
  elif command -v timeout >/dev/null 2>&1; then
    timeout 1 bash -c "exec 3<>/dev/tcp/$host/$ADB_PORT" >/dev/null 2>&1 && echo "$host"
  else
    (exec 3<>"/dev/tcp/$host/$ADB_PORT") >/dev/null 2>&1 && echo "$host"
  fi
  return 0
}

discover_tv() {
  local subnet
  subnet="$(local_subnet)" || die "Не удалось определить свою подсеть. Укажи IP вручную: --tv 192.168.x.x"

  info "Сканирую сеть ${subnet}.0/24 на открытый ADB-порт ${ADB_PORT}…"
  dim "(это занимает несколько секунд)"

  local found="$WORKDIR/found.txt"
  : > "$found"

  local i
  for i in $(seq 1 254); do
    probe_one "${subnet}.$i" >> "$found" &
    # не плодим больше 64 параллельных проб разом (wait -n есть не везде — отсюда sleep-фолбэк)
    while [[ "$(jobs -rp | wc -l)" -ge 64 ]]; do
      wait -n 2>/dev/null || sleep 0.1
    done
  done
  wait

  local hits
  hits="$(sort -u "$found" | grep -E '^[0-9]+\.' || true)"
  [[ -n "$hits" ]] || die "Устройство с открытым ADB не найдено.
  Проверь на телевизоре: Настройки → Для разработчиков → «Отладка по USB» / «Отладка по сети» включена,
  телевизор в той же сети Wi-Fi, и запусти скрипт с явным адресом: --tv <IP телевизора>"

  local count
  count="$(printf '%s\n' "$hits" | wc -l | tr -d ' ')"
  if [[ "$count" -gt 1 ]]; then
    warn "Найдено несколько устройств с открытым ADB:"
    printf '%s\n' "$hits" | sed 's/^/      /'
    warn "Беру первое. Если это не телевизор — перезапусти с --tv <IP>."
  fi
  printf '%s\n' "$hits" | head -n1
}

# --------------------------------------------------------------- соединение --

connect_tv() {
  local target="$1"

  if [[ -n "$PAIR_ADDR" ]]; then
    info "Сопряжение (pairing) с $PAIR_ADDR…"
    "$ADB" pair "$PAIR_ADDR" "$PAIR_CODE" \
      || die "Сопряжение не удалось. Код одноразовый — открой на ТВ «Отладка по Wi-Fi → Подключить с кодом» и возьми свежий."
  fi

  info "Подключаюсь к $target…"
  "$ADB" disconnect "$target" >/dev/null 2>&1 || true
  "$ADB" connect "$target" >/dev/null 2>&1 || true

  local state="" i
  for i in $(seq 1 45); do
    state="$("$ADB" -s "$target" get-state 2>/dev/null || echo offline)"
    case "$state" in
      device)
        info "Подключено, отладка разрешена."
        return 0 ;;
      unauthorized)
        [[ $i -eq 1 ]] && warn "На экране телевизора появился запрос — нажми «Разрешить отладку по сети» (можно поставить галочку «Всегда»). Жду…"
        ;;
      *)
        "$ADB" connect "$target" >/dev/null 2>&1 || true
        ;;
    esac
    sleep 2
  done

  die "Не удалось выйти в состояние 'device' (последнее состояние: $state).
  Частые причины: не нажали «Разрешить» на телевизоре; ТВ в другой сети; на Android 11+/Google TV
  нужен режим «Отладка по Wi-Fi» с сопряжением — тогда запусти: --pair <IP>:<порт сопряжения> <код>"
}

# ----------------------------------------------------------------- релизы ----

# Достаёт ссылку на .apk из последнего релиза GitHub без зависимости от jq.
latest_apk_url() {
  local repo="$1" json urls
  json="$(curl -fsSL -H 'Accept: application/vnd.github+json' \
          "https://api.github.com/repos/${repo}/releases/latest" 2>/dev/null)" \
    || { echo ""; return 1; }

  urls="$(printf '%s' "$json" \
          | grep -oE '"browser_download_url"[[:space:]]*:[[:space:]]*"[^"]+\.apk"' \
          | sed -E 's/.*"(https[^"]+)"/\1/')"
  [[ -n "$urls" ]] || { echo ""; return 1; }

  # приоритет: universal → arm64 → первый попавшийся
  local pick
  pick="$(printf '%s\n' "$urls" | grep -i 'universal' | head -n1)"
  [[ -n "$pick" ]] || pick="$(printf '%s\n' "$urls" | grep -iE 'arm64|aarch64' | head -n1)"
  [[ -n "$pick" ]] || pick="$(printf '%s\n' "$urls" | head -n1)"
  printf '%s\n' "$pick"
}

# ---------------------------------------------------------------- установка --

install_app() {
  local title="$1" repo="$2" pkg="$3" target="$4"

  info "$title: определяю последний релиз ($repo)…"
  local url
  url="$(latest_apk_url "$repo" || true)"
  [[ -n "$url" ]] || { warn "$title: не удалось получить ссылку на APK (GitHub недоступен?). Пропускаю."; return 1; }

  local file="$WORKDIR/$(basename "$url")"
  dim "$(basename "$url")"
  if [[ -s "$file" ]]; then
    dim "уже скачан, использую локальную копию"
  else
    curl -fL --progress-bar "$url" -o "$file" || { warn "$title: скачивание не удалось."; return 1; }
  fi

  info "$title: устанавливаю на телевизор…"
  if "$ADB" -s "$target" install -r -g "$file"; then
    :
  elif "$ADB" -s "$target" install -r "$file"; then
    :
  else
    warn "$title: установка не удалась. Если ошибка INSTALL_FAILED_UPDATE_INCOMPATIBLE —
      удали старую версию с ТВ и повтори:  $ADB -s $target uninstall $pkg"
    return 1
  fi

  if "$ADB" -s "$target" shell pm list packages 2>/dev/null | tr -d '\r' | grep -qx "package:$pkg"; then
    info "$title: установлен ✔  ($pkg)"
  else
    warn "$title: APK поставился, но пакет $pkg в списке не найден — проверь имя приложения на ТВ вручную."
  fi
  return 0
}

# ------------------------------------------------------------------- main ----

ensure_adb
"$ADB" start-server >/dev/null 2>&1 || true

if [[ -z "$TV_IP" ]]; then
  TV_IP="$(discover_tv)"
fi
[[ "$TV_IP" == *:* ]] || TV_IP="${TV_IP}:${ADB_PORT}"
info "Телевизор: $TV_IP"

connect_tv "$TV_IP"

model="$("$ADB" -s "$TV_IP" shell getprop ro.product.model 2>/dev/null | tr -d '\r' || true)"
andv="$("$ADB" -s "$TV_IP" shell getprop ro.build.version.release 2>/dev/null | tr -d '\r' || true)"
[[ -n "$model" ]] && dim "модель: $model, Android $andv"

rc=0
install_app "LAMPA"     "$LAMPA_REPO"     "$LAMPA_PKG"     "$TV_IP" || rc=1
install_app "TorrServe" "$TORRSERVE_REPO" "$TORRSERVE_PKG" "$TV_IP" || rc=1

echo
if [[ $rc -eq 0 ]]; then
  info "Готово. Оба приложения на телевизоре — ищи их в списке приложений."
  dim "Дальше: запусти TorrServe (он поднимет сервер на 127.0.0.1:8090), затем в LAMPA →"
  dim "Настройки → Торренты → TorrServe, адрес http://127.0.0.1:8090"
else
  warn "Завершено с ошибками — смотри сообщения выше."
fi

"$ADB" disconnect "$TV_IP" >/dev/null 2>&1 || true
exit $rc
