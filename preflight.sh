#!/usr/bin/env bash
# Preflight for the AnyGrasp sidecar. Run from the ur5e-manip-hardware repo
# root. Read-only: checks things, changes nothing.
IFACE="${1:-eno1}"
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; RC=1; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; }
RC=0

echo "== 1. repo root and patch applied =="
[ -f docker-compose.yml ] && pass "in a compose project: $PWD" || fail "no docker-compose.yml here -- cd to the repo root"
for f in docker/Dockerfile.anygrasp docker/anygrasp-entrypoint.sh grasp_server/server.py; do
    [ -f "$f" ] && pass "$f" || fail "$f missing -- patch not applied"
done
[ -x docker/anygrasp-entrypoint.sh ] && pass "entrypoint is executable" \
    || warn "entrypoint not +x (COPY preserves the bit; chmod in the Dockerfile covers it anyway)"

echo
echo "== 2. .env location and contents =="
# Compose reads .env from the project directory, i.e. next to docker-compose.yml
if [ -f .env ]; then
    pass ".env exists at $PWD/.env"
else
    fail "no .env in the project dir -- compose will use the placeholder MAC"
fi
n=$(grep -c '^[[:space:]]*ANYGRASP_MAC=' .env 2>/dev/null || echo 0)
case "$n" in
    0) fail "no ANYGRASP_MAC line in .env" ;;
    1) pass "exactly one ANYGRASP_MAC line" ;;
    *) warn "$n ANYGRASP_MAC lines -- last one wins, but delete the empty one from .env.example" ;;
esac
grep -n '^[[:space:]]*ANYGRASP_MAC=' .env 2>/dev/null | sed 's/^/        /'
if grep -q '^ANYGRASP_MAC=[0-9a-fA-F][0-9a-fA-F]\(:[0-9a-fA-F][0-9a-fA-F]\)\{5\}[[:space:]]*$' .env 2>/dev/null; then
    pass "value is a bare colon-separated MAC (no quotes, no trailing junk)"
else
    warn "last line above should be exactly ANYGRASP_MAC=xx:xx:xx:xx:xx:xx -- no quotes, no spaces"
fi
file .env 2>/dev/null | grep -q CRLF && fail ".env has CRLF line endings -- run: sed -i 's/\r$//' .env" \
    || pass "no CRLF line endings"

echo
echo "== 3. the MAC compose will actually use =="
CFG_MAC=$(docker compose config 2>/dev/null | grep -m1 -A1 'grasp_net:' | grep mac_address | awk '{print $2}')
HOST_MAC=$(cat "/sys/class/net/$IFACE/address" 2>/dev/null)
echo "        compose : ${CFG_MAC:-<none>}"
echo "        $IFACE : ${HOST_MAC:-<no such interface>}"
if [ -n "$CFG_MAC" ] && [ "$CFG_MAC" = "$HOST_MAC" ]; then
    pass "compose is pinning this machine's $IFACE"
elif [ "$CFG_MAC" = "02:00:00:00:00:00" ]; then
    fail "placeholder MAC -- .env not being read"
else
    fail "compose MAC does not match $IFACE"
fi
docker compose config 2>/dev/null | grep -q 'network_mode: host' \
    && warn "some service uses host networking (expected: all but grasp)" || true
docker compose config 2>/dev/null | sed -n '/^  grasp:/,/^  [a-z]/p' | grep -q 'network_mode' \
    && fail "grasp still has network_mode -- the whole fix is that it must not" \
    || pass "grasp is not host-networked"

echo
echo "== 4. will this MAC survive reboots? =="
AAT=$(cat "/sys/class/net/$IFACE/addr_assign_type" 2>/dev/null)
[ "$AAT" = "0" ] && pass "addr_assign_type=0 (permanent, burned into the NIC)" \
    || fail "addr_assign_type=$AAT (1=random, 2=stolen, 3=set) -- this MAC can change"
PERM=$(ethtool -P "$IFACE" 2>/dev/null | awk '{print $NF}')
if [ -n "$PERM" ]; then
    [ "$PERM" = "$HOST_MAC" ] && pass "ethtool permanent address == current address" \
        || fail "permanent=$PERM but current=$HOST_MAC -- something is spoofing it"
else
    warn "ethtool not installed (sudo apt install ethtool) -- skipped permanent-address check"
fi
[ -e "/sys/class/net/$IFACE/wireless" ] || [ -e "/sys/class/net/$IFACE/phy80211" ] \
    && fail "$IFACE is wireless -- NetworkManager randomizes wifi MACs by default; pin a wired NIC" \
    || pass "$IFACE is wired"
if command -v nmcli >/dev/null 2>&1; then
    CLONED=$(nmcli -t -f 802-3-ethernet.cloned-mac-address device show "$IFACE" 2>/dev/null | cut -d: -f2-)
    RAND=$(grep -rl 'mac-address-randomization\|cloned-mac-address' /etc/NetworkManager/conf.d/ 2>/dev/null)
    [ -n "$RAND" ] && warn "NetworkManager MAC config found in: $RAND (check it isn't randomizing $IFACE)" \
        || pass "no NetworkManager MAC randomization config"
fi
DEV=$(readlink -f "/sys/class/net/$IFACE" 2>/dev/null)
case "$DEV" in
    *usb*) warn "$IFACE is a USB NIC -- unplugging the dock takes the license with it" ;;
    *) pass "$IFACE is an onboard/PCIe NIC" ;;
esac

echo
echo "== 5. build prerequisites =="
CV=$(docker compose version --short 2>/dev/null)
echo "        docker compose $CV"
printf '%s\n2.24.0\n' "$CV" | sort -V | head -1 | grep -q '^2\.2[4-9]\|^2\.[3-9]\|^[3-9]' \
    && pass "compose >= 2.24 (networks.*.mac_address supported)" \
    || warn "compose $CV may predate networks.*.mac_address -- if 'compose' above showed <none>, move mac_address to the service level"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
echo "        compute cap $CC"
[ "$CC" = "8.6" ] && pass "matches TORCH_CUDA_ARCH_LIST=8.6" \
    || warn "TORCH_CUDA_ARCH_LIST=8.6 in Dockerfile.anygrasp does not match $CC -- fix before building"
for d in anygrasp_runtime/license anygrasp_runtime/checkpoints; do
    if [ -d "$d" ]; then
        OWNER=$(stat -c '%U' "$d")
        [ "$OWNER" = "$(id -un)" ] && pass "$d exists, owned by $OWNER" \
            || warn "$d owned by $OWNER -- docker auto-created it; sudo chown -R $(id -un): anygrasp_runtime"
    else
        warn "$d missing -- mkdir it before 'up' or docker creates it root-owned"
    fi
done

echo
[ "$RC" = "0" ] && echo "preflight: no blocking failures" || echo "preflight: fix the FAILs above first"
exit $RC
