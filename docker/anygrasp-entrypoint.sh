#!/usr/bin/env bash
# =============================================================================
# AnyGrasp sidecar entrypoint.
#
# The feature id AnyGrasp licenses against is sha256 over the SORTED SET of
# MAC addresses `ifconfig` reports (see docker/Dockerfile.anygrasp header).
# Everything here exists to make that set exactly one, always the same one,
# and to fail loudly and legibly when it isn't.
#
# Commands:
#   serve                 validate the license, then exec the ZMQ server (default)
#   feature-id            print this container's feature id and exit
#   check                 validate the mounted license and exit
#   scan MAC [MAC ...]    rewrite eth0's MAC to each candidate, print the
#                         resulting feature id, and flag the one that matches
#                         licenseCfg.json. Use this when you have a working
#                         license but don't know which MAC produced it.
#   shell                 drop into bash
# =============================================================================
set -euo pipefail

LICENSE_DIR="${ANYGRASP_LICENSE_DIR:-/opt/anygrasp/grasp_detection/license}"
LICENSE_CFG="${LICENSE_DIR}/licenseCfg.json"
IFACE="${ANYGRASP_IFACE:-eth0}"

log() { printf '[anygrasp] %s\n' "$*" >&2; }

# --- feature id, straight from the SDK (never reimplemented here) ------------
feature_id() {
    python - <<'PY'
from gsnet import get_feature_id
print(get_feature_id())
PY
}

licensed_id() {
    [ -f "${LICENSE_CFG}" ] || return 1
    python - "${LICENSE_CFG}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("feature_id", ""))
PY
}

macs_seen() {
    ifconfig 2>/dev/null | grep -oE '(ether|HWaddr)[[:space:]]+[0-9A-Fa-f:.-]{12,17}' \
        | awk '{print $2}' | tr 'a-f' 'A-F' | sort -u
}

set_mac() {
    ip link set dev "${IFACE}" down
    ip link set dev "${IFACE}" address "$1"
    ip link set dev "${IFACE}" up
}

# --- diagnosis ---------------------------------------------------------------
# Everything the mismatch case needs, in one block, so a restart loop in
# `docker compose logs grasp` is self-explanatory.
report() {
    log "interface        : ${IFACE} ($(cat "/sys/class/net/${IFACE}/address" 2>/dev/null || echo '?'))"
    log "MACs ifconfig sees: $(macs_seen | paste -sd, - )"
    log "local feature id : $(feature_id || echo '<failed>')"
    if have="$(licensed_id)"; then
        log "license feature id: ${have}"
    else
        log "license feature id: <no ${LICENSE_CFG}>"
    fi
}

fail_license() {
    report
    cat >&2 <<EOF

[anygrasp] License validation failed. In order of likelihood:

  1. eth0's MAC is not the one the license was issued for.
     Pin it: put  ANYGRASP_MAC=<the licensed MAC>  in .env, then
       docker compose up -d --force-recreate grasp
     If more than one MAC shows above, this container is on host networking
     -- it must not be; check the compose stanza.

  2. You don't know which MAC the license was issued for (e.g. it was
     registered while running on host networking, where the id hashed the
     whole host interface set). Recover it:
       docker compose run --rm grasp scan \\
           \$(cat /sys/class/net/*/address | tr '\\n' ' ')
     If nothing matches, the id was hashed from a multi-MAC set and cannot
     be reproduced from a single pinned interface -- re-register (below).

  3. The license predates the 2026-07-04 SDK, which changed feature-id
     generation. A license issued against an old feature id will never
     validate against the pinned commit in Dockerfile.anygrasp.

  4. license/ isn't mounted. Expected at ${LICENSE_DIR}
     (host: ./anygrasp_runtime/license/), containing licenseCfg.json,
     *.lic, *.public_key, *.signature.

  To re-register: run \`docker compose run --rm grasp feature-id\` with
  ANYGRASP_MAC already pinned to this machine's permanent NIC MAC, and
  submit that id at https://forms.gle/XVV3Eip8njTYJEBo6 . Because the MAC
  is pinned, the id will not drift again.
EOF
    exit 1
}

check() {
    python - "${LICENSE_DIR}" <<'PY'
import sys
path = sys.argv[1]
try:
    from gsnet import validate_license
    res = validate_license(path)
except ImportError:
    from gsnet import check_license
    res = {"ok": bool(check_license(path)), "message": ""}
msg = res.get("message") or ""
print(f"[anygrasp] license check: ok={res.get('ok')} {msg}", file=sys.stderr)
sys.exit(0 if res.get("ok") else 1)
PY
}

cmd="${1:-serve}"
[ $# -gt 0 ] && shift || true

case "${cmd}" in
    feature-id)
        report
        feature_id
        ;;

    check)
        report
        check || fail_license
        ;;

    scan)
        [ $# -ge 1 ] || { log "usage: scan MAC [MAC ...]"; exit 2; }
        want="$(licensed_id || true)"
        orig="$(cat "/sys/class/net/${IFACE}/address")"
        log "licensed feature id: ${want:-<none>}"
        hit=""
        for mac in "$@"; do
            # skip loopback / all-zero / broadcast, which the SDK drops anyway
            case "$(printf '%s' "${mac}" | tr -d ':-' | tr 'a-f' 'A-F')" in
                000000000000|FFFFFFFFFFFF) continue ;;
            esac
            set_mac "${mac}" || { log "  ${mac} -> could not set (need cap_add: NET_ADMIN)"; continue; }
            fid="$(feature_id || echo '<failed>')"
            if [ -n "${want}" ] && [ "${fid}" = "${want}" ]; then
                log "  ${mac} -> ${fid}   <== MATCH"
                hit="${mac}"
            else
                log "  ${mac} -> ${fid}"
            fi
        done
        set_mac "${orig}" || true
        if [ -n "${hit}" ]; then
            log ""
            log "put this in .env:  ANYGRASP_MAC=${hit}"
        else
            log ""
            log "no single MAC reproduces the licensed id -- see 'check' output for options"
        fi
        ;;

    shell)
        exec bash "$@"
        ;;

    serve)
        report
        check || fail_license
        log "starting grasp server on :${GRASP_PORT:-5666}"
        exec python /opt/grasp_server/server.py "$@"
        ;;

    *)
        exec "${cmd}" "$@"
        ;;
esac
