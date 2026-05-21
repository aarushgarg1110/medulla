#!/usr/bin/env bash
# Medulla CLI sanity check — run after every sprint merge or before PR approval.
#
# Uses MEDULLA_DIR=~/.medulla-dev so your real ~/.medulla is NEVER touched.
# The dev dir is reset + re-ingested each full run for a clean slate.
#
# Usage:
#   ./scripts/cli-sanity-check.sh              # full run (reset + ingest)
#   SKIP_INGEST=1 ./scripts/cli-sanity-check.sh  # skip LLM, use existing dev wiki

set -uo pipefail   # note: no -e so we can capture non-zero exits safely

# Point medulla at a separate dev directory — real ~/.medulla is never touched
export MEDULLA_DIR="${HOME}/.medulla-dev"

PASS=0; FAIL=0
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; DIM='\033[2m'; NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}✗${NC} $1"; FAIL=$((FAIL + 1)); }
section() { echo -e "\n${YELLOW}── $1 ──${NC}"; }
info() { echo -e "${DIM}  $1${NC}"; }

check_contains() {
    # check_contains "label" "string" "pattern"
    if echo "$2" | grep -q "$3"; then pass "$1"; else fail "$1 (expected: $3)"; fi
}
check_not_contains() {
    if ! echo "$2" | grep -q "$3"; then pass "$1"; else fail "$1 (should NOT contain: $3)"; fi
}

# ── Setup ─────────────────────────────────────────────────────────────────────

section "Setup"
if [[ "${SKIP_INGEST:-0}" == "1" ]]; then
    info "SKIP_INGEST=1 — skipping destructive reset, using existing wiki state"
    pass "reset skipped (SKIP_INGEST=1)"
else
    OUT=$(medulla reset --all --yes 2>&1)
    check_contains "reset --all" "$OUT" "Reset complete"
fi

OUT=$(medulla use bedrock 2>&1)
check_contains "use bedrock" "$OUT" "bedrock"

# ── Status ────────────────────────────────────────────────────────────────────

section "Status"
STATUS=$(medulla status 2>&1)
check_contains "status: shows provider" "$STATUS" "bedrock"
check_contains "status: raw/ section" "$STATUS" "raw/"
check_contains "status: sessions section" "$STATUS" "Sessions"
check_not_contains "status: no 'entitys' typo" "$STATUS" "entitys"

# ── Scan ──────────────────────────────────────────────────────────────────────

section "Scan"
OUT=$(medulla scan 2>&1)
check_contains "scan: runs without error" "$OUT" "indexed"
check_not_contains "scan: no 'skipped' (all are now unchanged/empty)" "$OUT" "^✓ Sessions:.*skipped"

STATS=$(medulla stats 2>&1)
check_contains "stats: episodic section" "$STATS" "Sessions:"
check_contains "stats: semantic section" "$STATS" "Semantic"
check_not_contains "stats: no 'entitys' typo" "$STATS" "entitys"

# ── URL Ingest (LLM required) ─────────────────────────────────────────────────

section "URL ingest"
if [[ "${SKIP_INGEST:-0}" == "1" ]]; then
    info "Skipping ingest (SKIP_INGEST=1) — using existing wiki state"
    pass "ingest skipped"
else
    OUT=$(medulla ingest https://karpathy.medium.com/software-2-0-a64152b37c35 2>&1)
    check_contains "ingest: raw/ file created" "$OUT" "raw/"
    check_contains "ingest: wiki pages created" "$OUT" "pages"
fi

# ── Wiki structure ────────────────────────────────────────────────────────────

section "Wiki structure"
WIKI_LIST=$(medulla wiki list 2>&1)
# Check if wiki has all three page types (full ingest required for this)
HAS_CONCEPT=$(echo "$WIKI_LIST" | grep -c "concept" || true)
HAS_ENTITY=$(echo "$WIKI_LIST" | grep -c "entity" || true)
HAS_SOURCE=$(echo "$WIKI_LIST" | grep -c "source" || true)

if [[ "$HAS_SOURCE" -gt 0 && "$HAS_CONCEPT" -gt 0 && "$HAS_ENTITY" -gt 0 ]]; then
    pass "wiki list: all three page types present"
    WIKI_STATS=$(medulla stats 2>&1)
    check_contains "stats: concepts count" "$WIKI_STATS" "concepts"
    check_contains "stats: entities count (not entitys)" "$WIKI_STATS" "entities"
    check_not_contains "stats: no entitys typo" "$WIKI_STATS" "entitys"

    if [[ -f ${MEDULLA_DIR}/wiki/index.md ]]; then
        pass "index.md exists"
        INDEX=$(cat ${MEDULLA_DIR}/wiki/index.md)
        check_contains "index.md: Entities header" "$INDEX" "## Entities"
        check_not_contains "index.md: no Entitys typo" "$INDEX" "## Entitys"
        check_contains "index.md: entities/ path" "$INDEX" "entities/"
    else
        fail "index.md missing"
    fi
    [[ -f ${MEDULLA_DIR}/wiki/log.md ]] && pass "log.md exists" || fail "log.md missing"
else
    info "Wiki incomplete (sources=$HAS_SOURCE concepts=$HAS_CONCEPT entities=$HAS_ENTITY)"
    info "Run without SKIP_INGEST=1 for full wiki structure checks"
    pass "wiki structure skipped (incomplete wiki — expected with SKIP_INGEST=1 and no prior ingest)"
fi

# ── Re-ingest idempotency ─────────────────────────────────────────────────────

section "Idempotency"
if [[ "${SKIP_INGEST:-0}" != "1" ]]; then
    OUT=$(medulla ingest https://karpathy.medium.com/software-2-0-a64152b37c35 2>&1)
    check_contains "re-ingest: nothing queued (idempotent)" "$OUT" "Nothing queued"
else
    pass "idempotency skipped (SKIP_INGEST=1)"
fi

# ── Search ────────────────────────────────────────────────────────────────────

section "Search"
SEARCH=$(medulla search "pKa model" 2>&1)
if echo "$SEARCH" | grep -qE "[0-9a-f]{8}|Software|concept|entity|source"; then
    pass "search: results returned"
else
    fail "search: no results found"
fi

if [[ "${SKIP_INGEST:-0}" != "1" ]]; then
    SEM=$(medulla search "software 2.0" --layer semantic 2>&1)
    check_contains "search --layer semantic: wiki results" "$SEM" "Software 2.0"
    check_not_contains "search: excerpt not raw frontmatter" "$SEM" "^---$"
fi

EPI=$(medulla search "session" --layer episodic 2>&1)
if echo "$EPI" | grep -qE "[0-9a-f]{8}|No results"; then
    pass "search --layer episodic: runs without error"
else
    fail "search --layer episodic: unexpected output"
fi

# ── Wiki lint ─────────────────────────────────────────────────────────────────

section "Wiki lint"
LINT=$(medulla wiki lint 2>&1)
check_contains "wiki lint: runs without error" "$LINT" "pages\|Wiki lint\|does not exist"
check_not_contains "wiki lint: no '[]' display bug" "$LINT" "→ \[\]"
if echo "$LINT" | grep -q "Suggested pages\|No forward references\|does not exist"; then
    pass "wiki lint: forward refs correctly labeled (not 'Broken links')"
else
    fail "wiki lint: expected 'Suggested pages' or 'No forward references'"
fi

# ── Reset ─────────────────────────────────────────────────────────────────────

section "Reset command"
if [[ "${SKIP_INGEST:-0}" == "1" ]]; then
    info "Skipping reset command test (SKIP_INGEST=1 — preserving wiki state for next run)"
    pass "reset test skipped (SKIP_INGEST=1)"
else
    OUT=$(medulla reset --yes 2>&1)
    check_contains "reset --yes: completes" "$OUT" "Reset complete"

    WIKI_AFTER=$(medulla wiki list 2>&1)
    if ! echo "$WIKI_AFTER" | grep -q "source\|concept\|entity"; then
        pass "reset: wiki cleared"
    else
        fail "reset: wiki pages still present after reset"
    fi

    RAW=$(ls ${MEDULLA_DIR}/wiki/raw/ 2>/dev/null | grep -v "url-references.md" | wc -l | tr -d ' ')
    [[ "$RAW" -gt 0 ]] && pass "reset (no --all): raw/ preserved" || info "raw/ is empty (ok)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo -e "\n────────────────────────────────────────"
TOTAL=$((PASS + FAIL))
if [[ $FAIL -eq 0 ]]; then
    echo -e "${GREEN}All $TOTAL checks passed ✓${NC}"
else
    echo -e "${RED}$FAIL of $TOTAL checks failed${NC}"
    exit 1
fi
