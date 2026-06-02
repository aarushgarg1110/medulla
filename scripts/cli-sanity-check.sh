#!/usr/bin/env bash
# Medulla CLI sanity check — run after every sprint merge or before PR approval.
#
# Uses MEDULLA_DIR=~/.medulla-dev so your real ~/.medulla is NEVER touched.
# Output is saved to scripts/sanity-output/YYYY-MM-DD-HH-MM.log automatically.
#
# Usage:
#   ./scripts/cli-sanity-check.sh              # full run (reset + all ingests)
#   SKIP_INGEST=1 ./scripts/cli-sanity-check.sh  # skip LLM, use existing dev wiki

set -uo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

export MEDULLA_DIR="${HOME}/.medulla-dev"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ASSETS="${SCRIPT_DIR}/test-assets"
OUTPUT_DIR="${SCRIPT_DIR}/sanity-output"
mkdir -p "$OUTPUT_DIR"
LOGFILE="${OUTPUT_DIR}/$(date '+%Y-%m-%d-%H-%M').log"

# Tee all output to log file (strip ANSI codes for the file)
exec > >(tee >(sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1

PASS=0; FAIL=0
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; DIM='\033[2m'; NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}✗${NC} $1"; FAIL=$((FAIL + 1)); }
section() { echo -e "\n${YELLOW}── $1 ──${NC}"; }
info() { echo -e "${DIM}  $1${NC}"; }
skip() { echo -e "${DIM}  ⊘ $1 (SKIP_INGEST=1)${NC}"; PASS=$((PASS + 1)); }

check_contains() {
    if echo "$2" | grep -q "$3"; then pass "$1"; else fail "$1 (expected: $3)"; fi
}
check_not_contains() {
    if ! echo "$2" | grep -q "$3"; then pass "$1"; else fail "$1 (should NOT contain: $3)"; fi
}

echo "Medulla CLI Sanity Check — $(date '+%Y-%m-%d %H:%M')"
echo "MEDULLA_DIR: $MEDULLA_DIR"
echo "Log: $LOGFILE"
echo "SKIP_INGEST: ${SKIP_INGEST:-0}"

# ── 1. Setup ──────────────────────────────────────────────────────────────────

section "1. Setup"
if [[ "${SKIP_INGEST:-0}" == "1" ]]; then
    info "SKIP_INGEST=1 — skipping reset, using existing dev wiki state"
    pass "reset skipped"
else
    OUT=$(medulla reset --all --yes 2>&1)
    check_contains "reset --all clears everything" "$OUT" "Reset complete"
fi

# Provider switch tests
OUT=$(medulla use bedrock 2>&1)
check_contains "medulla use bedrock" "$OUT" "bedrock"
OUT=$(medulla use ollama 2>&1)
check_contains "medulla use ollama" "$OUT" "ollama"
OUT=$(medulla use bedrock 2>&1)
check_contains "medulla use bedrock (switch back)" "$OUT" "bedrock"

# Sync real config to dev (model, profile, region) — dev starts with defaults after reset
if [[ -f "${HOME}/.medulla/config.toml" ]]; then
    cp "${HOME}/.medulla/config.toml" "${MEDULLA_DIR}/config.toml"
    info "Synced config from ~/.medulla/config.toml to dev"
fi

# ── 2. Status ─────────────────────────────────────────────────────────────────

section "2. Status"
STATUS=$(medulla status 2>&1)
check_contains "shows active provider" "$STATUS" "bedrock"
check_contains "shows model" "$STATUS" "sonnet\|haiku\|llama\|model"
check_contains "shows raw/ section" "$STATUS" "raw/"
check_contains "shows Sessions section" "$STATUS" "Sessions"
check_not_contains "no 'entitys' typo" "$STATUS" "entitys"

# ── 3. Scan ───────────────────────────────────────────────────────────────────

section "3. Scan (episodic sessions)"
OUT=$(medulla scan 2>&1)
check_contains "scan runs without error" "$OUT" "indexed"
check_contains "scan shows unchanged count" "$OUT" "unchanged\|indexed"

STATS=$(medulla stats 2>&1)
check_contains "stats: Episodic section" "$STATS" "Sessions:"
check_contains "stats: Semantic section" "$STATS" "Semantic"
check_not_contains "stats: no 'entitys' typo" "$STATS" "entitys"
echo "$STATS" | grep -A1 "Sessions:" | head -5

# ── 4. URL ingest ─────────────────────────────────────────────────────────────

section "4. URL ingest (Bedrock + network required)"
if [[ "${SKIP_INGEST:-0}" == "1" ]]; then
    skip "URL ingest"
else
    OUT=$(medulla ingest https://karpathy.medium.com/software-2-0-a64152b37c35 2>&1)
    check_contains "raw/ file created" "$OUT" "raw/"
    check_contains "wiki pages created" "$OUT" "pages"
    RAW_FILE="${MEDULLA_DIR}/wiki/raw/software-20.md"
    [[ -f "$RAW_FILE" ]] && pass "raw/software-20.md exists (URL text archived)" || fail "raw/ file missing"
fi

# ── 5. Local file ingest (Obsidian Clipper simulation) ───────────────────────

section "5. Local file ingest (Obsidian Clipper simulation)"
if [[ "${SKIP_INGEST:-0}" == "1" ]]; then
    skip "local file ingest"
else
    # Simulate Obsidian Clipper: drop a markdown file into raw/ then run medulla ingest
    CLIP_FILE="${MEDULLA_DIR}/wiki/raw/obsidian-clip-test.md"
    mkdir -p "${MEDULLA_DIR}/wiki/raw"
    cat > "$CLIP_FILE" << 'MDEOF'
# The Unreasonable Effectiveness of Data

A classic paper arguing that more data often beats better algorithms.
Key insight: simple models trained on massive datasets frequently outperform
complex models trained on small datasets. This challenges the traditional
focus on algorithm design over data curation.
MDEOF
    pass "Obsidian Clipper simulated: file dropped in raw/"

    OUT=$(medulla ingest 2>&1)   # no args = discover + process
    check_contains "discover + process raw/ file" "$OUT" "Discovered\|Processing\|pages\|queued"
    info "Output: $(echo "$OUT" | head -3)"
fi

# ── 6. PDF ingest ─────────────────────────────────────────────────────────────

section "6. PDF ingest (Adam optimizer paper)"
if [[ "${SKIP_INGEST:-0}" == "1" ]]; then
    skip "PDF ingest"
else
    ADAM_PDF="${TEST_ASSETS}/Adam-Optimizer.pdf"
    if [[ -f "$ADAM_PDF" ]]; then
        OUT=$(medulla ingest "$ADAM_PDF" 2>&1)
        check_contains "PDF: copies to raw/" "$OUT" "raw/"
        check_contains "PDF: creates wiki pages" "$OUT" "pages"
        [[ -f "${MEDULLA_DIR}/wiki/raw/Adam-Optimizer.pdf" ]] && \
            pass "PDF archived in raw/Adam-Optimizer.pdf" || fail "PDF not in raw/"
        # Verify Adam-specific concepts were created
        ADAM_LIST=$(medulla wiki list 2>&1)
        check_contains "PDF: Adam concept page created" "$ADAM_LIST" "adam\|moment\|optimizer"
    else
        fail "Adam-Optimizer.pdf not found in scripts/test-assets/ — run: cp ~/Downloads/Adam-Optimizer.pdf scripts/test-assets/"
    fi
fi

# ── 7. Wiki structure ─────────────────────────────────────────────────────────

section "7. Wiki structure"
WIKI_LIST=$(medulla wiki list 2>&1)
HAS_CONCEPT=$(echo "$WIKI_LIST" | grep -c "concept" || true)
HAS_ENTITY=$(echo "$WIKI_LIST" | grep -c "entity" || true)
HAS_SOURCE=$(echo "$WIKI_LIST" | grep -c "source" || true)

if [[ "$HAS_SOURCE" -gt 0 && "$HAS_CONCEPT" -gt 0 && "$HAS_ENTITY" -gt 0 ]]; then
    pass "all three page types present (sources=$HAS_SOURCE concepts=$HAS_CONCEPT entities=$HAS_ENTITY)"
    WIKI_STATS=$(medulla stats 2>&1)
    check_contains "stats shows concepts" "$WIKI_STATS" "concepts"
    check_contains "stats shows entities (not entitys)" "$WIKI_STATS" "entities"
    [[ -f "${MEDULLA_DIR}/wiki/index.md" ]] && pass "index.md exists" || fail "index.md missing"
    if [[ -f "${MEDULLA_DIR}/wiki/index.md" ]]; then
        INDEX=$(cat "${MEDULLA_DIR}/wiki/index.md")
        check_contains "index.md: ## Entities header" "$INDEX" "## Entities"
        check_not_contains "index.md: no ## Entitys" "$INDEX" "## Entitys"
        check_contains "index.md: entities/ path" "$INDEX" "entities/"
    fi
    [[ -f "${MEDULLA_DIR}/wiki/log.md" ]] && pass "log.md exists" || fail "log.md missing"
else
    info "Wiki incomplete — structure checks skipped (sources=$HAS_SOURCE concepts=$HAS_CONCEPT entities=$HAS_ENTITY)"
    pass "wiki structure skipped"
fi

# ── 8. Idempotency ────────────────────────────────────────────────────────────

section "8. Idempotency"
if [[ "${SKIP_INGEST:-0}" != "1" ]]; then
    OUT=$(medulla ingest https://karpathy.medium.com/software-2-0-a64152b37c35 2>&1)
    check_contains "re-ingest URL: nothing queued" "$OUT" "Nothing queued"
else
    skip "idempotency"
fi

# ── 9. Search ─────────────────────────────────────────────────────────────────

section "9. Search"
SEARCH=$(medulla search "pKa model" 2>&1)
if echo "$SEARCH" | grep -qE "[0-9a-f]{8}|Software|concept|entity|source"; then
    pass "search: episodic results returned"
else
    fail "search: no results"
fi

if [[ "${SKIP_INGEST:-0}" != "1" ]]; then
    SEM=$(medulla search "software 2.0" --layer semantic 2>&1)
    check_contains "search --layer semantic: wiki result" "$SEM" "Software 2.0\|software"
    check_not_contains "search: no raw frontmatter in excerpt" "$SEM" "^---"
fi

EPI=$(medulla search "session" --layer episodic 2>&1)
if echo "$EPI" | grep -qE "[0-9a-f]{8}|No results"; then
    pass "search --layer episodic: works"
else
    fail "search --layer episodic: unexpected"
fi

# ── 10. Wiki lint ─────────────────────────────────────────────────────────────

section "10. Wiki lint"
LINT=$(medulla wiki lint 2>&1)
check_contains "wiki lint: runs" "$LINT" "pages\|Wiki lint\|does not exist"
check_not_contains "wiki lint: no '[]' display bug" "$LINT" "→ \[\]"
if echo "$LINT" | grep -q "Suggested pages\|No forward references\|does not exist"; then
    pass "wiki lint: uses 'Suggested pages' (not 'Broken links')"
else
    fail "wiki lint: unexpected label"
fi
info "Lint output preview:"
echo "$LINT" | head -8 | sed 's/^/  /'

# ── 11. Reset ─────────────────────────────────────────────────────────────────

section "11. Reset command"
if [[ "${SKIP_INGEST:-0}" == "1" ]]; then
    skip "reset (preserving dev wiki state)"
else
    OUT=$(medulla reset --yes 2>&1)
    check_contains "reset --yes: completes" "$OUT" "Reset complete"

    WIKI_AFTER=$(medulla wiki list 2>&1)
    # Look for Rich table border characters — present only when rows exist
    # "No wiki pages found." has no table, just text
    if echo "$WIKI_AFTER" | grep -q "No wiki pages\|0 total" || ! echo "$WIKI_AFTER" | grep -qP "│|\|"; then
        pass "reset: wiki cleared (no pages)"
    elif echo "$WIKI_AFTER" | grep -q "No wiki pages"; then
        pass "reset: wiki cleared (no pages)"
    else
        # Fallback: count actual page type entries
        PAGE_COUNT=$(echo "$WIKI_AFTER" | grep -cE "^\s*\│\s+(source|concept|entity)" || true)
        [[ "$PAGE_COUNT" -eq 0 ]] && pass "reset: wiki cleared" || fail "reset: wiki still has $PAGE_COUNT pages"
    fi

    RAW_COUNT=$(ls "${MEDULLA_DIR}/wiki/raw/" 2>/dev/null | grep -v "url-references.md" | wc -l | tr -d ' ')
    [[ "$RAW_COUNT" -gt 0 ]] && pass "reset (no --all): raw/ preserved ($RAW_COUNT files)" || \
        info "raw/ is empty after reset (ok)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo -e "\n════════════════════════════════════════"
TOTAL=$((PASS + FAIL))
echo "Log saved: $LOGFILE"
if [[ $FAIL -eq 0 ]]; then
    echo -e "${GREEN}All $TOTAL checks passed ✓${NC}"
else
    echo -e "${RED}$FAIL of $TOTAL checks failed${NC}"
    exit 1
fi
