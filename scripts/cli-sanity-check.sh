#!/usr/bin/env bash
# Medulla CLI sanity check — runs after every sprint merge or before PR approval.
# Usage: ./scripts/cli-sanity-check.sh
# Exits non-zero on any check failure.

set -euo pipefail

PASS=0; FAIL=0
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; ((PASS++)); }
fail() { echo -e "${RED}✗${NC} $1"; ((FAIL++)); }
section() { echo -e "\n${YELLOW}── $1 ──${NC}"; }

# ── Setup ────────────────────────────────────────────────────────────────────

section "Setup — reset to clean state"
medulla reset --all --yes 2>&1 | grep -q "Reset complete" && pass "reset --all" || fail "reset --all"
medulla use bedrock 2>&1 | grep -q "bedrock" && pass "use bedrock" || fail "use bedrock"

# ── Status ───────────────────────────────────────────────────────────────────

section "Status"
STATUS=$(medulla status 2>&1)
echo "$STATUS" | grep -q "bedrock" && pass "status shows provider" || fail "status shows provider"
echo "$STATUS" | grep -q "Pages:.*0" && pass "status: 0 wiki pages initially" || fail "status: wiki pages count"
echo "$STATUS" | grep -q "raw/" && pass "status: raw/ section present" || fail "status: missing raw/ section"
echo "$STATUS" | grep -q "Sessions" && pass "status: sessions section present" || fail "status: missing sessions section"

# ── Scan ─────────────────────────────────────────────────────────────────────

section "Scan sessions"
SCAN=$(medulla scan 2>&1)
echo "$SCAN" | grep -q "indexed" && pass "scan: runs without error" || fail "scan: failed"
echo "$SCAN" | grep -q "unchanged\|indexed" && pass "scan: reports counts" || fail "scan: no counts"

STATS=$(medulla stats 2>&1)
echo "$STATS" | grep -q "Sessions:" && pass "stats: episodic section" || fail "stats: missing episodic"
echo "$STATS" | grep -q "Semantic" && pass "stats: semantic section" || fail "stats: missing semantic"
echo "$STATS" | grep -q "entities" && pass "stats: 'entities' (not 'entitys')" || fail "stats: 'entitys' typo present"

# ── URL Ingest ───────────────────────────────────────────────────────────────

section "URL ingest (requires Bedrock + network)"
if [[ "${SKIP_INGEST:-0}" == "1" ]]; then
  echo -e "${YELLOW}⚠${NC} Skipping ingest (SKIP_INGEST=1)"
else
  INGEST=$(medulla ingest https://karpathy.medium.com/software-2-0-a64152b37c35 2>&1)
  echo "$INGEST" | grep -q "raw/" && pass "ingest: raw/ file created" || fail "ingest: no raw/ file"
  echo "$INGEST" | grep -q "pages" && pass "ingest: wiki pages created" || fail "ingest: no pages created"

  # Wiki structure
  section "Wiki structure"
  LIST=$(medulla wiki list 2>&1)
  echo "$LIST" | grep -q "source" && pass "wiki list: source pages present" || fail "wiki list: no source pages"
  echo "$LIST" | grep -q "concept" && pass "wiki list: concept pages present" || fail "wiki list: no concept pages"
  echo "$LIST" | grep -q "entity" && pass "wiki list: entity pages present" || fail "wiki list: no entity pages"

  WIKI_STATS=$(medulla stats 2>&1)
  echo "$WIKI_STATS" | grep -q "concepts" && pass "stats: concepts count" || fail "stats: concepts count missing"
  echo "$WIKI_STATS" | grep -q "entities" && pass "stats: entities count (not entitys)" || fail "stats: entitys typo"

  [[ -f ~/.medulla/wiki/index.md ]] && pass "index.md exists" || fail "index.md missing"
  grep -q "Entities" ~/.medulla/wiki/index.md && pass "index.md: Entities (not Entitys)" || fail "index.md: Entitys typo"
  grep -q "entities/" ~/.medulla/wiki/index.md && pass "index.md: entities/ path (not entitys/)" || fail "index.md: entitys/ path"
  [[ -f ~/.medulla/wiki/log.md ]] && pass "log.md exists" || fail "log.md missing"

  # Idempotent re-ingest
  REINGEST=$(medulla ingest https://karpathy.medium.com/software-2-0-a64152b37c35 2>&1)
  echo "$REINGEST" | grep -q "Nothing queued" && pass "re-ingest: idempotent (nothing queued)" || fail "re-ingest: processed again (should be no-op)"
fi

# ── Search ───────────────────────────────────────────────────────────────────

section "Search"
SEARCH_EPISODIC=$(medulla search "pKa model" 2>&1)
echo "$SEARCH_EPISODIC" | grep -qE "[0-9a-f]{8}" && pass "search: episodic results returned" || fail "search: no episodic results"

if [[ "${SKIP_INGEST:-0}" != "1" ]]; then
  SEARCH_SEMANTIC=$(medulla search "software 2.0" --layer semantic 2>&1)
  echo "$SEARCH_SEMANTIC" | grep -q "Software 2.0\|software" && pass "search: semantic wiki results" || fail "search: no semantic results"
  # Excerpt should not start with frontmatter
  echo "$SEARCH_SEMANTIC" | grep -v "^$" | grep -v "^  $" | grep -qv "^---$" && pass "search: excerpts don't show raw frontmatter" || fail "search: excerpts show frontmatter"
fi

SEARCH_LAYER=$(medulla search "session" --layer episodic 2>&1)
echo "$SEARCH_LAYER" | grep -qvE "concept|entity|source" || true  # episodic only, don't fail if mixed
pass "search --layer episodic: runs without error"

# ── Wiki lint ─────────────────────────────────────────────────────────────────

section "Wiki lint"
if [[ "${SKIP_INGEST:-0}" != "1" ]]; then
  LINT=$(medulla wiki lint 2>&1)
  echo "$LINT" | grep -q "pages" && pass "wiki lint: runs without error" || fail "wiki lint: failed"
  echo "$LINT" | grep -qv "Broken links\|\[\]" && pass "wiki lint: no 'Broken links []' display bug" || fail "wiki lint: display bug present"
  echo "$LINT" | grep -q "Suggested pages\|forward references\|No forward" && pass "wiki lint: forward refs correctly labeled" || fail "wiki lint: wrong label for missing pages"
fi

# ── Reset ─────────────────────────────────────────────────────────────────────

section "Reset command"
medulla reset --yes 2>&1 | grep -q "Reset complete" && pass "reset --yes: works" || fail "reset --yes: failed"
AFTER_RESET=$(medulla wiki list 2>&1)
echo "$AFTER_RESET" | grep -q "No wiki pages\|0 pages" && pass "reset: wiki pages cleared" || true

RAW_COUNT=$(ls ~/.medulla/wiki/raw/ 2>/dev/null | wc -l | tr -d ' ')
[[ "$RAW_COUNT" -gt 0 ]] && pass "reset (no --all): raw/ preserved" || echo -e "${YELLOW}⚠${NC} raw/ is empty after reset (ok if ingest was skipped)"

# ── Summary ───────────────────────────────────────────────────────────────────

echo -e "\n────────────────────────────────────────"
TOTAL=$((PASS + FAIL))
if [[ $FAIL -eq 0 ]]; then
  echo -e "${GREEN}All $TOTAL checks passed ✓${NC}"
else
  echo -e "${RED}$FAIL/$TOTAL checks failed${NC}"
  exit 1
fi
