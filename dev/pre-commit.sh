#!/usr/bin/env bash
# Pre-commit tripwire: block known secret patterns from ever being committed.
# Installed as .git/hooks/pre-commit (symlink). Bypass ONLY with a conscious
# `git commit --no-verify` — accidents can't pass, intent can.
PAT='apify_api_|gsk_|tvly-|nvapi-|ghp_|github_pat_|sk-ant-|ctx7sk-|fc-[A-Za-z0-9]|sonatype_pat_|AKIA[0-9A-Z]{16}|xox[bap]-|BEGIN [A-Z ]*PRIVATE KEY'
matches=$(git diff --cached --unified=0 | grep -inoE "$PAT" | head -5)
if [ -n "$matches" ]; then
    printf '%s\n' "$matches" >&2
    echo "pre-commit BLOCKED: secret pattern above. Remove it, or 'git commit --no-verify' if you truly mean it." >&2
    exit 1
fi
