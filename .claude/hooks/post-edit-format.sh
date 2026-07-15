#!/bin/bash
# PostToolUse hook: 写入 .py 文件后自动用 black 格式化

FILE_PATH=$(echo "$CLAUDE_TOOL_INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('file_path', d.get('new_file_path', '')))" 2>/dev/null)

if [[ "$FILE_PATH" == *.py ]]; then
  /usr/local/bin/black "$FILE_PATH" 2>/dev/null && echo "black: formatted $FILE_PATH" || true
fi
