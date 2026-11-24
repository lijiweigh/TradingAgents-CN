#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print '缺少 .venv，请先安装项目依赖。'
  exit 1
fi
exec .venv/bin/python scripts/local_services.py start
