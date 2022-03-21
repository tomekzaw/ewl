#!/usr/bin/env bash
python3 -m flake8 . --ignore E501,E741 --exclude .venv,build
