#!/bin/bash

# Script to rewrite commit messages for conventional commits
# Makes first line lowercase and truncates to 30 chars

git filter-branch -f --msg-filter '
read -r msg
# Process only the first line
first_line=$(echo "$msg" | head -n1)
rest=$(echo "$msg" | tail -n+2)
# Lowercase and truncate first line
new_first=$(echo "$first_line" | tr "[:upper:]" "[:lower:]" | cut -c1-30)
# Output new message
echo "$new_first"
if [ -n "$rest" ]; then
    echo "$rest"
fi
' -- --all
