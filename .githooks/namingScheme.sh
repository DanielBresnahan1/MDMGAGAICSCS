#!/usr/bin/env bash
LC_ALL=C

local_branch="$(git rev-parse --abbrev-ref HEAD)"

valid_branch_regex="^(feature|bugfix|release|hotfix)+/[a-zA-Z0-9]+-+[a-z0-9._-]"

message="Im sowwy, thewe iws something wwong with youw bwanch nawme. Bwanch names in thiws pwoject must adhewe tuwu thiws contwact: $valid_branch_regex youw commit wiww be wejected. Uwu shouwd wename youw bwanch tuwu a vawid nawme awnd twy again."

if [[ ! $local_branch =~ $valid_branch_regex ]]
then
    echo "$message"
    exit 1
fi

exit 0