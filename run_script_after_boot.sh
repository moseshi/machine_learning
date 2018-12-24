#!/bin/bash
# chkconfig: 4 99 20
# description: date logging
# processname: date_logging
TARGET_PATH=/home/ec2-user/machine_learning
pushd $TARGET_PATH
    git fetch origin develop
    git checkout develop
    git pull origin develop
    pipenv install
    pipenv run python main.py
popd
if [[ ! -f $TARGET_PATH/blocking_file ]]; then
    shutdown
fi
