#!/bin/bash
# chkconfig: 4 99 20
# description: date logging
# processname: date_logging
TARGET_PATH=/home/ec2-user/machine_learning
LOG_PATH=/home/ec2-user/log
su - ec2-user
pushd $TARGET_PATH
    git fetch origin develop 2>&1 >> $LOG_PATH
    git checkout develop 2>&1 >> $LOG_PATH
    git pull origin develop 2>&1 >> $LOG_PATH
    cat version > result2
    if [[ ! -f $TARGET_PATH/blocking_file ]]; then
	echo "blocking file not found" >> result2
    else
	echo "blocking file found" >> result2
    fi
popd
