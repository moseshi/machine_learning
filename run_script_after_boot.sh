#!/bin/bash
# chkconfig: 4 99 20
# description: date logging
# processname: date_logging
TARGET_PATH=/home/ec2-user/machine_learning
LOG_PATH=/home/ec2-user/log
HOME=/home/ec2-user
pushd $TARGET_PATH
    sudo -u ec2-user git pull origin develop 2>&1 >> $LOG_PATH
    if [[ ! -f $TARGET_PATH/blocking_file ]]; then
	shutdown 0
    fi
popd
