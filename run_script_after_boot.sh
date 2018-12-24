#!/bin/bash
# chkconfig: 4 99 20
# description: date logging
# processname: date_logging
TARGET_PATH=/home/ec2-user/machine_learning
su - ec2-user
pushd $TARGET_PATH
    git fetch origin develop
    git checkout develop
    git pull origin develop
    cat version > result2
popd
