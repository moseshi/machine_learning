#!/bin/bash
# chkconfig: 4 99 20
# description: date logging
# processname: date_logging
pushd /home/ec2-user/machine_learning
    git fetch origin develop
    git checkout develop
    git pull origin develop
    pipenv install
    pipenv run python main.py
popd
shutdown
