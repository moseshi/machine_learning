#!/bin/bash
# ここは適当に置き換えて
pushd /home/ec2-user/machine_learning
    git fetch origin develop
    git checkout develop
    git pull origin develop
    pipenv install
    pipenv run python main.py
popd
shutdown
