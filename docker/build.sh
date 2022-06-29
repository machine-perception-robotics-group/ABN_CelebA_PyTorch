#!/bin/bash


# add exec permission to entrypoint.sh
chmod a+x entrypoint.sh


# build
docker build --tag=cumprg/mtlabn:1.11.0 --force-rm=true --file=./Dockerfile_1_11_0 .


echo "Build docker; done."
