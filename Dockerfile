FROM ubuntu:latest
LABEL authors="rafal"

ENTRYPOINT ["top", "-b"]