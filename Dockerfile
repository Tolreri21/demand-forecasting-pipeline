FROM ubuntu:latest
LABEL authors="TOLK"

ENTRYPOINT ["top", "-b"]