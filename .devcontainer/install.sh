#!/bin/bash

export DEBIAN_FRONTEND=noninteractive

sudo apt update

# Install project dependencies
uv sync
