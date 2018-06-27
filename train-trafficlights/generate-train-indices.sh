#!/bin/bash
set -ex
grep '\.[0-7].*\.jpg' all.txt > train.txt
grep -v '\.[0-7].*\.jpg' all.txt > validate.txt
