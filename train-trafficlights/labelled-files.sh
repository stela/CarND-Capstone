#!/bin/bash
find . -name site_\*.txt | sed -e 's/\.txt/.jpg/'
