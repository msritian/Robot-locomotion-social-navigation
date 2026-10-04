#!/bin/bash
# Part D plumbing test: zero policy, 10 s, box scene T6 (+4 extra people) and InteriorGS 839962 T8.
bash isaac_eval.sh zero plumb T6 1000 1 4 none 10
bash isaac_eval.sh zero plumb T8 1000 1 0 839962 10
