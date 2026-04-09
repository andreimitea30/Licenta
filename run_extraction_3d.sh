#!/bin/bash
export LD_LIBRARY_PATH=/home/andrei/Licenta/Licenta/venv_linux/lib:${LD_LIBRARY_PATH}
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so.1
source /home/andrei/Licenta/Licenta/venv_linux/bin/activate
cd /home/andrei/Licenta/Licenta
python src/pose_extraction_3d.py
