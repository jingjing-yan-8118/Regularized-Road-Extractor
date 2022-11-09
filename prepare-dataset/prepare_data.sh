#!/bin/bash

# step1 skeleton mask to centerline
python thining.py # --mask_root <mask label root> --skeleton_root <centerline output root>

# step2 change centerline to graph
python mapextract.py # --skeleton_root <centerline root> --graph_root <graph output root>

# step3 resample graph
python resample_road_nodes.py # --graph_root <graph input root> --resample_root <resample output root> --density <resample density>

# step4 calculate width label and cut image
python prepare_width_and_image_of_nodes.py



