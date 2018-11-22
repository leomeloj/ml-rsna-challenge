#!/bin/bash

#$1 -> input image
#$2 -> output image
#$3 -> radius

iftCloseBasins $1 basins.png 
iftSub $1 basins.png sub.png
iftThreshold -i sub.png -o th.png --output-value 255 --vi 0 --vf 127
iftLinearStretch th.png neg.png 0 255 255 0
iftOpenBin neg.png $2 $3
eog $2
