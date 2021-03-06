# Abstract Video Compression

It is difficult to send video over a very low bandwidth stream. For example, on a drone
there is limited power and antenna size, forcing the bandwidth to be minimal. The proposed
solution is a type of extreme video compression that would send the basic shapes and colors of
each frame and eliminate unnecessary detail. There are a number of both lossless and lossy
video compression techniques out there that do a great job at what they were designed to do,
but few are able to compress down to an incredibly minimal bandwidth while keeping some sort
of recognizable and useful video stream. This project is focused on streams with 3 kbps
bandwidth and under. Our solution would provide a clean, recognizable video full of basic
shapes meeting this minimal bandwidth requirement by only sending the necessary few bytes
per shape. Effectively, we’re attempting to find an abstract representation of each frame that
looks “close enough” to the original video that its basic features are recognizable to a human
observer.

To try this compression out, use the following command:

python test2.py <nameOfInputImage>