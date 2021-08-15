orig='/piq/samples/20030117_1_320_0.ppm'
jpg='/piq/samples/20030117_1_320_0.jpg'
webp='/piq/samples/20030117_1_320_0.webp'
docker run --rm -v `pwd`:/piq -it lomorage:image-quality /piq/image_metrics.py $orig $jpg $webp
