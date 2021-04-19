# MCV-M5-Team7
Repository for the project for module 5: Visual Recognition of group number 7 (name yet to be defined).

Week 3 presentation: [google slides](https://docs.google.com/presentation/d/1--7MbWl71W3U5sLoO-SUNBETfq-5ur9yv5PA_siLZw8/edit?usp=sharing)

## Team members
- Juan Chaves (<juanvictor.chaves@e-campus.uab.cat>)
- Carmen García (<carmen.garciano@e-campus.uab.cat>)

## Overlaf project & GSlides presentation
Our Overleaf report can be accessed [here](https://www.overleaf.com/read/jkhtwcpnvqkq). Our final [presentation](https://docs.google.com/presentation/d/1zQIDSzE2GYsMPK8bgN-he19n28jDWdL08HvVd0_7o6c/edit?usp=sharing).

# Help

## Docker

A simple Dockerfile with a working version of tensorflow+keras and torch (including pytorch) is available. I'm still quite noob with, so there surely is a better way of doing this

## Installation

- First,  build the docker image with the following command:

```
cd <Dockerfile location>
docker build -t <tag> . 
```

> Make sure you have a requirements.txt on the same directory as the Dockerfile

- Now you can run the image with

```
docker run --gpus all -it -v <Data dir in your pc>/:/data/ -v <Code dir in your pc>:/code/ <image-tag>
```

For example:

```
docker build -t gpu-keras-torch .
docker run --gpus all -it -v ~/code/mcv/Databases/:/data/ -v ~/code/mcv/MCV-M5-Team7/week1:/code/ gpu-keras-torch 
```
