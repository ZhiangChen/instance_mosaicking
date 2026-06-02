(1) Git clone the repository
```
git clone https://github.com/ZhiangChen/instance_segmentation_remote_sensing.git
```

(2) Build a docker image
```
docker build -t instance_mosaicking .
```

(3) Run the docker image in a container
```
docker run -p 8888:8888 -it --name instance_mosaicking -v $(pwd)/../:/root/instance_mosaicking/ instance_mosaicking
```
`$(pwd)/../` should be your repository directory. 

(4) Run jupyter notebook in the container
```
cd
jupyter notebook --allow-root --ip=0.0.0.0
```

(5) Access to a new terminal in the container
```
docker exec -it instance_mosaicking bash
```

(6) Start the exited but stopped container
```
docker start instance_mosaicking
```

