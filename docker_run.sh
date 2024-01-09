#!/bin/bash
XAUTH=/tmp/.docker.xauth
HASH_FILE=./.created_container.hash
MOUNT_FOLDER="$(pwd)"

if [ ! -f "$HASH_FILE" ]; then
	echo "No Hash file, creating container"
	docker create -it --shm-size=8gb --privileged \
		--env=NVIDIA_DRIVER_CAPABILITIES=all \
		--ipc host \
		--env="DISPLAY=$DISPLAY" \
		--env="QT_X11_NO_MITSHM=1" \
		--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		--env="XAUTHORITY=$XAUTH" \
		--volume="$XAUTH:$XAUTH" \
		--mount type=bind,source=$MOUNT_FOLDER,target=/home/user/work/ \
		botsort-trt:latest >$HASH_FILE
	echo "Starting container"
	docker start $(cat $HASH_FILE)
	echo "Stoping Container"
	docker stop $(cat $HASH_FILE)
fi

container_id=$(cat "$HASH_FILE")
if [ -n "$container_id" ] && [ "$(docker inspect -f '{{.State.Running}}' "$container_id")" = "true" ]; then
	docker exec -it "$container_id" /bin/bash
else
	docker start -i "$container_id"
fi
