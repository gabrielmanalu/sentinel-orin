FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y python3-pip \
    && pip3 install --no-cache-dir paho-mqtt==2.1.0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/bridge
COPY services/messaging/ros2_bridge.py ./

CMD ["python3", "ros2_bridge.py"]
