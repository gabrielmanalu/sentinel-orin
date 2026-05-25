FROM ros:humble-ros-base

WORKDIR /opt/rviz
COPY services/messaging/rviz_subscriber/ ./

CMD ["python3", "sentinel_rviz_node.py"]
