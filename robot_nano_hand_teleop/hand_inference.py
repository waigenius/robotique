import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import JointState

class HandInferenceNode(Node):
    def __init__(self):
        super().__init__('hand_inference_node')

        #subscriber
    

    def main():
        node = HandInferenceNode()