from math import sin, cos, pi 
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster, TransformStamped

class StatePublisher(Node):

    def __init__(self):
        rclpy.init()
        super().__init__('state_publisher')

        qos_profile = QoSProfile(depth=10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', qos_profile)

        self.nodeName = self.get_name()
        self.get_logger().info("{0} started".format(self.nodeName))

        degree = pi /180.0
        loop_rate = self.create_rate(30)

        #message declarations
        joint_state = JointState()
        angle = 0.0

        try:
            while rclpy.ok():
                rclpy.spin_once(self)

                # update joint_state
                now = self.get_clock().now()
                joint_state.header.stamp = now.to_msg()

                joint_state.name = [
                    'base_palm_joint',
                    'palm_finger11_joint',
                    'finger112_joint',
                    'finger123_joint',
                    'finger134_joint',

                    'palm_finger21_joint',
                    'finger212_joint',
                    'finger223_joint',
                    'finger234_joint'
                ]

                joint_state.position = [angle, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0

                
                ]

                #send the joint state
                self.joint_pub.publish(joint_state)

                #Create new robot state
                angle = angle + 0.001

                #This will adjust as needed per iteration 
                loop_rate.sleep()

        except KeyboardInterrupt:
            pass




def main():
    node =  StatePublisher()



if __name__ == '__main__':
    main()
