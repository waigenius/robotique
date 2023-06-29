import os
from glob import glob
from setuptools import setup
from setuptools import find_packages

package_name = 'robot_nano_hand_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name), glob('urdf/*')),
        (os.path.join('share', package_name), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wai',
    maintainer_email='wailekone@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_publisher = robot_nano_hand_teleop.state_publisher:main'
        ],
    },
)
