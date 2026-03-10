from setuptools import setup
import os
from glob import glob

package_name = 'autosweeper_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email',
    description='Coverage sweeping robot using Nav2',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sweeper_node = autosweeper_robot.sweeper_node:main',
        ],
    },
)