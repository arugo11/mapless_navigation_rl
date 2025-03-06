from setuptools import setup
import os
from glob import glob

package_name = 'mapless_navigation_rl'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yudai Shimanaka',
    maintainer_email='al23088@shibaura-it.ac.jp',
    description='TurtleBot3での強化学習を用いたマップレスナビゲーション',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_navigator = mapless_navigation_rl.rl_navigator:main',
            'train_agent = mapless_navigation_rl.train_agent:main',
            'evaluate_agent = mapless_navigation_rl.evaluate_agent:main',
        ],
    },
)