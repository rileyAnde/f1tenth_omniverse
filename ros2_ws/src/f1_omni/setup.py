from setuptools import find_packages, setup

package_name = 'f1_omni'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='r478a194',
    maintainer_email='r478a194@ku.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crop = f1_omni.crop_scan:main',
            'tln = f1_omni.TLN:main'
        ],
    },
)
