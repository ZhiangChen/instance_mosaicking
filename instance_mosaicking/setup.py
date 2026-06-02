from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='instance-mosaicking',
    version='0.1.1',
    description='Instance mosaicking for geospatial maps and large images',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ZhiangChen/instance_segmentation_remote_sensing',
    author='Zhiang Chen',
    author_email='zxc251@case.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    packages=["instance_mosaicking"]
    + [
        f"instance_mosaicking.{package}"
        for package in find_packages(exclude=["tests", "tests_*"])
        if not package.startswith("tests")
    ],
    package_dir={"instance_mosaicking": "."},
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.24.2',
        'pandas>=2.0.0',
        'geopandas>=0.12.0',
        'rasterio>=1.3.6',
        'rioxarray>=0.13.4',
        'opencv-python>=4.7.0.72',
        'tqdm>=4.65.0',
        'shapely>=2.0.1',
        'matplotlib>=3.7.1',
        'fiona>=1.9.3',
        'pyproj>=3.5.0',
        'GDAL>=3.3.2',
    ],
)

