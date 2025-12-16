from setuptools import setup, find_packages

VERSION = '0.0.4'
DESCRIPTION = 'RetroGeo:  A simple offline Reverse Geocoder Library'
LONG_DESCRIPTION = '''
Our reverse geocoding library is a robust Python tool designed to translate latitude and longitude coordinates into detailed addresses, entirely offline. It processes geographical coordinates, such as latitude and longitude, and returns a full address corresponding to the location, including country, state, city. The library is optimized for performance and accuracy, leveraging pre-downloaded datasets of geographical locations, eliminating the need for online queries or API calls.

This library is especially useful for applications where internet connectivity is limited or unavailable, or where privacy concerns require offline processing. It supports large-scale reverse geocoding operations by efficiently handling massive datasets, making it ideal for geospatial analysis, mobile applications, and data processing pipelines that require rapid location lookup without external dependencies.

With built-in data compression and storage optimizations, our reverse geocoding library minimizes disk space while maintaining high accuracy, offering developers a reliable solution for offline geocoding tasks.'''

# Setting up
setup(
    name="RetroGeo",
    version=VERSION,
    author="SOORAJ TS",
    author_email="sjts007@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'aiocache==0.12.2',
        'shapely==2.1.2',
        'annotated-types==0.7.0',
        'anyio==4.4.0',
        'certifi==2024.8.30',
        'charset-normalizer==3.3.2',
        'docutils==0.21.2',
        'h11==0.14.0',
        'httpcore==1.0.5',
        'httpx==0.27.2',
        'idna==3.8',
        'importlib_metadata==8.4.0',
        'jaraco.classes==3.4.0',
        'jaraco.context==6.0.1',
        'jaraco.functools==4.0.2',
        'keyring==25.3.0',
        'markdown-it-py==3.0.0',
        'mdurl==0.1.2',
        'more-itertools==10.5.0',
        'nh3==0.2.18',
        'numpy==2.1.1',
        'pandas==2.2.2',
        'pkginfo==1.10.0',
        'pydantic==2.9.0',
        'pydantic_core==2.23.2',
        'Pygments==2.18.0',
        'python-dateutil==2.9.0.post0',
        'pytz==2024.1',
        'readme_renderer==44.0',
        'requests==2.32.3',
        'requests-toolbelt==1.0.0',
        'rfc3986==2.0.0',
        'rich==13.8.0',
        'scipy==1.14.1',
        'setuptools==74.1.2',
        'six==1.16.0',
        'sniffio==1.3.1',
        'typing_extensions==4.12.2',
        'tzdata==2024.1',
        'urllib3==2.2.2',
        'zipp==3.20.1',
    ],
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['reverse geocoding', 'geolocation', 'offline', 'python'],
    classifiers=[
        "Development Status :: 4 - Beta",  # Indicating the package is more stable than Alpha
        "Intended Audience :: Developers",  # Broader audience targeting developers
        "Topic :: Software Development :: Libraries :: Python Modules",  # Relevant to Python modules
        "License :: OSI Approved :: MIT License",  # Update based on your license type
        "Programming Language :: Python :: 3",  # General support for Python 3
        "Operating System :: OS Independent",  # If your package works across multiple platforms
    ]
)
