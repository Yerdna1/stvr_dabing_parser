from setuptools import setup, find_packages

setup(
    name="STVR_DUBBING_PARSE",
    version="0.2",
    packages=find_packages(include=["STVR_DUBBING_PARSE", "STVR_DUBBING_PARSE.agents*"]),
    install_requires=[line.strip() for line in open("requirements.txt")],
    package_dir={
        "STVR_DUBBING_PARSE": "STVR_DUBBING_PARSE",
        "STVR_DUBBING_PARSE.agents": "agents"
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "stvr-dubbing=STVR_DUBBING_PARSE.main:main"
        ]
    }
)
