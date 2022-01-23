import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyavia",
    version="0.0.3",
    author="Eric J. Whitney",
    author_email="eric.j.whitney@optusnet.removethispart.com.au",
    description="Useful functions commonly used in aerospace engineering.",
    include_package_data=True,  # <<< Note!
    install_requires=[
        'numpy', 'pexpect'
    ],
    keywords='aerospace engineering tools',
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=["numpy"],
    url="https://github.com/ericjwhitney/pyavia",
    packages=setuptools.find_packages(include=['pyavia', 'pyavia.*']),
    python_requires='>=3.9',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering"
    ]
)
