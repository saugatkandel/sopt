from setuptools import setup

setup(
    name='sopt',
    version='v0.1-beta.1',
    author='Saugat Kandel',
    author_email='saugat.kandel@u.northwestern.edu',
    packages=['sopt'],
    #packages=['optimizers', 'tests', 'benchmarks', 'examples'],
    #package_data={'data': ['*.png']},
    scripts=[],
    description='Some second order optimization methods for AD-based optimization',
    requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tensorflow",
        #"autograd"
    ],
)
