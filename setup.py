from setuptools import setup

setup(
    name='sopt',
    version='0.0.1',
    author='Saugat Kandel',
    author_email='saugat.kandel@u.northwestern.edu',
    packages=['sopt'],
    #packages=['optimizers', 'tests', 'benchmarks', 'examples'],
    #package_data={'data': ['*.png']},
    scripts=[],
    description='Some second order methods for phase retrieval',
    requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tensorflow",
        #"autograd"
    ],
)
