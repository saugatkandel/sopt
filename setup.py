from setuptools import setup

setup(
    name='second_order_phase_retrieval',
    version='0.0.1',
    author='Saugat Kandel',
    author_email='saugat.kandel@u.northwestern.edu',
    packages=['optimizers', 'tests'],
    #package_data={'data': ['*.png']},
    scripts=[],
    description='Some second order methods for phase retrieval',
    requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tensorflow",
        "autograd"
    ],
)
