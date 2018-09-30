from distutils.core import setup

setup(
    name='resnext',
    version='0.1',
    packages=['ResNeXt'],
    install_requires=['torch', 'tensorboardX', 'pytest', 'numpy'],
    author='Ilya Kaysin',
    author_email='demarkok@gmail.com'
)