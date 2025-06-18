import os
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))

def read_readme():
    readme_path = os.path.join(HERE, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name='tennis_ball_tracking',
    version='0.1.0',
    author='Your Name', # TODO: あなたの名前に変更してください
    author_email='you@example.com', # TODO: あなたのメールアドレスに変更してください
    description='A package for tennis ball tracking functionality from the tennisvision project.',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/tennis_ball_tracking', # TODO: プロジェクトのURLに変更してください (任意)
    packages=find_packages(where=HERE, include=['tennis_ball_tracking', 'tennis_ball_tracking.*']),
    install_requires=[
        'numpy==1.23.5',
        'opencv-python==4.6.0.66',
        'torch==1.12.1',
        'torchvision==0.13.1',
        'Pillow==9.2.0',
        'matplotlib==3.5.1',
        'scikit-learn==1.1.2',
        'pandas==1.4.3',
        'ultralytics',
        'hmmlearn'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License', # TODO: プロジェクトのライセンスに合わせて変更してください
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    python_requires='>=3.8',
    include_package_data=True,
)
