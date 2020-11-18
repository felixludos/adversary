
name = 'adversary'
long_name = 'adversary'

version = '0.1'
url = 'https://github.com/felixludos/adversary'

description = 'Pytorch implementations of various GAN variants using foundation and omni-fig'

author = 'Felix Leeb'
author_email = 'felixludos.info@gmail.com'

license = 'MIT'

readme = 'README.md'

installable_packages = ['adversary']

import os
try:
	with open(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'requirements.txt'), 'r') as f:
		install_requires = f.readlines()
except:
	install_requires = ['omnifig', 'numpy', 'matplotlib', 'torch', 'tensorflow', 'foundation']
del os

