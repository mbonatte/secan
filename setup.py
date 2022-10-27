from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name = 'secan',
    version='0.0.1',
    description='Section Analysis - modelling and analysis of custom cross-sections',
    keywords=['reinforced-concrete', 
              'beam-analysis', 
               'concrete-design',
               'cross-section', 
               'structural-engineering', 
               'structural-analysis', 
               'eurocode2', 
               'NBR6118'],
    author = 'Maurício Bonatte',
    author_email='mbonatte@ymail.com',
    url = 'https://github.com/mbonatte/secan',
    license='MIT',
    long_description=long_description,
    
    # Dependencies
    install_requires=['numpy', 
                      'matplotlib'],
    
    # Packaging
    packages =['secan'],
    
)