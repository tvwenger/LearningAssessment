from setuptools import find_packages, setup
import versioneer

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="learning_assessment",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A Gaussian Process Model for Student Learning Assessment",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.11",
    license="MIT",
    url="https://github.com/tvwenger/LearningAssessment",
)
