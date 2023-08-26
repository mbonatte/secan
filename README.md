[![unittest](https://github.com/mbonatte/secan/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/mbonatte/secan/actions/workflows/python-app.yml)

# SecAn - Section Analysis
<i>Section Analysis</i> is a module that enables the modelling and analysis of custom cross-sections. <i>Section Analysis</i> is helpful for the evaluation of section properties and nonlinear response.

You can try the module via <a href="https://colab.research.google.com/drive/1rYkoyhi-yrTdOGnE0fBuhZJfMV5kUh9-?usp=sharing">Google Colab</a>.

# How to import SecAn

To access <i>SecAn</i> and its functions import it in your Python code like this:

```python
import secan as sa
```

We shorten the imported name to sa for better code readability using <i>SecAn</i>.

# How to use SecAn

<h3> Import SecAn </h3>

```python
import secan as sa
```

<h3> Define the Materials </h3>

The first step is to define the materials.

```python
concrete_25 = sa.material.Concrete(fc=25e6)

steel_500   = sa.material.Steel(young=210e9,
                                fy=500e6,
                                ultimate_strain=10e-3)
```

<b>Note:</b> Be aware that <i>SecAn</i> does not apply any partial safety factors, so their application should be done before creating the material.

<h3> Define the Geometries </h3>

The second step is to create concrete geometries.

```python
rect_20x60 = sa.geometry.Rect_section(width=0.2,
                                      height=0.6,
                                      material=concrete_25)
```

<h3> Create the Section </h3>

In this step we create the beam <i>Object</i> with the concrete geometries.

```python
beam = sa.Section(section=[rect_20x60])
```

<h3> Add the Rebas </h3>

Now it is time to add the rebars to the beam.

```python
beam.addSingleRebar(diameter=16e-3,
                    material=steel_500,
                    position=(-.07, -0.27))
beam.addSingleRebar(diameter=16e-3,
                    material=steel_500,
                    position=(-0.03, -0.27))
beam.addSingleRebar(diameter=16e-3,
                    material=steel_500,
                    position=(0.03, -0.27))
beam.addSingleRebar(diameter=16e-3,
                    material=steel_500,
                    position=(.07, -0.27))
```

<h3> Draw cross-section </h3>

Let's check our beam by drawing its cross-section.

```python
beam.plot_section()
```

<img src="/Figures/cross-section.png">

<h3> Plot Moment vs. Curvature </h3>

```python
beam.plot_moment_curvature(k_max=0.025)
```

<img src="/Figures/moment_curvature.png">

<h3> Max Moment </h3>

```python
beam.get_max_moment()
# 211866
```

<h3> Check Section </h3>

We can verify if the beam can support a specific load combination.

```python
e0, k = beam.check_section(normal=0,
                           moment=211866)
# e0: 0.00443
# k:  0.01801
```

<h3> Plot Stress and Strain </h3>

We can check the beam's behaviour by inspecting the stress and strain distribution.

```python
beam.plot_stress_strain(e0,k)
```

<img src="/Figures/stress_strain.png">
