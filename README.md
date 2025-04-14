# WaymoCar
Gymnasium environment of a MuJoCo car with 4 wheels and differential steering of the front wheels


## Organization

- `car.xml` : The xml file describing the car
- `car.py` : The Gymnasium environment
- `scripted_policy.py` : A scripted policy driving the car between the two obstacles


## Common issues

If you get the error `ValueError: XML Error: top-level default class 'main' cannot be renamed. Element 'default', line 23`, it might be due to a mujoco version. I get this error with `mujoco-3.3.0` but not with `mujoco-3.1.5`.

If you get the error `TypeError: as_quat() got an unexpected keyword argument 'scalar_first'`, then you need to upgrade scipy to at least 1.14.1.


## Acknowledgments

The xml file describing the car is largely based on the rover of [griloHBG](https://github.com/griloHBG/Rover4We)
