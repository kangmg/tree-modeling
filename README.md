
![christmas_tree (4)](https://github.com/user-attachments/assets/dc2165cf-5fae-48eb-a17e-d9dd837dc84e)

> Tree Growth
```python
from xmastree import TreeConfig, build_christmas_tree, viewer
from ase.io.trajectory import Trajectory

config = TreeConfig()
tree = build_christmas_tree(config)

traj = Trajectory('christmas_tree.traj')
viewer(traj)
```

> Tree Twinklin
```python
from xmastree import create_twinkling_trajectory

frames = create_twinkling_trajectory(
    atoms=traj[-1],
    n_frames=10,
    random_seed=42,
    output_file="twinkling_tree.traj"
)

viewer(frames)
```
