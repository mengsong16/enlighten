from pyassimp import load, export
ply_file = "/dataset/replica_v1/frl_apartment_0/mesh.ply"
scene = load(ply_file)  # work with pip install pyassimp==4.1.3, sudo apt-get install libassimp-dev
glb_file = "/dataset/replica_v1/frl_apartment_0/mesh.glb"
export(scene, glb_file) # does not work
print('Done.')
    