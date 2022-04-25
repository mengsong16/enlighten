from plyfile import PlyData, PlyElement
from pygltflib import GLTF2, Scene

def read_ply():
    plydata = PlyData.read('/dataset/replica_v1/office_3/mesh.ply')
    print("ply loaded.")
    print(plydata)
    return plydata

def read_glb():
    #glb_filename = "/dataset/replica_v1/office_3/habitat/mesh_semantic.glb"
    glb_filename = "/home/meng/habitat-sim/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    glb_data = GLTF2().load(glb_filename) 
    print("glb loaded.")
    print(glb_data)

if __name__ == "__main__":  
    read_glb()

    
