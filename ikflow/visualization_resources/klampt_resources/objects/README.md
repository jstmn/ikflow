


# Working with object files and klampt


Klampt can load object files either as Terrains, or as RigidObject ( see `loadRigidObject(self, fn)`, `loadTerrain(self, fn)` in robotsim.py). Calling `loadElement(self, fn)` will load a mesh as a Terrain automatically. Terrains in klampt's vis are tan/brown with checkered squares. This looks fine for the ground, but doesn't look good / realistic for objects, so make sure to load objects as RigidObject. 

Process for creating a rigid body:
1. Go to clara.io and download the object you want to add as a .obj file. Move the file to models/klampt_resources/objects and add a copy to models/klampt_resources/objects/claraio_downloads. This is so if you mess up the file you have a backup 
2. Run `ctmconv <object_file>.obj <object_file>.off`. Copy `srimugsmooth.obj`, and rename it `klampt_<object_file>.obj`. Update the first line to `"<object_file>.off"`












# Objects

1. Mug. From downloaded from https://clara.io/view/e71cd85a-d179-4905-827a-5febbc842987#





# Not imported 

1. Conveyor belt https://clara.io/view/fd7007b6-3499-4e5d-9297-dfdac27195f9
2. Table https://clara.io/view/c162c073-ec50-428c-8e1e-22cd34535ee2