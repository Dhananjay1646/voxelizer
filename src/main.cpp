#include <iostream>

#include <Eigen/Dense>

#include "meshio.hpp"
#include "Voxelizer.hpp"
#include "triangleCube.h"


int main(int argc, char** argv)
{
	if (argc < 4)
	{
		std::cout << "Wrong arguments. Usage: ./voxelizer meshpath.obj voxelpath.bin resolution" << std::endl;
		return -1;
	}

	std::vector<Eigen::Vector3f> verts;
	std::vector<std::vector<int> > faces;
	meshio::loadOBJ(argv[1], verts, faces);
	Voxelizer voxelizer(atoi(argv[3]));
	voxelizer.voxelize(verts, faces);
	voxelizer.save(argv[2]);
	
	return 0;
}
