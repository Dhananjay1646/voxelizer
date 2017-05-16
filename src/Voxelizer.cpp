#include "Voxelizer.hpp"

Voxelizer::Voxelizer(const int res) : resolution(res)
{
	data = std::vector< std::vector< std::vector<uint8_t> > >
		(resolution, std::vector< std::vector<uint8_t> >
			(resolution, std::vector<uint8_t> (resolution, 0)));
	voxelDimension = 1.0f/resolution;
}

bool Voxelizer::checkIntersection(Triangle3 base_tri, size_t i, size_t j, size_t k)
{
	Eigen::Vector3f xvec = Eigen::Vector3f::UnitX() * voxelDimension;
	Eigen::Vector3f yvec = Eigen::Vector3f::UnitY() * voxelDimension;
	Eigen::Vector3f zvec = Eigen::Vector3f::UnitZ() * voxelDimension;

	Eigen::Vector3f min_pos (-0.5 + voxelDimension/2.0f, -0.5 + voxelDimension/2.0f, -0.5 + voxelDimension/2.0f);
	Eigen::Vector3f t = min_pos + i * xvec + j * yvec +	k * zvec;
	Triangle3 tri = base_tri.translate(-t).scale(1.0f/voxelDimension);

	if (!t_c_intersection(tri)) return true;
	else return false;
}

int Voxelizer::getGridCoord(float v)
{
	return (int)((v + 0.5f)/voxelDimension);
}

std::vector<std::pair<size_t, size_t> > Voxelizer::getGridInterval(Triangle3 tri)
{
	std::vector<std::pair<size_t, size_t> > interval;
	int xmin, xmax, ymin, ymax, zmin, zmax;

	xmin = getGridCoord(std::min({tri.v1.x, tri.v2.x, tri.v3.x}));
	xmax = getGridCoord(std::max({tri.v1.x, tri.v2.x, tri.v3.x}));

	ymin = getGridCoord(std::min({tri.v1.y, tri.v2.y, tri.v3.y}));
	ymax = getGridCoord(std::max({tri.v1.y, tri.v2.y, tri.v3.y}));

	zmin = getGridCoord(std::min({tri.v1.z, tri.v2.z, tri.v3.z}));
	zmax = getGridCoord(std::max({tri.v1.z, tri.v2.z, tri.v3.z}));

	xmax = std::max(0, std::min(xmax, resolution-1));
	xmin = std::max(0, std::min(xmin, resolution-1));

	ymax = std::max(0, std::min(ymax, resolution-1));
	ymin = std::max(0, std::min(ymin, resolution-1));

	zmax = std::max(0, std::min(zmax, resolution-1));
	zmin = std::max(0, std::min(zmin, resolution-1));

	interval.push_back(std::make_pair(xmin, xmax));
	interval.push_back(std::make_pair(ymin, ymax));
	interval.push_back(std::make_pair(zmin, zmax));

	return interval;
}

void Voxelizer::voxelize(const std::vector<Eigen::Vector3f>& vertices, 
		const std::vector<std::vector<int> >& faces)
{
	for (size_t face_idx = 0; face_idx < faces.size(); ++face_idx)
	{
		std::vector<int> face_idvec = faces[face_idx];
		Triangle3 face = { Point3(vertices[face_idvec[0]]), Point3(vertices[face_idvec[1]]), 
			Point3(vertices[face_idvec[2]]) };
		std::vector<std::pair<size_t, size_t> > interval = getGridInterval(face);
		std::cout << "xmin " << interval[0].first << " | xmax " << interval[0].second << std::endl;
		std::cout << "ymin " << interval[1].first << " | ymax " << interval[1].second << std::endl;
		std::cout << "zmin " << interval[2].first << " | zmax " << interval[2].second << std::endl;

		for (size_t i = interval[0].first; i <= interval[0].second; ++i)
		{
			for (size_t j = interval[1].first; j <= interval[1].second; ++j)
			{
				for (size_t k = interval[2].first; k <= interval[2].second; ++k)
				{
					if (this->checkIntersection(face, i, j, k))
					{
						data[i][j][k] = -1;
					}
				}
			}
		}
	}
	this->fillInterior();
}

std::vector<std::tuple<int, int, int> > 
Voxelizer::getNeighbors(const std::tuple<int, int, int>& p)
{
	std::vector<std::tuple<int, int, int> > neighbors;

	if (std::get<0>(p) > 0)
	{
		std::tuple<int, int, int> n(std::get<0>(p)-1, std::get<1>(p), std::get<2>(p));
		if (data[std::get<0>(n)][std::get<1>(n)][std::get<2>(n)] == 0){
			neighbors.push_back(n);
		}
	}
	if (std::get<0>(p) < resolution-1)
	{
		std::tuple<int, int, int> n(std::get<0>(p)+1, std::get<1>(p), std::get<2>(p));
		if (data[std::get<0>(n)][std::get<1>(n)][std::get<2>(n)] == 0){
			neighbors.push_back(n);
		}
	}

	if (std::get<1>(p) > 0)
	{
		std::tuple<int, int, int> n(std::get<0>(p), std::get<1>(p)-1, std::get<2>(p));
		if (data[std::get<0>(n)][std::get<1>(n)][std::get<2>(n)] == 0){
			neighbors.push_back(n);
		}
	}
	if (std::get<1>(p) < resolution-1)
	{
		std::tuple<int, int, int> n(std::get<0>(p), std::get<1>(p)+1, std::get<2>(p));
		if (data[std::get<0>(n)][std::get<1>(n)][std::get<2>(n)] == 0){
			neighbors.push_back(n);
		}
	}

	if (std::get<2>(p) > 0)
	{
		std::tuple<int, int, int> n(std::get<0>(p), std::get<1>(p), std::get<2>(p)-1);
		if (data[std::get<0>(n)][std::get<1>(n)][std::get<2>(n)] == 0){
			neighbors.push_back(n);
		}
	}
	if (std::get<2>(p) < resolution-1)
	{
		std::tuple<int, int, int> n(std::get<0>(p), std::get<1>(p), std::get<2>(p)+1);
		if (data[std::get<0>(n)][std::get<1>(n)][std::get<2>(n)] == 0){
			neighbors.push_back(n);
		}
	}

	return neighbors;
}

void Voxelizer::fillInterior()
{
	std::unordered_set<std::tuple<int, int, int> > visited;
	std::unordered_set<std::tuple<int, int, int> > open;
	open.insert(std::make_tuple(0, 0, 0));

	while (!open.empty())
	{
		std::tuple<int, int, int> current = *(open.begin());
		std::vector<std::tuple<int, int, int> > neighbors = this->getNeighbors(current);
		for (auto n : neighbors)
			if (visited.find(n) == visited.end())
				open.insert(n);
		visited.insert(current);
		open.erase(current);
	}

	for (size_t i=0; i < data.size(); ++i)
		for (size_t j=0; j < data[i].size(); ++j)
			for (size_t k=0; k < data[i][j].size(); ++k)
				if (visited.find(std::make_tuple(i, j, k)) == visited.end())
					data[i][j][k] = -1;
}

void Voxelizer::writeHeader(const std::string& path)
{
	std::ofstream fs(path, std::ios::out);
	fs << "#binvox 1\n"
	   << "dim " << resolution << " " << resolution << " " << resolution << "\n"
	   << "translate " << 0 << " " << 0 << " " <<  0 << "\n"
	   << "scale " << 1 << "\n"
	   << "data\n";
	fs.close();
}

void Voxelizer::save(const std::string& path)
{
	//this->writeHeader(path);
	std::ofstream fs(path, std::ios::out | std::ios::binary);
	for (size_t i=0; i < data.size(); ++i)
		for (size_t j=0; j < data[i].size(); ++j)
			for (size_t k=0; k < data[i][j].size(); ++k)
				fs.write(reinterpret_cast<const char*>(&data[i][j][k]), sizeof(uint8_t));

	fs.close();
}

