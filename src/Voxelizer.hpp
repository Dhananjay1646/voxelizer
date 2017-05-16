#ifndef VOXELIZER_HPP
#define VOXELIZER_HPP

#include <iostream>
#include <vector>
#include <fstream>
#include <utility>
#include <tuple>
#include <algorithm>
#include <unordered_set>
#include <Eigen/Dense>

#include "triangleCube.h"

typedef std::vector< std::vector< std::vector<uint8_t> > > VoxelGrid;

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

namespace std
{
	template<> struct hash<std::tuple<int, int, int> >
	{
		typedef std::tuple<int, int, int> argument_type;
		typedef std::size_t result_type;
		result_type operator() (argument_type const& s) const
		{
			const result_type h1 (std::hash<int>{}(std::get<0>(s)));
			const result_type h2 (std::hash<int>{}(std::get<1>(s)));
			const result_type h3 (std::hash<int>{}(std::get<2>(s)));

			result_type hash = 0;

			hash_combine(hash, h1);
			hash_combine(hash, h2);
			hash_combine(hash, h3);

			return hash;
		}
	};
}

class Voxelizer
{
	public:
		Voxelizer(const int res);
		void voxelize(const std::vector<Eigen::Vector3f>& vertices, 
				const std::vector<std::vector<int> >& faces);
		void save(const std::string& path);
		int resolution;
		float voxelDimension;
		VoxelGrid data;

	private:
		bool checkIntersection(Triangle3 tri, size_t i, size_t j, size_t k);
		std::vector<std::pair<size_t, size_t> > getGridInterval(Triangle3 tri);
		int getGridCoord(float v);
		void writeHeader(const std::string& path);
		void fillInterior();
		std::vector<std::tuple<int, int, int> > getNeighbors(const std::tuple<int, int, int>& p);
};

#endif
