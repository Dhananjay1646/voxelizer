#ifndef MESHIO_HPP
#define MESHIO_HPP

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include <Eigen/Dense>

namespace meshio{

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

inline void display(const Eigen::Vector3f& v)
{
	std::cout << v(0) << " " << v(1) << " " << v(2) << std::endl;
}

inline void loadOBJ(const std::string path, std::vector<Eigen::Vector3f>& vertices,
		std::vector<std::vector<int> >& faces)
{
	std::ifstream file(path.c_str());
	std::string line;
	
	while (std::getline(file, line))
	{
		std::vector<std::string> words = split(line, ' ');
		if (words.size() > 1 && words[0] == "v")
		{
			Eigen::Vector3f vertex(std::stof(words[1]), 
					std::stof(words[2]), 
					std::stof(words[3]));
			vertices.push_back(vertex);
		}

		if (words.size() > 1 && words[0] == "f")
		{
			std::vector<int> face;
			for (size_t i = 1; i < words.size(); ++i)
			{
				face.push_back(std::stoi(words[i])-1);
			}
			faces.push_back(face);
		}
	}
}

}

#endif
